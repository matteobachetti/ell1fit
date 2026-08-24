"""The weighted fit must quote the phase precision the data actually support.

The Pletsch-Clarke kernel is ``log(1 + w_i (T - 1))``: it hands every event a
copy of the template diluted by that event's own weight. Folding *with* the
weights already produces a once-diluted profile, so pairing the two dilutes it
twice and inflates every phase uncertainty. These tests pin the calibration
down by comparing the achieved posterior width against the width the events can
deliver, with the unweighted fit as the control -- it shares the whole machinery
except the weighting, so if it lands on 1 and the weighted fit does not, the
fault is in the weighting path.
"""

import numpy as np
import pytest

from ..likelihoods import pletsch_clarke_likelihood
from ..pipeline import _undilute_template
from ..templates import create_template_from_profile_harm, get_template_func


NBIN = 32
NHARM = 2


def _draw(seed=0, n_events=200_000, skew=3.0):
    """Events whose pulsed amplitude is strongly skewed across the population.

    ``sum(w) / sum(w**2)`` is what the dilution error costs, and it grows as the
    weight distribution gets more skewed -- most events barely pulsed, a
    minority strongly so, which is the regime weighting exists for.
    """
    rng = np.random.default_rng(seed)
    amplitude = 0.35 * rng.random(n_events) ** skew + 0.005

    phases = np.empty(n_events)
    todo = np.arange(n_events)
    while todo.size:
        trial = rng.random(todo.size)
        keep = rng.random(todo.size) < 0.5 * (1 + amplitude[todo] * np.cos(2 * np.pi * trial))
        phases[todo[keep]] = trial[keep]
        todo = todo[~keep]

    return phases, amplitude / amplitude.max()


def _supported_sigma(phases, weights, nharm=NHARM):
    """Phase precision the events can deliver, from the weighted :math:`Z^2_k`."""
    harmonics = np.arange(1, nharm + 1)
    power = np.array(
        [
            2 * np.abs(np.sum(weights * np.exp(2j * np.pi * k * phases))) ** 2 / np.sum(weights**2)
            for k in harmonics
        ]
    )
    return 1.0 / np.sqrt(np.sum((2 * np.pi * harmonics) ** 2 * power))


def _achieved_sigma(phases, template, weights):
    """Posterior width from the curvature of the log-likelihood in phase.

    Deterministic: no sampler, so the assertions below carry no Monte Carlo
    scatter of their own.
    """
    template_func = get_template_func(template)
    step = 2e-3
    values = np.array(
        [
            pletsch_clarke_likelihood(phases + shift, template_func, weights=weights)
            for shift in (-step, 0.0, step)
        ]
    )
    curvature = (values[0] - 2 * values[1] + values[2]) / step**2
    assert curvature < 0, "log-likelihood is not peaked in phase"
    return 1.0 / np.sqrt(-curvature)


def _template(phases, weights=None):
    profile = np.histogram(phases, bins=np.linspace(0, 1, NBIN + 1), weights=weights)[0]
    template, _ = create_template_from_profile_harm(
        profile, nharm=NHARM, final_nbin=200, plot=False
    )
    return template


def _calibration(seed):
    phases, weights = _draw(seed)
    uniform = np.ones_like(weights)
    weighted = _template(phases, weights)
    return {
        "unweighted": (
            _achieved_sigma(phases, _template(phases), uniform) / _supported_sigma(phases, uniform)
        ),
        "diluted": (_achieved_sigma(phases, weighted, weights) / _supported_sigma(phases, weights)),
        "undiluted": (
            _achieved_sigma(phases, _undilute_template(weighted, weights), weights)
            / _supported_sigma(phases, weights)
        ),
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_fit_is_as_well_calibrated_as_the_unweighted_one(seed):
    """The weighted fit must meet the tolerance the unweighted fit already does."""
    ratios = _calibration(seed)

    assert ratios["unweighted"] == pytest.approx(1.0, abs=0.05)
    assert ratios["undiluted"] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pairing_the_weighted_fold_with_the_weights_inflates_the_error_bars(seed):
    """Guard the reason :func:`_undilute_template` exists.

    Without it the template is diluted once by the fold and again by the
    likelihood, so the quoted phase uncertainty comes out far too wide. If this
    ever stops being true the correction has become a no-op and the test above
    would pass for the wrong reason.
    """
    ratios = _calibration(seed)

    assert ratios["diluted"] > 1.2
    assert ratios["undiluted"] < ratios["diluted"]


def test_undilution_makes_the_fit_invariant_to_the_weight_scale():
    """Rescaling all weights must not move the fit -- the docstring's claim.

    The undiluted template's modulation carries a compensating ``1 / sum(w^2)``,
    so ``w_i (T - 1)`` is unchanged under ``w -> s w``. Without the correction
    the error bars would track the scale.
    """
    phases, weights = _draw(seed=0)
    reference = None
    for scale in (0.25, 0.5, 1.0):
        scaled = weights * scale
        template = _undilute_template(_template(phases, scaled), scaled)
        sigma = _achieved_sigma(phases, template, scaled)
        if reference is None:
            reference = sigma
        assert sigma == pytest.approx(reference, rel=2e-3)


def test_a_negative_template_is_not_blocked_by_the_likelihood():
    """Undiluting can push ``T`` below zero where ``1 + w (T - 1)`` is still safe.

    The floor in the likelihood belongs on the mixture, not on the template: a
    template that dips negative at the trough of a strong pulse is legitimate,
    and clamping it there would silently flatten the model.
    """
    phases = np.linspace(0, 1, 5000, endpoint=False)
    # Modulation deeper than the mean level: this template is genuinely negative
    # over part of the cycle.
    template = 1 + 1.6 * np.cos(2 * np.pi * np.linspace(0, 1, 200, endpoint=False))
    assert template.min() < 0

    weights = np.full(phases.size, 0.5)
    mixture_floor = 1 + weights.max() * (template.min() / template.mean() - 1)
    assert mixture_floor > 0, "the mixture itself must stay positive for this test"

    template_func = get_template_func(template)
    fused = template_func.loglike(phases, weights=weights)
    generic = pletsch_clarke_likelihood(phases, lambda p: template_func(p), weights=weights)

    assert np.isfinite(fused)
    assert fused == pytest.approx(generic, rel=1e-9)
    # A clamped template would lose the trough entirely and score differently.
    clamped = get_template_func(np.clip(template, 0, None))
    assert fused != pytest.approx(clamped.loglike(phases, weights=weights), rel=1e-6)
