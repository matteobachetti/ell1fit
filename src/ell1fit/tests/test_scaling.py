"""Tests for parameter scaling and preconditioning.

Local coordinates exist so that every fitted direction has a comparable scale:
the optimizer and the sampler both take steps that are the same size in every
direction, so directions of wildly different natural scale are handled badly.
The convention is that one standard deviation is
:data:`ell1fit.scaling.TARGET_LOCAL_SIGMA` local units.
"""

import dataclasses

import numpy as np
import pytest

from ell1fit.posterior import _build_posterior_functions
from ell1fit.scaling import TARGET_LOCAL_SIGMA, order_of_magnitude, precondition_factors

from .datagen import make_multi_epoch_dataset
from .helpers import build_pipeline_state


@pytest.fixture(scope="module")
def state(tmp_path_factory):
    """Observations and fit setup for a deliberately hard two-epoch fit."""
    outdir = str(tmp_path_factory.mktemp("scaling"))
    # Deliberately a *hard* fit: enough events to have real structure, and a
    # parfile offset so the optimizer has somewhere to travel. On an easy
    # problem both scalings converge perfectly and there is no difference to
    # measure -- an earlier version of this fixture used 4000 events and no
    # offset, and the resulting comparison of two ~1e-5 nat spreads was
    # arbitrary enough to pass locally and fail in CI.
    dataset = make_multi_epoch_dataset(
        outdir,
        epoch_offsets=(0.0, 37.0),
        n_events=12000,
        duration=100_000.0,
        offsets={"A1": 0.01},
        uncertainties={"A1": 1e-1, "F0": 1e-7},
        prefix="scal",
    )
    return build_pipeline_state(dataset, fit_parameters=("F0", "A1"), nharm=2)


def _local_sigma(observations, setup):
    """Local step that lowers the log-posterior by 0.5, per parameter."""
    _, _, func = _build_posterior_functions(observations, setup)
    origin = np.zeros(setup.n_parameters)
    base = func(origin)
    sigmas = []
    for i in range(setup.n_parameters):
        step = 1e-9
        for _ in range(80):
            probe = np.zeros(setup.n_parameters)
            probe[i] = step
            value = func(probe)
            if np.isfinite(value) and base - value > 0.5:
                break
            step *= 1.5
        drop = base - value if np.isfinite(value) else np.nan
        sigmas.append(step * np.sqrt(0.5 / max(drop, 1e-30)) if np.isfinite(drop) else np.nan)
    return np.array(sigmas)


def test_raw_factors_are_badly_conditioned(state):
    """The problem preconditioning exists to solve is real, not hypothetical.

    ``Phase_i`` has no uncertainty recorded anywhere, so ``get_factors`` falls
    through to its default of 1 while every other parameter gets a data-derived
    scale. If this test ever fails because the spread got small on its own, the
    preconditioning step may no longer be needed.
    """
    observations, setup = state
    sigmas = _local_sigma(observations, setup)
    assert np.max(sigmas) / np.min(sigmas) > 50


def test_preconditioning_equalises_the_scales(state):
    """After rescaling, every direction should sit near the target scale."""
    observations, setup = state
    _, _, func = _build_posterior_functions(observations, setup)

    rescaled = dataclasses.replace(
        setup,
        factors=precondition_factors(func, setup.factors, setup.n_parameters),
    )

    before = _local_sigma(observations, setup)
    after = _local_sigma(observations, rescaled)

    assert np.max(after) / np.min(after) < np.max(before) / np.min(before) / 10
    # Every direction should land within a couple of decades of the convention.
    assert np.all(after < 1e3 * TARGET_LOCAL_SIGMA)
    assert np.all(after > 1e-3 * TARGET_LOCAL_SIGMA)


#: Two optima closer than this in log-posterior count as the same one.
SAME_OPTIMUM = 0.01


def test_preconditioning_helps_the_optimizer_find_the_best_optimum(state):
    """The point of the exercise: every start should reach the same optimum.

    The assertion is on the *preconditioned* run reaching one optimum from every
    start, rather than on it beating the raw run by some margin. Those are
    different claims, and only the first is robust: on an easy fit both scalings
    converge and comparing their residual spreads compares two numbers that are
    both essentially zero.

    Measured on this fixture across six data seeds, the raw factors reach the
    best optimum from 1-6 starts out of 8, with spreads of 0.6 to 8 nats; after
    preconditioning it is 8 of 8 every time, with spreads around 1e-5 nats.
    """
    from scipy.optimize import minimize

    from ell1fit.fitting import _bounds_in_local_coordinates

    observations, setup = state
    n_starts = 8
    _, _, func = _build_posterior_functions(observations, setup)
    rescaled = dataclasses.replace(
        setup,
        factors=precondition_factors(func, setup.factors, setup.n_parameters),
    )

    def reached_best(st):
        _, _, f = _build_posterior_functions(observations, st)
        bounds = _bounds_in_local_coordinates(st.baseline_values, st.factors, st.logprior_funcs)
        rng = np.random.default_rng(0)
        values = []
        for k in range(n_starts):
            start = (
                np.zeros(st.n_parameters)
                if k == 0
                else rng.normal(0, 5 * TARGET_LOCAL_SIGMA, st.n_parameters)
            )
            values.append(-minimize(lambda p: -f(p), start, bounds=bounds).fun)
        values = np.array(values)
        return int(np.sum(values > values.max() - SAME_OPTIMUM)), values.max() - values.min()

    n_before, spread_before = reached_best(setup)
    n_after, spread_after = reached_best(rescaled)

    assert n_after == n_starts, (
        f"only {n_after}/{n_starts} starts converged to the same optimum after "
        f"preconditioning (spread {spread_after:.3g} nats)"
    )
    assert spread_after < SAME_OPTIMUM

    # And it must not be a step backwards. The max() lets both pass when the raw
    # scaling already converged, where the comparison would be meaningless.
    assert spread_after <= max(spread_before, SAME_OPTIMUM)
    assert n_after >= n_before


def test_a_flat_direction_keeps_its_factor(state):
    """A parameter with no measurable curvature must not blow the scale up."""
    observations, setup = state

    def flat_posterior(pars):
        return 0.0

    factors = precondition_factors(flat_posterior, setup.factors, setup.n_parameters)
    assert factors == list(setup.factors)


def test_a_non_finite_starting_point_is_survived(state):
    """Preconditioning must degrade gracefully, not raise."""
    observations, setup = state

    def broken(pars):
        return -np.inf

    factors = precondition_factors(broken, setup.factors, setup.n_parameters)
    assert factors == list(setup.factors)


def test_order_of_magnitude_is_one_decade_below():
    """Pins the actual behaviour, including an asymmetry worth knowing about.

    The implementation uses ``int()``, which truncates toward zero rather than
    flooring, so values below 1 come out one decade higher than "one decade
    below" would suggest: 0.05 gives 0.01, not 0.005. Left as-is deliberately --
    it feeds a heuristic starting scale that
    :func:`~ell1fit.scaling.precondition_factors` now measures and corrects, so
    changing it would perturb every result to no benefit.
    """
    assert order_of_magnitude(1234.0) == pytest.approx(100.0)
    assert order_of_magnitude(-1234.0) == pytest.approx(100.0)
    assert order_of_magnitude(1.0) == pytest.approx(0.1)
    assert order_of_magnitude(0.05) == pytest.approx(0.01)


#: Width of the analytic Gaussian used by the curvature tests, in local units.
ANALYTIC_SIGMA = 4e-5


def _analytic_gaussian(peak=0.0, upper_bound=np.inf, lower_bound=-np.inf):
    """A one-dimensional Gaussian log-posterior with known width and optional walls."""

    def logpost(pars):
        x = float(np.asarray(pars, dtype=float)[0])
        if x > upper_bound or x < lower_bound:
            return -np.inf
        return -0.5 * ((x - peak) / ANALYTIC_SIGMA) ** 2

    return logpost


@pytest.mark.parametrize("offset_sigmas", [0.0, 0.3, -0.3, 1.0, -1.0, 3.0, -3.0])
def test_preconditioning_measures_width_not_slope(offset_sigmas):
    """The measured scale must not depend on where the starting point sits.

    A starting point away from the peak is the normal case, not a pathology:
    ``Phase_i`` is centred on a grid whose cells are a full sigma wide, and any
    real parfile starts somewhere near the answer rather than on it. The scale
    that matters is how wide the posterior is, which does not move when the
    starting point does.

    An analytic Gaussian rather than a fitted posterior, so the right answer is
    known exactly and the assertion needs no tolerance for noise. Both signs of
    the offset are covered because the one-sided fall this replaced errs in
    opposite directions either way: probing away from the peak it returns 0.20x
    at 0.3 sigma, 0.11x at 1 sigma and 0.064x at 3 sigma, and probing toward it
    2.0x, 1.6x and 4.0x. Only the centred case, which is the one that never
    happens in practice, comes out right. So this fails loudly if the symmetric
    stencil is ever taken back out.
    """
    (factor,) = precondition_factors(
        _analytic_gaussian(peak=offset_sigmas * ANALYTIC_SIGMA), [1.0], 1
    )
    assert factor == pytest.approx(ANALYTIC_SIGMA / TARGET_LOCAL_SIGMA, rel=1e-6)


@pytest.mark.parametrize("bound_sigmas", [0.03, 0.2, 1.0])
def test_preconditioning_measures_width_against_a_hard_prior_bound(bound_sigmas):
    """A wall on one side must not degrade the estimate.

    ``EPS`` is confined to +-1 and ``A1`` to twice its value, so a symmetric
    probe can land outside the prior's support and come back ``-inf``. The
    three-point stencil on the feasible side removes the linear term just as
    exactly, so the answer should be unchanged.
    """
    (factor,) = precondition_factors(
        _analytic_gaussian(upper_bound=bound_sigmas * ANALYTIC_SIGMA), [1.0], 1
    )
    assert factor == pytest.approx(ANALYTIC_SIGMA / TARGET_LOCAL_SIGMA, rel=1e-6)


def test_preconditioning_keeps_its_factor_when_walls_block_both_sides():
    """Pinned behaviour: an unmeasurable direction keeps the factor it came in with.

    With hard bounds closer than the smallest step that shows any curvature,
    there is no honest estimate to be had. Returning the incoming factor is the
    same graceful degradation as for a flat direction, and is preferred to
    inventing a scale from a blocked probe.
    """
    narrow = _analytic_gaussian(
        upper_bound=0.01 * ANALYTIC_SIGMA, lower_bound=-0.01 * ANALYTIC_SIGMA
    )
    assert precondition_factors(narrow, [7.0], 1) == [7.0]
