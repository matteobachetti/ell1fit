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
    outdir = str(tmp_path_factory.mktemp("scaling"))
    dataset = make_multi_epoch_dataset(
        outdir,
        epoch_offsets=(0.0, 37.0),
        n_events=4000,
        duration=100_000.0,
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


def test_preconditioning_helps_the_optimizer_find_the_best_optimum(state):
    """The point of the exercise: reliable convergence from varied starts."""
    from scipy.optimize import minimize

    from ell1fit.fitting import _bounds_in_local_coordinates

    observations, setup = state
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
        for k in range(8):
            start = (
                np.zeros(st.n_parameters)
                if k == 0
                else rng.normal(0, 5 * TARGET_LOCAL_SIGMA, st.n_parameters)
            )
            values.append(-minimize(lambda p: -f(p), start, bounds=bounds).fun)
        values = np.array(values)
        return int(np.sum(values > values.max() - 0.01)), values.max() - values.min()

    n_before, spread_before = reached_best(setup)
    n_after, spread_after = reached_best(rescaled)

    assert n_after >= n_before
    assert spread_after <= spread_before + 1e-9


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
