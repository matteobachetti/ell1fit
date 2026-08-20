"""Tests for pulse-template evaluation, especially the fast path.

Template evaluation is the hottest code in the package -- once per event, on
every posterior evaluation -- so it is specialized: the sample grid is uniform,
which turns the interpolator's binary search into arithmetic. These tests pin
the property that specialization must preserve, namely that it computes the same
thing as the general-purpose implementation it replaced.

``get_template_func(..., backend="scipy")`` is retained precisely so that claim
remains checkable rather than merely asserted in a comment.
"""

import numpy as np
import pytest

from ell1fit.likelihoods import _pc_like, pletsch_clarke_likelihood
from ell1fit.templates import (
    PARALLEL_TEMPLATE_THRESHOLD,
    _evaluate_uniform_cubic_floored,
    _evaluate_uniform_cubic_floored_parallel,
    get_template_func,
)


def _template(n=200, seed=0):
    """A pulse-shaped template with a little noise, as folding would produce."""
    rng = np.random.default_rng(seed)
    return 1.0 + 0.4 * np.cos(2 * np.pi * np.arange(n) / n) + 0.05 * rng.normal(size=n)


@pytest.mark.parametrize("n_bins", [64, 200])
def test_numba_and_scipy_backends_agree(n_bins):
    """The fast path must reproduce scipy's cubic spline to machine precision.

    Not bit-for-bit: the fast path evaluates each interval's cubic in Taylor
    form about its left edge, scipy in the B-spline basis. Those are the same
    polynomial evaluated by different arithmetic, so they differ in the last
    couple of bits and neither is more correct.
    """
    template = _template(n_bins)
    fast = get_template_func(template, backend="numba")
    reference = get_template_func(template, backend="scipy")

    phases = np.linspace(-2.0, 3.0, 20001)
    got, expected = fast(phases), reference(phases)

    assert np.max(np.abs(got - expected)) < 1e-12
    scale = np.abs(expected) > 1e-6
    assert np.max(np.abs((got[scale] - expected[scale]) / expected[scale])) < 1e-13


def test_phase_is_wrapped_so_any_real_input_is_valid():
    """Evaluation must be periodic: the template is a function of pulse phase."""
    fast = get_template_func(_template())
    base = np.linspace(0.0, 1.0, 501)[:-1]
    for shift in (-3.0, -1.0, 1.0, 7.0):
        assert np.allclose(fast(base), fast(base + shift), rtol=0, atol=1e-12)


def test_parallel_kernel_is_bitwise_identical_to_serial():
    """Threading must change only the wall-clock time, not a single bit.

    Each iteration writes one independent element, so there is no reduction
    whose association could change. This is what licenses dispatching on array
    size without documenting two different answers.
    """
    template = _template()
    fast = get_template_func(template)
    phases = np.random.default_rng(1).uniform(-1.0, 2.0, 250_000)

    serial = _evaluate_uniform_cubic_floored(
        phases, fast.coefficients, fast.x0, fast.dx, fast.n_intervals, 1e-12
    )
    parallel = _evaluate_uniform_cubic_floored_parallel(
        phases, fast.coefficients, fast.x0, fast.dx, fast.n_intervals, 1e-12
    )
    assert np.array_equal(serial, parallel)


def test_size_threshold_does_not_change_the_answer():
    """Results either side of the parallel cutoff must be consistent."""
    template = _template()
    fast = get_template_func(template)
    rng = np.random.default_rng(2)

    small = rng.uniform(0.0, 1.0, 1000)
    below = fast.loglike(small)
    # Same events repeated enough times to cross the threshold: the total must
    # scale exactly, so any discrepancy from the dispatch shows up immediately.
    repeats = int(np.ceil(PARALLEL_TEMPLATE_THRESHOLD / small.size)) + 1
    above = fast.loglike(np.tile(small, repeats))
    assert np.isclose(above, below * repeats, rtol=1e-12, atol=0)


def test_fused_loglike_matches_the_generic_path():
    """The template's own scoring must agree with the generic likelihood code."""
    template = _template()
    fast = get_template_func(template)
    phases = np.random.default_rng(3).uniform(0.0, 1.0, 50_000)

    probs = np.clip(np.asarray(fast(phases), dtype=float), 1e-12, None)
    generic = _pc_like(probs)
    fused = fast.loglike(phases)

    assert abs(fused - generic) / abs(generic) < 1e-12


def test_fused_loglike_is_more_accurate_than_a_running_sum():
    """The summation change is an improvement, not merely a wash.

    A running total accumulates error growing like N; numpy sums pairwise, with
    error growing like log(N). Checked against a high-precision reference so the
    claim rests on a measurement rather than on theory.
    """
    template = _template()
    fast = get_template_func(template)
    phases = np.random.default_rng(4).uniform(0.0, 1.0, 500_000)

    probs = np.clip(np.asarray(fast(phases), dtype=float), 1e-12, None)
    reference = float(np.sum(np.sort(np.log(probs)).astype(np.longdouble)))

    running_total = _pc_like(probs)
    pairwise = fast.loglike(phases)

    assert abs(pairwise - reference) <= abs(running_total - reference)


def test_weighted_loglike_matches_the_generic_path():
    """Weighted scoring must agree with the generic weighted likelihood."""
    template = _template()
    fast = get_template_func(template)
    reference = get_template_func(template, backend="scipy")
    rng = np.random.default_rng(5)
    phases = rng.uniform(0.0, 1.0, 20_000)
    weights = rng.uniform(0.0, 1.0, 20_000)

    fused = pletsch_clarke_likelihood(phases, fast, weights=weights)
    generic = pletsch_clarke_likelihood(phases, reference, weights=weights)
    assert abs(fused - generic) / abs(generic) < 1e-12


def test_unknown_backend_is_rejected():
    """A typo in the backend name must fail loudly, not silently pick one."""
    with pytest.raises(ValueError, match="Unknown template backend"):
        get_template_func(_template(), backend="cubic")
