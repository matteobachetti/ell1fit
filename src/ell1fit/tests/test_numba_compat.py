"""Tests for the numba compatibility layer.

The package must be importable without numba installed. Running a fit that way
would be unusably slow, but *importing* it must work: building the documentation
introspects every module, and no documentation build should depend on llvmlite
compiling.

The fallback path cannot be reached by simply importing the module in an
environment that has numba, so these tests load a private copy of the shim with
numba blocked, leaving the real one -- and everything already importing from it
-- untouched.
"""

import importlib.util
import sys

import numpy as np
import pytest

import ell1fit._numba_compat as compat


def _load_shim_without_numba():
    """Import a fresh copy of the shim with numba unavailable.

    Uses a distinct module name so the real ``ell1fit._numba_compat`` -- already
    imported and already used by the compiled kernels -- is left alone.
    """
    spec = importlib.util.spec_from_file_location("_shim_no_numba", compat.__file__)
    module = importlib.util.module_from_spec(spec)

    saved = sys.modules.get("numba", "absent")
    # Binding the name to None makes ``from numba import ...`` raise ImportError,
    # which is exactly what the shim is written to handle.
    sys.modules["numba"] = None
    try:
        spec.loader.exec_module(module)
    finally:
        if saved == "absent":
            del sys.modules["numba"]
        else:
            sys.modules["numba"] = saved
    return module


def test_has_numba_reports_the_real_situation():
    """The flag must reflect whether numba was actually imported."""
    assert isinstance(compat.HAS_NUMBA, bool)
    assert compat.HAS_NUMBA is (importlib.util.find_spec("numba") is not None)


def test_fallback_is_selected_when_numba_is_missing():
    shim = _load_shim_without_numba()
    assert shim.HAS_NUMBA is False
    assert shim.prange is range


@pytest.mark.parametrize("parametrised", [False, True])
def test_fallback_njit_returns_the_function_untouched(parametrised):
    """Both decorator spellings must yield the original function.

    This is what distinguishes the shim from a generic mock: a mock replaces the
    function with a mock object and the docstring is lost, which would strip the
    compiled kernels out of the API documentation.
    """
    shim = _load_shim_without_numba()

    def original(x):
        """A docstring that must survive."""
        return x * 2

    decorated = shim.njit(parallel=True)(original) if parametrised else shim.njit(original)

    assert decorated is original
    assert decorated.__doc__ == "A docstring that must survive."
    assert decorated(3) == 6


@pytest.mark.parametrize("parametrised", [False, True])
def test_fallback_vectorize_broadcasts_over_arrays(parametrised):
    """vectorize cannot be a no-op: the decorated functions need broadcasting.

    ``phases_around_zero`` uses scalar ``while`` loops, so returning it unchanged
    would silently give wrong answers on array input rather than failing.
    """
    shim = _load_shim_without_numba()

    def scalar_only(x):
        """Return -1, 0 or 1, using control flow that does not broadcast."""
        if x > 0:
            return 1.0
        if x < 0:
            return -1.0
        return 0.0

    decorated = (
        shim.vectorize([("float64",)])(scalar_only) if parametrised else shim.vectorize(scalar_only)
    )

    result = decorated(np.array([-2.0, 0.0, 3.0]))
    assert np.allclose(result, [-1.0, 0.0, 1.0])


def test_compiled_kernels_agree_with_the_pure_python_fallback():
    """The fallback must compute the same answer, not merely run.

    Only meaningful where numba is actually installed; otherwise both sides are
    the same code and the comparison is vacuous.
    """
    if not compat.HAS_NUMBA:
        pytest.skip("numba is not installed, so there is nothing to compare against")

    from ell1fit.phase_utils import simple_ell1_deorbit_numba

    def pure_python_deorbit(times, PB, A1, TASC, EPS1, EPS2, tolerance=1e-8):
        omega = 2 * np.pi / PB
        out = np.empty_like(times)
        for i in range(times.size):
            t = times[i] - TASC
            out[i] = t - A1 * np.sin(omega * t)
            old = out[i] + 2 * tolerance + 1.0
            n = 0
            while np.abs(out[i] - old) > tolerance and n < 1000:
                old = out[i]
                phase = omega * out[i]
                out[i] = t - A1 * (
                    np.sin(phase) + EPS2 / 2 * np.sin(2 * phase) - EPS1 / 2 * np.cos(2 * phase)
                )
                n += 1
            out[i] += TASC
        return out

    times = np.linspace(0, 1e5, 500)
    args = (218849.0, 22.215, 0.0, 1.5e-4, -2.1e-4)
    compiled = simple_ell1_deorbit_numba(times, *args, 1e-8)
    interpreted = pure_python_deorbit(times, *args, 1e-8)

    assert np.max(np.abs(compiled - interpreted)) < 1e-9
