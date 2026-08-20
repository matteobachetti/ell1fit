import os

import pytest
import numpy as np
from astropy.table import Table
from ..phase_utils import NonInvertibleOrbitError, _calculate_phases
from ..phase_utils import add_circular_orbit_numba, add_ell1_orbit_numba
from ..phase_utils import orbit_is_invertible
from ..phase_utils import simple_circular_deorbit_numba, simple_ell1_deorbit_numba
from ..results_io import safe_save


@pytest.mark.parametrize("A1", [1, 10, 40])
@pytest.mark.parametrize("PB", [0.3, 3, 30])
def test_circular_orbit(PB, A1):
    A1 /= 86400
    times = np.random.uniform(56000, 59000, 10)
    TASC = np.random.uniform(56000, 59000)

    orbited = add_circular_orbit_numba(times, PB, A1, TASC)
    deorbited = simple_circular_deorbit_numba(orbited, PB, A1, TASC, tolerance=1e-8)
    assert np.all(np.abs(deorbited - times) < 1e-8)


@pytest.mark.parametrize("A1", [10, 40])
@pytest.mark.parametrize("PB", [0.3, 3])
@pytest.mark.parametrize("E1", [0.0001, 0.1])
@pytest.mark.parametrize("E2", [0.0001, 0.1])
def test_ell1_orbit(PB, A1, E1, E2):
    A1 /= 86400
    times = np.random.uniform(56000, 59000, 10)
    TASC = np.random.uniform(56000, 59000)

    orbited = add_ell1_orbit_numba(times, PB, A1, TASC, E1, E2)
    deorbited = simple_ell1_deorbit_numba(orbited, PB, A1, TASC, E1, E2, tolerance=1e-8)
    assert np.all(np.abs(deorbited - times) < 1e-8)


def test_safe_save():
    results = Table({"a": [2]})
    results_2 = Table({"a": ["3"]})
    output_file = "blabla.csv"
    safe_save(results, output_file)
    safe_save(results_2, output_file)
    with pytest.warns(UserWarning, match="Merging old and new"):
        safe_save(results_2, output_file)
    os.unlink(output_file)
    os.unlink("old_" + output_file)


@pytest.mark.parametrize("A1", [4e4, -8.7e4])
def test_deorbit_terminates_outside_the_invertible_region(A1):
    """Superluminal projected motion must not hang the deorbiting iteration.

    The fixed-point map ``t -> t_obs - A1 sin(omega t)`` contracts only while
    the projected orbital velocity ``A1 * omega`` is below c. Beyond it the map
    has no fixed point, and before the iteration cap was added this loop ran
    forever -- inside an njit(parallel=True) kernel, so not even interruptible.
    An optimizer probing a wild trial position reaches this in practice.
    """
    PB = 218849.0
    times = np.linspace(0, 60000, 200)

    assert not orbit_is_invertible(PB, A1, 1e-4, 1e-4)
    # The assertion that matters is simply that this returns at all.
    out = simple_ell1_deorbit_numba(times, PB, A1, 0.0, 1e-4, 1e-4, 1e-8)
    assert out.shape == times.shape


def test_orbit_is_invertible_tracks_projected_velocity():
    """The invertibility bound is the subluminal condition on A1 * omega."""
    PB = 218849.0
    # A real pulsar sits far inside the limit.
    assert orbit_is_invertible(PB, 22.215, 1e-4, 1e-4)
    # Just below and just above v = c.
    assert orbit_is_invertible(PB, 0.99 * PB / (2 * np.pi), 0.0, 0.0)
    assert not orbit_is_invertible(PB, 1.01 * PB / (2 * np.pi), 0.0, 0.0)
    # Degenerate inputs are not invertible either.
    assert not orbit_is_invertible(0.0, 1.0)
    assert not orbit_is_invertible(PB, np.inf)


def test_phases_reject_non_invertible_orbits():
    """_calculate_phases must refuse, not grind, on superluminal parameters."""
    parameters = {
        "PB": np.float64(218849.0),
        "A1": np.float64(-8.7e4),
        "TASC": np.float64(56682.0),
        "EPS1": np.float64(0.0),
        "EPS2": np.float64(0.0),
        "F0_0": np.float64(7.5),
        "PEPOCH_0": np.float64(56682.0),
        "Phase_0": np.float64(0.0),
    }
    with pytest.raises(NonInvertibleOrbitError, match="not invertible"):
        _calculate_phases([np.linspace(0, 60000, 100)], parameters)


def test_posterior_rejects_non_invertible_orbits_as_impossible():
    """The posterior must return -inf there, since a Gaussian prior cannot.

    A Gaussian log-prior at a wild A1 is hugely negative but *finite*, so the
    existing ``isinf(lp)`` early-out never fires and evaluation proceeds into
    the deorbiting iteration. This is the guard that stops it.
    """
    from ell1fit.likelihoods import pletsch_clarke_likelihood
    from ell1fit.posterior import _build_posterior_functions
    from ell1fit.setup_types import FitSetup, ObservationSet

    times = [np.linspace(0, 60000, 100)]
    parameters = {
        "PB": np.float64(218849.0),
        "A1": np.float64(22.215),
        "TASC": np.float64(56682.0),
        "EPS1": np.float64(0.0),
        "EPS2": np.float64(0.0),
        "F0_0": np.float64(7.5),
        "PEPOCH_0": np.float64(56682.0),
        "Phase_0": np.float64(0.0),
    }
    observations = ObservationSet(
        files=["synthetic"], models=[None], ref_model=None, pepoch=[56682.0],
        times_from_pepoch=times, energies=[None], exposures=np.array([60000.0]),
        observation_length=np.array([60000.0]),
    )
    setup = FitSetup(
        parameter_names=["A1"],
        baseline_values=[22.215],
        # A Gaussian prior: finite everywhere, so it can never reject anything.
        logprior_funcs=[lambda x: -0.5 * ((x - 22.215) / 0.1) ** 2],
        factors=[1.0],
        template_funcs=[lambda ph: np.ones_like(np.asarray(ph, dtype=float))],
        parameters=parameters,
        likelihood_func=pletsch_clarke_likelihood,
    )
    _, _, func_to_maximize = _build_posterior_functions(observations, setup)

    assert np.isfinite(func_to_maximize([0.0])), "the starting point should be valid"
    # A1 driven far past the subluminal limit.
    assert func_to_maximize([-8.7e4]) == -np.inf
