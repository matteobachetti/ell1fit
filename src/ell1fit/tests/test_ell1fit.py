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


def test_deorbit_is_correct_for_an_event_exactly_at_tasc():
    """An event sitting precisely at TASC must still get the EPS terms applied.

    The iteration seeds its "previous value" and compares against it to decide
    whether to keep going. With a plain sentinel of 0 that comparison was false
    on entry for an event at exactly TASC -- where the first, circular-only
    estimate is also exactly 0 -- so the loop was skipped and the returned time
    was missing the whole EPS2 cos(2 phi) term. That term is at its maximum at
    phi = 0, so the error was maximal precisely where it was silently ignored:
    A1 * EPS2 / 2, here 2.3e-3 s, against a requested tolerance of 1e-8.
    """
    PB, A1, TASC = 218849.0, 22.215, 0.0
    EPS1, EPS2 = 1.5e-4, -2.1e-4
    omega = 2 * np.pi / PB

    times = np.array([0.0, 1.0, 100.0])
    out = simple_ell1_deorbit_numba(times, PB, A1, TASC, EPS1, EPS2, 1e-8)

    # The returned emission time must satisfy the defining equation.
    te = out - TASC
    phase = omega * te
    reconstructed = te + A1 * (
        np.sin(phase) + EPS1 / 2 * np.sin(2 * phase) + EPS2 / 2 * np.cos(2 * phase)
    )
    residual = np.abs(reconstructed + TASC - times)
    assert np.all(residual < 1e-8), f"deorbit did not solve its own equation: {residual}"

    # Specifically at TASC the solution is not zero: it is -A1 * EPS2 / 2.
    assert abs(out[0] - TASC) > 1e-6, "the EPS terms were never applied at TASC"


def test_deorbit_tolerance_actually_tightens_the_solution():
    """A smaller tolerance must produce a demonstrably better solution.

    While the loop was being skipped, the tolerance argument had no effect
    whatsoever -- every value returned an identical, unconverged answer.
    """
    PB, A1, TASC = 218849.0, 22.215, 0.0
    EPS1, EPS2 = 1.5e-4, -2.1e-4
    times = np.linspace(0, 1e5, 2000)

    converged = simple_ell1_deorbit_numba(times, PB, A1, TASC, EPS1, EPS2, 1e-14)
    loose = simple_ell1_deorbit_numba(times, PB, A1, TASC, EPS1, EPS2, 1e-6)
    tight = simple_ell1_deorbit_numba(times, PB, A1, TASC, EPS1, EPS2, 1e-10)

    assert np.max(np.abs(tight - converged)) < np.max(np.abs(loose - converged))


class TestRayleighLimitations:
    """The Rayleigh statistic ignores most of what the pipeline can offer.

    It depends on the event phases and nothing else. That is worth pinning
    down, because the options it cannot honour used to be discarded in silence:
    a user asking for energy weighting got an unweighted fit and no indication
    that anything had been dropped.
    """

    @staticmethod
    def _phases_and_templates():
        from ell1fit.templates import get_template_func

        rng = np.random.default_rng(0)
        phases = np.concatenate([rng.uniform(0, 1, 5000), rng.normal(0.3, 0.05, 2000) % 1.0])
        grid = 2 * np.pi * np.arange(200) / 200
        one_harmonic = get_template_func(1 + 0.3 * np.cos(grid))
        two_harmonics = get_template_func(1 + 0.3 * np.cos(grid) + 0.2 * np.cos(2 * grid))
        return phases, one_harmonic, two_harmonics, rng.uniform(0, 1, phases.size)

    def test_the_template_is_ignored(self):
        from ell1fit.likelihoods import rayleigh_as_likelihood

        phases, one, two, _ = self._phases_and_templates()
        assert rayleigh_as_likelihood(phases, one) == rayleigh_as_likelihood(phases, two)

    def test_weights_are_ignored(self):
        from ell1fit.likelihoods import rayleigh_as_likelihood

        phases, one, _, weights = self._phases_and_templates()
        assert rayleigh_as_likelihood(phases, one) == rayleigh_as_likelihood(
            phases, one, weights=weights
        )

    def test_pletsch_clarke_uses_both(self):
        """The contrast that makes the two assertions above meaningful."""
        from ell1fit.likelihoods import pletsch_clarke_likelihood

        phases, one, two, weights = self._phases_and_templates()
        base = pletsch_clarke_likelihood(phases, one)
        assert pletsch_clarke_likelihood(phases, two) != base
        assert pletsch_clarke_likelihood(phases, one, weights=weights) != base
