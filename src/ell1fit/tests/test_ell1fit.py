"""Unit tests for the orbital kernels, phase conventions, and result I/O."""

import os
import re

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
    """Applying a circular orbital delay and removing it must be a round trip."""
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
    """Same round trip with the ELL1 eccentricity terms included.

    The forward model is closed-form; the inverse is solved iteratively, so this
    checks the iteration actually converges to the value it was given.
    """
    A1 /= 86400
    times = np.random.uniform(56000, 59000, 10)
    TASC = np.random.uniform(56000, 59000)

    orbited = add_ell1_orbit_numba(times, PB, A1, TASC, E1, E2)
    deorbited = simple_ell1_deorbit_numba(orbited, PB, A1, TASC, E1, E2, tolerance=1e-8)
    assert np.all(np.abs(deorbited - times) < 1e-8)


def test_safe_save():
    """Results accumulate across runs, and a schema clash warns instead of losing data."""
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
        files=["synthetic"],
        models=[None],
        ref_model=None,
        pepoch=[56682.0],
        times_from_pepoch=times,
        energies=[None],
        exposures=np.array([60000.0]),
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
    was missing the whole EPS1 cos(2 phi) term. That term is at its maximum at
    phi = 0, so the error was maximal precisely where it was silently ignored:
    A1 * EPS1 / 2, here 1.7e-3 s, against a requested tolerance of 1e-8.
    """
    PB, A1, TASC = 218849.0, 22.215, 0.0
    EPS1, EPS2 = 1.5e-4, -2.1e-4
    omega = 2 * np.pi / PB

    times = np.array([0.0, 1.0, 100.0])
    out = simple_ell1_deorbit_numba(times, PB, A1, TASC, EPS1, EPS2, 1e-8)

    # The returned emission time must satisfy the defining equation.
    te = out - TASC
    phase = omega * te
    reconstructed = te + A1 * _tempo_dre_over_x(phase, EPS1, EPS2)
    residual = np.abs(reconstructed + TASC - times)
    assert np.all(residual < 1e-8), f"deorbit did not solve its own equation: {residual}"

    # Specifically at TASC the solution is not zero: to leading order it is
    # +A1 * EPS1 / 2 (the o(e^2) terms shift it by a further 0.02% here). The
    # converged value sits ~0.06% below that, because the
    # circular sin term is no longer exactly zero once the EPS terms have
    # moved the solution off TASC -- hence the loose tolerance on a check whose
    # point is the sign and the order of magnitude, not the last digit.
    assert out[0] - TASC == pytest.approx(A1 * EPS1 / 2, rel=1e-2), (
        "the EPS terms were never applied at TASC"
    )


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


def test_mjd_to_sec_accepts_plain_floats_and_arrays():
    """Conversion must not depend on being handed numpy types.

    The original implementation called ``.astype`` on its result, so a plain
    Python float raised ``AttributeError``. Nothing in the pipeline hit it,
    because PINT returns ``np.float64`` -- but any code building a parameter
    dictionary by hand did.
    """
    from ell1fit.phase_utils import _mjd_to_sec

    assert _mjd_to_sec(56682.5, 56682.0) == pytest.approx(43200.0)
    assert _mjd_to_sec(np.float64(56682.5), np.float64(56682.0)) == pytest.approx(43200.0)
    assert _mjd_to_sec(np.float64(56682.5), 56682.0) == pytest.approx(43200.0)
    assert np.allclose(_mjd_to_sec(np.array([56682.5, 56683.0]), 56682.0), [43200.0, 86400.0])


def test_phases_are_flat_in_pbdot():
    """The phase model must be provably independent of ``PBDOT``.

    This is the fact that justifies rejecting ``-P PBDOT`` in
    :func:`ell1fit.pipeline._reject_unfittable_parameters`: ``_calculate_phases``
    passes a constant ``PB`` to the deorbiting kernel, so the likelihood has no
    gradient in ``PBDOT`` and fitting it would return the prior.

    If ``PBDOT`` ever enters the phase model for real, this test fails -- which
    is the point. The guard must then be removed rather than left to reject a
    parameter that has become fittable.
    """
    parameters = {
        "PB": 30000.0,
        "A1": 1.0,
        "TASC": 57000.0,
        "EPS1": 1e-4,
        "EPS2": -2e-4,
        "PBDOT": 0.0,
        "PEPOCH_0": 57000.0,
        "F0_0": 100.0,
        "Phase_0": 0.0,
    }
    times = [np.linspace(0, 1e6, 1000)]

    reference = _calculate_phases(times, parameters)[0]
    for pbdot in (1e-12, 1e-10, 1e-8):
        perturbed = _calculate_phases(times, dict(parameters, PBDOT=pbdot))[0]
        assert np.array_equal(perturbed, reference), (
            f"PBDOT={pbdot} changed the phases: it is no longer an inert parameter"
        )


def _kepler_delay_over_x(phi, e, om):
    """Exact Keplerian Roemer delay divided by ``x``, ``TASC`` at the ascending node.

    Solves ``M = E - e sin E`` by Newton iteration and evaluates
    ``(r/a) sin(omega + nu) = sin(omega)(cos E - e)
    + cos(omega) sqrt(1 - e^2) sin E``. No approximation in ``e``, and nothing
    imported from the code under test.
    """
    M = phi - om
    E = M.copy()
    for _ in range(80):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return np.sin(om) * (np.cos(E) - e) + np.cos(om) * np.sqrt(1 - e**2) * np.sin(E)


def test_pint_defines_eps1_as_e_sin_omega(tmp_path):
    """Pin the convention on PINT's side of the boundary too.

    The package reads ``EPS1``/``EPS2`` straight out of a PINT-parsed parfile,
    so the delay above is only right if PINT means by them what the delay
    assumes: ``EPS1 = e sin(omega)``, ``EPS2 = e cos(omega)``. PINT derives
    ``ECC`` and ``OM`` from them, which states its convention without needing
    TOAs or an ephemeris.
    """
    from pint.models import get_model

    eps1, eps2 = 0.003, 0.004
    source = open(os.path.join(os.path.dirname(__file__), "data", "events0.par")).read()
    source = re.sub(r"^EPS1\s.*$", f"EPS1 {eps1}", source, flags=re.M)
    source = re.sub(r"^EPS2\s.*$", f"EPS2 {eps2}", source, flags=re.M)
    parfile = tmp_path / "eccentric.par"
    parfile.write_text(source)

    model = get_model(str(parfile))

    assert float(model.ECC.value) == pytest.approx(np.hypot(eps1, eps2))
    assert float(model.OM.value) == pytest.approx(np.degrees(np.arctan2(eps1, eps2)))


def _delay_over_x(kernel, times, PB, A1, eps1, eps2):
    """Roemer delay divided by ``A1``, without the ``times + delay`` round trip.

    Evaluating at ``times = 0`` with ``TASC = -t`` gives the same phase while
    keeping the result small, so subtracting ``times`` back off does not throw
    away twelve digits to cancellation. Measured through the round trip the
    kernels agree with their reference to 6.5e-13; measured this way, 1e-15.
    """
    return kernel(np.zeros_like(times), PB, A1, -times, eps1, eps2) / A1


def _tempo_dre_over_x(phi, eps1, eps2):
    """``dre / x`` transcribed from tempo's ``bnryell1.f``, first and second order."""
    return (np.sin(phi) - 0.5 * (eps1 * np.cos(2 * phi) - eps2 * np.sin(2 * phi))) + (-1 / 8.0) * (
        -2 * eps1 * eps2 * np.cos(phi)
        + 6 * eps1 * eps2 * np.cos(3 * phi)
        + 3 * eps1 * eps1 * np.sin(phi)
        + 5 * eps2 * eps2 * np.sin(phi)
        + 3 * eps1 * eps1 * np.sin(3 * phi)
        - 3 * eps2 * eps2 * np.sin(3 * phi)
    )


@pytest.mark.parametrize("e,om", [(1e-3, 1.1), (1e-2, 0.3), (0.1, 2.7)])
def test_delay_matches_the_tempo_expression(e, om):
    """The o(e^2) kernel must reproduce ``bnryell1.f``'s ``dre`` exactly.

    The kernel evaluates the 2Phi and 3Phi harmonics from ``sin(Phi)`` and
    ``cos(Phi)`` through multiple-angle identities, so this also checks that the
    rearrangement is exact rather than merely close.
    """
    PB, A1 = 218849.0, 22.215
    times = np.linspace(0, PB, 4001, endpoint=False)
    phi = 2 * np.pi * times / PB
    eps1, eps2 = e * np.sin(om), e * np.cos(om)

    got = _delay_over_x(add_ell1_orbit_numba, times, PB, A1, eps1, eps2)
    assert np.max(np.abs(got - _tempo_dre_over_x(phi, eps1, eps2))) < 1e-14


@pytest.mark.parametrize("om", [0.0, 0.7, 2.5, -1.3])
def test_ell1_delay_matches_an_exact_keplerian_orbit(om):
    """The modelled delay must track an exact Keplerian orbit to third order.

    This pins two things at once: that ``EPS1``/``EPS2`` pair onto the harmonics
    the way tempo defines them -- mispairing them rotates the orbit by 90 degrees
    in ``omega`` and leaves a residual falling only as ``e`` -- and that the
    Wex--Zhu o(e^2) block is present, without which the residual falls as
    ``e**2``. Measured at omega = 1.1: 6.7e-10, 6.7e-07, 6.8e-04 for
    e = 1e-3, 1e-2, 1e-1.
    """
    PB, A1 = 218849.0, 22.215
    times = np.linspace(0, PB, 4001, endpoint=False)
    phi = 2 * np.pi * times / PB

    residuals = {}
    for e in (1e-3, 1e-2, 1e-1):
        eps1, eps2 = e * np.sin(om), e * np.cos(om)
        exact = _kepler_delay_over_x(phi, e, om)
        got = _delay_over_x(add_ell1_orbit_numba, times, PB, A1, eps1, eps2)
        residual = got - exact
        residual -= residual.mean()
        residuals[e] = np.max(np.abs(residual))

        # Dropping the o(e^2) block leaves ~0.77 e^2, and mispairing the
        # parameters leaves ~0.7 e; both exceed this bound at every e.
        assert residuals[e] < 2 * e**3, (
            f"omega={om}, e={e}: residual {residuals[e]:.3e} is not third order"
        )

    for smaller, larger in ((1e-3, 1e-2), (1e-2, 1e-1)):
        ratio = residuals[larger] / residuals[smaller]
        assert ratio == pytest.approx(1000, rel=0.2), (
            f"omega={om}: residual scales as e^{np.log10(ratio):.2f}, not e^3"
        )


def test_ell1_reduces_to_a_circular_orbit():
    """With no eccentricity the higher-order kernel must be exactly circular."""
    PB, A1, TASC = 218849.0, 22.215, 56682.0
    times = np.linspace(56682.0, 56684.6, 3000)

    assert np.array_equal(
        add_ell1_orbit_numba(times, PB, A1 / 86400, TASC, 0.0, 0.0),
        add_circular_orbit_numba(times, PB, A1 / 86400, TASC),
    )


@pytest.mark.parametrize("e", [1e-3, 1e-2, 5e-2])
def test_deorbit_inverts_its_forward_model_at_higher_eccentricity(e):
    """The iterative inverse must undo the closed-form forward model."""
    PB, A1, TASC, om = 218849.0, 22.215, 0.0, 0.3
    times = np.linspace(0, PB, 2001, endpoint=False)
    eps1, eps2 = e * np.sin(om), e * np.cos(om)

    orbited = add_ell1_orbit_numba(times, PB, A1, TASC, eps1, eps2)
    recovered = simple_ell1_deorbit_numba(orbited, PB, A1, TASC, eps1, eps2, tolerance=1e-12)

    assert np.max(np.abs(recovered - times)) < 1e-9


def test_unknown_fit_parameters_are_rejected():
    """A requested parameter the model does not carry must raise, not vanish.

    ``_collect_parameter_names`` walks the parameter dictionary and keeps the
    entries a requested name matches. A name that matches nothing used to be
    dropped in silence, so ``-P F0,A1DOT`` fitted ``F0`` alone and reported a
    successful two-parameter run -- the same silence a typo like ``-P TSAC``
    bought. Nothing downstream could notice: the fit is internally consistent,
    it is simply not the fit that was asked for.
    """
    from ..pipeline import _collect_parameter_names
    from ..likelihoods import pletsch_clarke_likelihood

    parameters = {
        "PB": 1.0,
        "A1": 1.0,
        "TASC": 1.0,
        "EPS1": 0.0,
        "EPS2": 0.0,
        "F0_0": 1.0,
        "F1_0": 0.0,
        "PEPOCH_0": 57000.0,
        "Phase_0": 0.0,
    }

    collected = _collect_parameter_names(parameters, ["F0", "F1", "A1"], pletsch_clarke_likelihood)
    assert set(collected) == {"A1", "F0_0", "F1_0", "Phase_0"}

    with pytest.raises(ValueError, match="TSAC"):
        _collect_parameter_names(parameters, ["F0", "TSAC"], pletsch_clarke_likelihood)

    with pytest.raises(ValueError, match="A1DOT"):
        _collect_parameter_names(parameters, ["F0", "A1DOT"], pletsch_clarke_likelihood)


def test_per_file_parameters_are_requested_by_their_bare_name():
    """``Phase`` and ``F0`` name per-file entries and must not be rejected.

    The parameter dictionary holds ``F0_0``/``Phase_0``, never a bare ``F0`` or
    ``Phase``, so the check for unknown names cannot simply look for an exact
    key or it would reject every legitimate request.
    """
    from ..pipeline import _collect_parameter_names
    from ..likelihoods import pletsch_clarke_likelihood

    parameters = {"A1": 1.0, "F0_0": 1.0, "F0_1": 1.0, "Phase_0": 0.0, "Phase_1": 0.0}

    collected = _collect_parameter_names(parameters, ["F0", "Phase"], pletsch_clarke_likelihood)
    assert "F0_0" in collected and "F0_1" in collected


def test_a1dot_survives_a_round_trip_through_a_parfile(tmp_path):
    """A fitted ``A1DOT`` must read back as the value that was written.

    PINT implements the parfile convention that ``A1DOT 7.2`` means 7.2e-12 in
    the *assignment*: a bare float above 1e-7 is multiplied by 1e-12 on the way
    in, so ``model.A1DOT.value = x`` is not the identity. ``update_model`` has
    to assign a Quantity, which takes the units-carrying branch and does not
    rescale. The same trap bit ``PBDOT`` on the ``ell1decay`` side.
    """
    from pint.models import get_model
    from ..create_parfile import update_model

    datadir = os.path.join(os.path.dirname(__file__), "data")
    parfile = str(tmp_path / "in.par")
    open(parfile, "w").write(open(os.path.join(datadir, "events0.par")).read())

    a1dot = 3.5e-12  # lt-s/s: the scale a real measurement lands on
    row = {
        "PEPOCH": get_model(parfile).PEPOCH.value,
        "dA1DOT_mean": a1dot,
        "dA1DOT_50": a1dot,
        "dA1DOT_16": a1dot - 1e-13,
        "dA1DOT_84": a1dot + 1e-13,
        "dA1DOT_initial": 0.0,
        "dA1DOT_factor": 1.0,
    }

    new_model = update_model(get_model(parfile), row, include_info=False)
    assert new_model.A1DOT.value == pytest.approx(a1dot, rel=1e-12)

    written = str(tmp_path / "out.par")
    with open(written, "w") as fobj:
        fobj.write(new_model.as_parfile(include_info=False))
    assert get_model(written).A1DOT.value == pytest.approx(a1dot, rel=1e-9)


def test_a1dot_above_the_parfile_threshold_is_written_faithfully(tmp_path):
    """The fitted value must reach the model even where a parfile cannot hold it.

    Above 1e-7 the parfile format is genuinely ambiguous -- PINT will read the
    number back rescaled by 1e-12, and nothing on the writing side can prevent
    that -- but the value ``update_model`` puts into the model must still be
    the fitted one, and the user must be told. No physical drift comes near
    1e-7; a fit that wandered into the tail of its own prior does, since the
    default ``A1DOT`` prior runs out to ``|A1| * 2pi / PB``, about 6e-4 for
    M82 X-2.
    """
    import logging as _logging

    from pint.models import get_model
    from ..create_parfile import update_model

    datadir = os.path.join(os.path.dirname(__file__), "data")
    parfile = str(tmp_path / "in.par")
    open(parfile, "w").write(open(os.path.join(datadir, "events0.par")).read())
    model = get_model(parfile)

    a1dot = 3.5e-6
    row = {
        "PEPOCH": model.PEPOCH.value,
        "dA1DOT_mean": a1dot,
        "dA1DOT_50": a1dot,
        "dA1DOT_16": a1dot - 1e-8,
        "dA1DOT_84": a1dot + 1e-8,
        "dA1DOT_initial": 0.0,
        "dA1DOT_factor": 1.0,
    }

    logger = _logging.getLogger()
    records = []

    class _Collect(_logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collect()
    logger.addHandler(handler)
    try:
        new_model = update_model(model, row, include_info=False)
    finally:
        logger.removeHandler(handler)

    assert new_model.A1DOT.value == pytest.approx(a1dot, rel=1e-12)
    assert any(
        record.levelno >= _logging.WARNING and "A1DOT" in record.getMessage() for record in records
    ), "writing an A1DOT a parfile cannot represent must warn"


def test_a1dot_without_a_lever_arm_is_rejected():
    """Fitting ``A1DOT`` from one epoch must raise, not sample the prior.

    ``A1DOT`` enters the phase model only as ``A1DOT * binary_dt_i``. A single
    file *is* the reference epoch, so every lever arm is zero, the likelihood
    is exactly flat in ``A1DOT``, and a chain would report the prior as a
    measurement -- the same failure ``UNFITTABLE_PARAMETERS`` refuses ``PBDOT``
    for, arrived at from the data rather than from the model.
    """
    from ..pipeline import _reject_unmeasurable_derivatives

    one_epoch = {"A1DOT": 0.0, "binary_dt_0": 0.0}
    two_epochs = {"A1DOT": 0.0, "binary_dt_0": -3.0e7, "binary_dt_1": 3.0e7}

    with pytest.raises(ValueError, match="A1DOT"):
        _reject_unmeasurable_derivatives(one_epoch, ["A1", "A1DOT"])

    # Two epochs at the same PEPOCH are no better than one.
    with pytest.raises(ValueError, match="A1DOT"):
        _reject_unmeasurable_derivatives(
            {"A1DOT": 0.0, "binary_dt_0": 0.0, "binary_dt_1": 0.0}, ["A1DOT"]
        )

    _reject_unmeasurable_derivatives(two_epochs, ["A1", "A1DOT"])
    _reject_unmeasurable_derivatives(one_epoch, ["A1"])


def test_jax_posterior_follows_a1dot_like_the_numba_one(tmp_path):
    """The JAX rebuild must apply ``A1DOT`` exactly as the numba phase model does.

    :mod:`ell1fit.nuts_sampling` is a reimplementation, not a wrapper, so every
    parameter that enters the phase model has to be added to it by hand -- and
    an omission is silent: the gradient sampler would simply sample a posterior
    that is flat in the missing direction and report a confident prior.

    Probing along ``A1DOT`` alone, rather than jittering every parameter, is
    what makes this test about ``A1DOT``. The displacements are several sigma
    of the achievable precision, which the second assertion checks actually
    move the log-posterior -- otherwise agreement would be the agreement of two
    functions that both ignore it.
    """
    pytest.importorskip("jax")

    from .datagen import InjectedSolution, make_multi_epoch_dataset
    from .helpers import build_pipeline_state
    from ..nuts_sampling import build_jax_logpost, enable_x64
    from ..posterior import _build_posterior_functions

    enable_x64()
    dataset = make_multi_epoch_dataset(
        str(tmp_path),
        solution=InjectedSolution(A1DOT=6e-10),
        epoch_offsets=(0.0, 700.0),
        n_events=800,
        prefix="jaxa1dot",
    )
    observations, setup = build_pipeline_state(dataset, fit_parameters=["F0", "A1", "A1DOT"])
    _, _, numba_logpost = _build_posterior_functions(observations, setup)
    jax_logpost = build_jax_logpost(observations, setup)

    from ..scaling import TARGET_LOCAL_SIGMA

    index = list(setup.parameter_names).index("A1DOT")
    values = []
    for offset in (-4.0, -2.0, 0.0, 2.0, 4.0):
        position = np.zeros(len(setup.parameter_names))
        # One *local* unit is about a million sigma -- the scaling puts one
        # sigma at TARGET_LOCAL_SIGMA -- so probe in sigma, not in local units.
        position[index] = offset * TARGET_LOCAL_SIGMA
        reference = float(numba_logpost(position))
        rebuilt = float(jax_logpost(position))
        values.append(reference)
        assert abs(reference - rebuilt) < 1e-4, (
            f"JAX and numba disagree by {reference - rebuilt:.3g} at A1DOT offset {offset}"
        )

    assert max(values) - min(values) > 1.0, (
        f"the log-posterior barely moved over the probed A1DOT range ({values}): "
        "this comparison would pass even if both paths ignored A1DOT"
    )
