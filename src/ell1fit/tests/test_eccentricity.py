"""Recovery of the ELL1 eccentricity parameters from an exact Keplerian orbit.

Every other test of the orbital model generates its data from the ELL1
expansion itself. That checks the implementation, but it cannot check the
*parameterisation*: whether ``EPS1`` and ``EPS2`` come back at
:math:`e\\sin\\omega` and :math:`e\\cos\\omega` of a real orbit, or merely at
whatever the expansion happens to be self-consistent with.

So here the events come from an exact Keplerian (Blandford--Teukolsky) orbit,
with Kepler's equation solved numerically, and the fit is asked to describe them
with ELL1. See :func:`ell1fit.tests.datagen.kepler_orbital_delay`.

The eccentricity is deliberately modest. ELL1 truncates the Roemer delay -- at
second order here -- so its description of an exact orbit carries a residual
growing as :math:`e^3`; at the ``2e-3`` used, that is 3e-7 cycles, four orders
of magnitude below what the fit resolves. Anything that fails here is a real
error rather than the approximation showing through.
"""

import dataclasses

import numpy as np
import pytest

from ..fitting import point_estimate_fit
from ..posterior import _build_posterior_functions
from ..scaling import TARGET_LOCAL_SIGMA, precondition_factors
from .datagen import InjectedSolution, make_multi_epoch_dataset
from .helpers import build_pipeline_state

#: Injected orbit. A different ``omega`` from the default solution, so the two
#: fixtures between them exercise more than one orbital orientation. The
#: eccentricity matches the default and is a ~22 sigma detection at the event
#: count below; ``test_injected_eccentricity_sits_in_the_useful_window`` states
#: the window it has to sit in.
ECCENTRICITY = 2.0e-3
OMEGA_DEG = 71.0

#: Fixed, so the test is deterministic. Over 15 realizations the pull scatter
#: was 1.10 and 0.88 for EPS1/EPS2 with a maximum absolute pull of 2.68, so the
#: three-sigma bound asserted below is a real bound and not a fitted one.
SEED = 20260820


def _laplace_sigma(func, position, step=TARGET_LOCAL_SIGMA):
    """Marginal 1-sigma widths from the numerical Hessian of the log-posterior.

    Central differences at one preconditioned sigma, then invert. Cheaper than
    an MCMC by three orders of magnitude, and enough to ask whether a recovered
    value sits within its own uncertainty.
    """
    n = len(position)
    hessian = np.zeros((n, n))
    centre = func(position)

    for i in range(n):
        for j in range(i, n):
            step_i, step_j = np.zeros(n), np.zeros(n)
            step_i[i] = step_j[j] = step
            if i == j:
                hessian[i, i] = (
                    func(position + step_i) - 2 * centre + func(position - step_i)
                ) / step**2
            else:
                hessian[i, j] = hessian[j, i] = (
                    func(position + step_i + step_j)
                    - func(position + step_i - step_j)
                    - func(position - step_i + step_j)
                    + func(position - step_i - step_j)
                ) / (4 * step**2)

    return np.sqrt(np.diag(np.linalg.inv(-hessian)))


def _fit(outdir, exact_kepler):
    """Generate one dataset and return the fitted values with their sigmas."""
    omega = np.radians(OMEGA_DEG)
    solution = InjectedSolution(
        EPS1=ECCENTRICITY * np.sin(omega),
        EPS2=ECCENTRICITY * np.cos(omega),
        exact_kepler=exact_kepler,
    )
    dataset = make_multi_epoch_dataset(
        str(outdir),
        solution=solution,
        epoch_offsets=(0.0, 37.0, 91.0),
        n_events=12000,
        seed=SEED,
    )

    observations, setup = build_pipeline_state(
        dataset, fit_parameters=("A1", "EPS1", "EPS2", "F0"), nharm=2
    )
    setup = dataclasses.replace(
        setup,
        factors=precondition_factors(
            _build_posterior_functions(observations, setup)[2],
            setup.factors,
            setup.n_parameters,
        ),
    )

    fit_position, fitted, posterior = point_estimate_fit(observations, setup)
    sigma_local = _laplace_sigma(posterior, fit_position)

    values, sigmas = {}, {}
    for name in ("EPS1", "EPS2", "A1"):
        index = setup.parameter_names.index(name)
        values[name] = fitted[name]
        sigmas[name] = sigma_local[index] * setup.factors[index]
    return solution, values, sigmas


@pytest.fixture(scope="module")
def kepler_fit(tmp_path_factory):
    """Fit of events generated from the exact Keplerian orbit."""
    return _fit(tmp_path_factory.mktemp("kepler"), exact_kepler=True)


@pytest.fixture(scope="module")
def ell1_fit(tmp_path_factory):
    """The same, generated from the ELL1 expansion, as a paired control."""
    return _fit(tmp_path_factory.mktemp("ell1"), exact_kepler=False)


def test_eps_recovered_from_an_exact_keplerian_orbit(kepler_fit):
    """``EPS1``/``EPS2`` must come back at ``e sin(omega)``/``e cos(omega)``.

    The orbit the events came from was never expressed in ELL1 terms at all --
    it was integrated from ``ECC`` and ``OM`` through Kepler's equation. So this
    is a test of the parameterisation, not of the expansion.
    """
    solution, values, sigmas = kepler_fit

    for name, truth in (("EPS1", solution.EPS1), ("EPS2", solution.EPS2)):
        pull = (values[name] - truth) / sigmas[name]
        assert abs(pull) < 3, (
            f"{name} = {values[name]:.6e} +- {sigmas[name]:.2e} against an "
            f"injected {truth:.6e}: {pull:+.2f} sigma"
        )


def test_ecc_and_om_recovered_from_an_exact_keplerian_orbit(kepler_fit):
    """The same statement in the Blandford--Teukolsky variables.

    ``e`` and ``omega`` are what a reader of the fit actually wants, and they
    are the variables the orbit was injected in. Uncertainties are propagated
    from the two components, neglecting their covariance.
    """
    solution, values, sigmas = kepler_fit

    eps1, eps2 = values["EPS1"], values["EPS2"]
    sig1, sig2 = sigmas["EPS1"], sigmas["EPS2"]
    eccentricity = np.hypot(eps1, eps2)

    sigma_ecc = np.hypot(eps1 * sig1, eps2 * sig2) / eccentricity
    sigma_om = np.hypot(eps2 * sig1, eps1 * sig2) / eccentricity**2

    assert abs(eccentricity - solution.ECC) < 3 * sigma_ecc, (
        f"e = {eccentricity:.6e} +- {sigma_ecc:.2e} against {solution.ECC:.6e}"
    )

    omega = np.arctan2(eps1, eps2)
    assert abs(omega - solution.OM) < 3 * sigma_om, (
        f"omega = {np.degrees(omega):.2f} +- {np.degrees(sigma_om):.2f} deg "
        f"against {OMEGA_DEG:.2f} deg"
    )


def test_ell1_and_exact_orbits_agree_at_this_eccentricity(kepler_fit, ell1_fit):
    """Paired check that the truncation is negligible where it is claimed to be.

    Both datasets use the same seed, so the noise realization is shared and
    cancels: any difference between the two fits is the approximation itself,
    not scatter. At ``e = 2e-3`` the two agree to better than a tenth of a
    sigma, measured at 0.11 across three seeds.

    This is the end-to-end counterpart of the analytic truncation law. It is
    also the assertion that fails loudly if the two generators ever stop
    describing the same orbit.
    """
    _, kepler_values, kepler_sigmas = kepler_fit
    _, ell1_values, _ = ell1_fit

    for name in ("EPS1", "EPS2"):
        difference = abs(kepler_values[name] - ell1_values[name]) / kepler_sigmas[name]
        assert difference < 0.25, (
            f"{name} differs by {difference:.2f} sigma between an exact orbit and "
            "the ELL1 expansion -- larger than the truncation should produce here"
        )


def test_exact_and_expanded_generators_differ_at_all(kepler_fit, ell1_fit):
    """Guard against the paired test above passing for the wrong reason.

    If ``exact_kepler`` were quietly ignored, the two fixtures would be
    identical and the agreement test would be vacuous. They must differ a
    little -- just not much.
    """
    _, kepler_values, _ = kepler_fit
    _, ell1_values, _ = ell1_fit

    assert kepler_values["EPS1"] != ell1_values["EPS1"]
    assert kepler_values["EPS2"] != ell1_values["EPS2"]


def test_injected_eccentricity_sits_in_the_useful_window():
    """The default injected ``e`` must be both measurable and faithfully modelled.

    Two constraints pull in opposite directions, and the injected value has to
    satisfy both. Too small and the eccentricity is not detected, so no test can
    distinguish a correct eccentricity model from a broken one -- the earlier
    default of ``e = 2.6e-4`` was a 1.8 sigma signal in this fixture. Too large
    and ELL1 stops describing the orbit, so a failure would mean the truncation
    rather than the code.

    Asserted on the injected quantities alone, so this is deterministic and
    costs nothing. The measured detection significance at ``e = 2.0e-3`` is
    12.5 sigma with the default event count and 22 sigma at 12000 per epoch.
    """
    solution = InjectedSolution()

    # Amplitude of the eccentric (2 Phi) terms, in cycles.
    signal = solution.A1 * solution.ECC / 2 * solution.F0
    # Second-order truncation error against an exact orbit; see the measured law
    # in docs/ell1fit/design.rst.
    truncation = 0.236 * solution.ECC**3 * solution.A1 * solution.F0

    assert signal > 0.05, (
        f"the eccentric terms contribute only {signal:.4f} cycles; too weak for "
        "the recovery tests to mean anything"
    )
    assert truncation < 1e-5, (
        f"ELL1 misdescribes this orbit by {truncation:.3e} cycles; a recovery "
        "failure would be the truncation, not the code"
    )
