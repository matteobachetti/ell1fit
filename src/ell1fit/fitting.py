"""Point-estimate optimization and MCMC sampling for ell1fit.

Two stages, deliberately separable:

:func:`point_estimate_fit`
    A bounded local optimization that returns a single best-fit position. Cheap,
    deterministic, and repeatable -- which is what lets the pulse template be
    refined iteratively without paying for a full posterior exploration each
    time round.
:func:`optimize_solution`
    The full run: an optional point estimate to warm-start, then MCMC, then
    diagnostics and result summaries.

Both work in the local coordinates described in :mod:`ell1fit.posterior`.
"""

import copy
import logging

import numpy as np
from scipy.optimize import minimize

from .mcmc_utils import safe_run_sampler
from .phase_utils import phases_from_zero_to_one
from .posterior import _build_posterior_functions
from .profile_plotting import _compare_phaseograms


__all__ = [
    "_bounds_in_local_coordinates",
    "optimize_solution",
    "point_estimate_fit",
]


def _plot_phaseogram_set(reference_phases, fitted_phases, times_from_pepoch, outroots, suffix=""):
    """Plot reference vs fitted phaseograms for each input file."""
    for i in range(len(times_from_pepoch)):
        _compare_phaseograms(
            reference_phases[i],
            phases_from_zero_to_one(fitted_phases[i]),
            times_from_pepoch[i],
            fname=outroots[i] + suffix + ".jpg",
        )


def _augment_results_with_fit_metadata(
    results,
    fit_parameter_names,
    fit_pars,
    values,
    factors,
    phase_source,
):
    """Store local-fit summaries and scaling metadata in sampler output."""
    rough_results = {}
    for par, value in zip(fit_parameter_names, fit_pars):
        rough_results["rough_d" + par] = value

    count = 0
    while f"Phase_{count}" in phase_source:
        results[f"additional_phase_{count}"] = phase_source[f"Phase_{count}"]
        count += 1

    results.update(rough_results)

    for par, initial, f in zip(fit_parameter_names, values, factors):
        results["d" + par + "_mean"] = results["d" + par + "_50"]
        results["d" + par + "_initial"] = initial
        results["d" + par + "_factor"] = f

    return results


def _bounds_in_local_coordinates(values, factors, logprior_funcs):
    """Translate each prior's hard support into the optimizer's coordinates.

    Hard-bounded priors (``Phase``, ``EPS``, ...) return ``-inf`` outside their
    window. An unconstrained optimizer cannot see that and will step past it,
    which surfaces as "invalid value encountered in subtract" when scipy's
    finite-difference gradient probes two infinite points. Passing real bounds
    keeps the search inside the support and makes scipy select L-BFGS-B.

    Returns
    -------
    list of tuple
        ``(low, high)`` per fitted parameter, in local coordinates.
    """
    bounds = []
    for initial, factor, logp_func in zip(values, factors, logprior_funcs):
        lo, hi = getattr(logp_func, "phys_bounds", (-np.inf, np.inf))
        bounds.append(((lo - initial) / factor, (hi - initial) / factor))
    return bounds


def point_estimate_fit(observations, setup):
    """Find the best-fit position by bounded local optimization, without MCMC.

    Separated from :func:`optimize_solution` so that callers which only need a
    point estimate -- notably iterative template refinement, which refits once
    per pass -- do not have to pay for a full posterior exploration each time.

    Returns
    -------
    fit_pars : np.ndarray
        Best-fit position in local coordinates.
    fitted_parameters : dict
        Full parameter mapping with the fitted values substituted in.
    func_to_maximize : callable
        The posterior function that was optimized, so callers can evaluate it
        again without rebuilding it.
    """
    _, _, func_to_maximize = _build_posterior_functions(observations, setup)

    def func_to_minimize(pars):
        return -func_to_maximize(pars)

    values = setup.baseline_values
    factors = setup.factors
    bounds = _bounds_in_local_coordinates(values, factors, setup.logprior_funcs)
    result = minimize(func_to_minimize, [0] * len(values), bounds=bounds)
    fit_pars = result.x

    fitted_parameters = copy.deepcopy(setup.parameters)
    for par, initial, value, factor in zip(setup.parameter_names, values, fit_pars, factors):
        fitted_parameters[par] = value * factor + initial

    return fit_pars, fitted_parameters, func_to_maximize


def optimize_solution(
    observations,
    setup,
    nsteps=1000,
    minimize_first=False,
    outroots=("out",),
    reference_phases=None,
):
    """Optimize and sample pulsar timing parameters for multiple event files.

    Workflow:
    1. Build a posterior from priors + profile likelihood.
    2. Optionally run deterministic minimization for a starting point.
    3. Run MCMC with :func:`safe_run_sampler`.
    4. Produce diagnostic phaseogram comparisons and return summary fields.

    Parameters are handled in local coordinates: for each fitted parameter,
    ``physical = local * factor + initial``.

    Parameters
    ----------
    reference_phases : list of np.ndarray or None, optional
        Phases to draw in the left-hand panel of the comparison phaseograms --
        the "before" of a before-and-after. Pass the phases of the solution the
        run *started* from.

        This must be supplied by the caller rather than derived here. The
        obvious derivation, evaluating the posterior at local coordinates zero,
        silently means "whatever ``setup.baseline_values`` currently holds",
        which is only the starting solution as long as nothing has re-centred
        the baseline in between. Iterative template refinement does exactly
        that, and the comparison then comes out as the refined solution against
        itself: two identical panels, and a diagnostic that always looks
        perfect. Defaults to the local-zero solution when omitted, which is
        correct only when no refinement has run.

    Returns
    -------
    dict
        Aggregated result dictionary containing posterior summaries, initial
        values, scaling factors, and copied model metadata.
    """
    times_from_pepoch = observations.times_from_pepoch
    parameters = setup.parameters
    fit_parameter_names = setup.parameter_names
    values = setup.baseline_values
    factors = setup.factors
    logprior_funcs = setup.logprior_funcs

    _, local_phases, func_to_maximize = _build_posterior_functions(
        observations,
        setup,
        debug_local_phases=True,
        debug_func=True,
    )

    def func_to_minimize(pars):
        return -func_to_maximize(pars)

    logging.info("Initial parameters: ")
    for par in fit_parameter_names:
        logging.info(f"  {par}: {parameters[par]}")

    logging.info("Initial likelihood: " + str(func_to_maximize([0] * len(values))))
    all_zeros = [0] * len(values)
    if minimize_first:
        bounds = _bounds_in_local_coordinates(values, factors, logprior_funcs)
        res = minimize(func_to_minimize, all_zeros, bounds=bounds)
        fit_pars = res.x
    else:
        fit_pars = all_zeros

    logging.info("Fitted (rescaled) parameters: " + str(fit_pars))

    fitted_parameters = copy.deepcopy(parameters)

    for par, initial, value, f in zip(fit_parameter_names, values, fit_pars, factors):
        fitted_parameters[par] = value * f + initial

    for key in fit_parameter_names:
        logging.info(
            f"  {key}: {fitted_parameters[key]} "
            f"(difference from initial: {fitted_parameters[key] - parameters[key]})"
        )
    logging.info("Fitted likelihood: " + str(func_to_maximize(fit_pars)))
    phases = local_phases(fit_pars)
    if reference_phases is None:
        reference_phases = local_phases(all_zeros)

    _plot_phaseogram_set(reference_phases, phases, times_from_pepoch, outroots, suffix="")

    corner_labels = [
        "d" + par + f"{np.log10(fac):+g}" for (par, fac) in zip(fit_parameter_names, factors)
    ]
    results = safe_run_sampler(
        func_to_maximize,
        fit_pars,
        max_n=nsteps,
        outroot=outroots[-1],
        labels=["d" + par for par in fit_parameter_names],
        corner_labels=corner_labels,
    )

    results.update(parameters)
    results = _augment_results_with_fit_metadata(
        results,
        fit_parameter_names,
        fit_pars,
        values,
        factors,
        phase_source=fitted_parameters,
    )

    fit_pars = [results["d" + par + "_50"] for par in fit_parameter_names]
    phases = local_phases(fit_pars)

    _plot_phaseogram_set(reference_phases, phases, times_from_pepoch, outroots, suffix="_final")

    return results
