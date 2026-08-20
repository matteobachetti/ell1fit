"""Posterior construction and single-parameter likelihood traces for ell1fit.

Local coordinates
-----------------
Every fitted parameter is handled in a normalized local coordinate rather than
in its physical units::

    physical = local * factor + initial

``initial`` is the starting value from the timing model and ``factor`` is the
per-parameter scale from :mod:`ell1fit.scaling`. The point is conditioning: the
fitted quantities span wildly different magnitudes -- ``F0`` in Hz to fifteen
digits alongside ``A1`` in light-seconds -- and an optimizer or an MCMC walker
stepping the same distance in every direction only behaves if those directions
have comparable scale. In local coordinates the starting point is the origin
and a step of order one is of order one sigma.

Anything that reports or plots a parameter has to undo the transformation; the
mapping is applied in one place, :func:`_build_posterior_functions`, so that
callers do not each re-derive it.
"""

import copy
import dataclasses
import logging

import matplotlib.pyplot as plt
import numpy as np

from .phase_utils import NonInvertibleOrbitError, _calculate_phases
from .plotting import plot_style_context as _plot_style_context


def _trace_phase_0_likelihood(observations, setup, outroot):
    """Trace the likelihood around each file's Phase_i and move it to the peak.

    Mutates ``setup.parameters`` in place, leaving each ``Phase_i`` at its
    best-fit value. Callers must re-centre the local coordinate system
    afterwards -- see :meth:`FitSetup.with_baseline_from`.
    """
    parameters = setup.parameters
    factors = setup.factors
    for parameter in [p for p in setup.parameter_names if p.startswith("Phase_")]:
        idx = setup.parameter_names.index(parameter)
        results_trace = trace_likelihood_over_parameter(
            observations,
            setup,
            parameter_name=parameter,
            parameter_values=np.linspace(
                parameters[parameter] - 6 * factors[idx],
                parameters[parameter] + 6 * factors[idx],
                100,
            ),
        )

        phase_values = list(results_trace.keys())
        ll_values = list(results_trace.values())
        best_phase = phase_values[np.nanargmax(ll_values)]

        with _plot_style_context():
            fig = plt.figure("trace_" + parameter)
            plt.plot(phase_values, ll_values, color="black")
            plt.axvline(parameters[parameter], color="k", alpha=0.5, ls="--")
            plt.axvline(best_phase, color="r", ls="--")
            plt.xlabel(parameter)
            plt.ylabel("log likelihood")
            plt.savefig(outroot + f"_trace_{parameter}.jpg")
            plt.close(fig)

        ll_values_clean = [
            ll for ll in list(results_trace.values()) if not np.isnan(ll) and not np.isinf(ll)
        ]
        min_ll = np.nanmin(ll_values_clean)
        max_ll = np.nanmax(ll_values_clean)
        logging.info(f"Delta log likelihood for {parameter}: {max_ll - min_ll:.2f}")
        parameters[parameter] = phase_values[np.nanargmax(ll_values)]
    return parameters


def trace_likelihood_over_parameter(
    observations,
    setup,
    parameter_name,
    parameter_values,
):
    """Trace the posterior (log-likelihood + log-prior) over one parameter.

    All fitted parameters are represented in a normalized local coordinate
    system used by :func:`optimize_solution`, where the physical value is
    ``local * factor + initial``. This helper fixes every local parameter at
    zero except ``parameter_name``, which is scanned over ``parameter_values``
    -- given here as *physical* (absolute) values, not local ones; each is
    converted to the local coordinate that reproduces it before evaluating the
    posterior, so the scan lands where requested regardless of what baseline
    ``values`` happens to hold for that parameter.

    Returns
    -------
    dict
        Mapping from scanned physical parameter value to posterior value.
    """

    # A trace is a diagnostic of the likelihood surface itself, so it is taken
    # unweighted regardless of how the fit is configured.
    _, _, func_to_maximize = _build_posterior_functions(
        observations,
        dataclasses.replace(setup, weights=None),
    )

    values = setup.baseline_values
    factors = setup.factors
    idx = setup.parameter_names.index(parameter_name)
    results = {}
    for val in parameter_values:
        pars = [0] * len(values)
        pars[idx] = (val - values[idx]) / factors[idx]
        results[val] = func_to_maximize(pars)

    return results


def _build_posterior_functions(
    observations,
    setup,
    debug_local_phases=False,
    debug_func=False,
):
    """Build shared posterior components used by trace and optimization flows.

    Parameters
    ----------
    observations : ObservationSet
        The event data to evaluate against.
    setup : FitSetup
        What is being fitted: free parameters, baseline, scaling, priors,
        templates, likelihood and weights.

    Returns
    -------
    tuple of callable
        ``(logprior, local_phases, func_to_maximize)``, each taking a position
        in local coordinates.
    """
    times_from_pepoch = observations.times_from_pepoch
    parameters = setup.parameters
    fit_parameter_names = setup.parameter_names
    values = setup.baseline_values
    logprior_funcs = setup.logprior_funcs
    factors = setup.factors
    template_func = setup.template_funcs
    likelihood_func = setup.likelihood_func
    tolerance = setup.tolerance
    weights = setup.weights

    def logprior(pars):
        if np.any(np.isnan(pars)):
            return -np.inf
        if np.any(np.isinf(pars)):
            return -np.inf

        logp = 0
        for parname, logp_func, initial, local_value, f in zip(
            fit_parameter_names, logprior_funcs, values, pars, factors
        ):
            value = local_value * f + initial
            logp += logp_func(value)
        return logp

    def local_phases(pars):
        trial_parameters = copy.deepcopy(parameters)

        for par, initial, value, f in zip(fit_parameter_names, values, pars, factors):
            trial_parameters[par] = value * f + initial
        if debug_local_phases:
            logging.debug(f"Local phases for parameters: {trial_parameters}")

        return _calculate_phases(times_from_pepoch, trial_parameters, tolerance=tolerance)

    def func_to_maximize(pars):
        lp = logprior(pars)
        if np.isinf(lp):
            return lp

        try:
            phases = local_phases(pars)
        except NonInvertibleOrbitError:
            # A trial position outside the physically invertible region: the
            # pulsar's projected orbital motion would be superluminal, so pulse
            # phases are undefined there. Reject it as impossible rather than
            # letting the deorbiting iteration grind against a map that has no
            # fixed point. A Gaussian prior alone cannot do this -- its log-pdf
            # is hugely negative but finite, so the check above never fires.
            return -np.inf

        ll = 0
        for i in range(len(phases)):
            ll += likelihood_func(
                phases[i], template_func[i], weights=weights[i] if weights is not None else None
            )

        if debug_func:
            logging.debug(f"pars: {pars}, func: {ll + lp}")

        return ll + lp

    return logprior, local_phases, func_to_maximize
