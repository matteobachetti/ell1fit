"""Prior helper utilities for ell1fit parameter inference."""

import logging

import numpy as np
from scipy.stats import norm


def _flat_logprior(bound0, bound1):
    """Create a uniform log-prior function between two bounds.

    The returned function carries its ``(bound0, bound1)`` as a
    ``phys_bounds`` attribute, so callers that need a hard search-space
    bound (e.g. a bounded local optimizer) can find it without having to
    re-derive or duplicate the bound rules used to build each prior.

    Returns
    -------
    callable
        Function returning ``0`` inside bounds and ``-inf`` outside.
    """
    def func(x):
        if x < bound0 or x > bound1:
            return -np.inf
        return 0

    func.phys_bounds = (bound0, bound1)
    return func


def _periodic_uniform_logprior(center, period, half_width=None):
    """Create a periodic uniform log-prior around a center value.

    Parameters
    ----------
    center : float
        Reference value at the center of the periodic interval.
    period : float
        Period of the wrapped parameter.
    half_width : float or None, optional
        Half-width of accepted interval around ``center`` after wrapping.
        Defaults to ``period / 2``.

    Returns
    -------
    callable
        Function returning ``0`` inside interval and ``-inf`` outside.
    """
    if half_width is None:
        half_width = period / 2

    def func(x):
        dx = ((x - center + 0.5 * period) % period) - 0.5 * period
        if np.abs(dx) > half_width:
            return -np.inf
        return 0

    return func


def _periodic_normal_logprior(center, sigma, period):
    """Create a periodic Gaussian log-prior around a center value.

    Parameters
    ----------
    center : float
        Reference value.
    sigma : float
        Gaussian standard deviation in the same units as ``center``.
    period : float
        Period of the wrapped parameter.

    Returns
    -------
    callable
        Function returning the wrapped Gaussian log-pdf value.
    """
    sigma = np.abs(sigma)
    if sigma == 0 or not np.isfinite(sigma):
        return _periodic_uniform_logprior(center, period)

    norm_const = -0.5 * np.log(2 * np.pi) - np.log(sigma)

    def func(x):
        dx = ((x - center + 0.5 * period) % period) - 0.5 * period
        return norm_const - 0.5 * (dx / sigma) ** 2

    return func


def assign_logpriors(fit_parameter_names, parameters_with_unc, obs_length=1):
    """Assign per-parameter log-prior functions from values and uncertainties.

    Priors are rule-based: bounded uniforms for orbital-shape/phase parameters,
    broad uniforms when uncertainties are unavailable, and Gaussian priors when
    uncertainties are provided.

    Parameters
    ----------
    fit_parameter_names : list of str
        Free parameters needing a prior, in fit order.
    parameters_with_unc : dict
        ``{name: [value, uncertainty]}``. A NaN uncertainty means the parfile
        did not provide one, which selects the broad-uniform branch below.
    obs_length : array-like, optional
        Per-file observation durations in seconds.

    Returns
    -------
    list of callable
        One log-prior per entry of ``fit_parameter_names``, evaluated in
        physical units. Those with hard support also carry a ``phys_bounds``
        attribute; see :func:`_flat_logprior`.
    """
    logps = []
    logging.info("Setting up priors")

    for par in fit_parameter_names:
        log_line = f"{par}: "
        if par == "TASC":
            period = parameters_with_unc["PB"][0] / 86400.0
            tasc_center = parameters_with_unc["TASC"][0]
            tasc_unc = parameters_with_unc["TASC"][1]

            if np.isnan(tasc_unc):
                log_line += (
                    "periodic uniform prior over one orbital cycle " f"(period={period:.6g} d)"
                )
                logps.append(_periodic_uniform_logprior(tasc_center, period, half_width=period / 2))
            else:
                log_line += (
                    "periodic normal prior with "
                    f"mean {tasc_center} d, std {abs(tasc_unc):.2e} d, period {period:.6g} d"
                )
                logps.append(_periodic_normal_logprior(tasc_center, tasc_unc, period))
            logging.info(log_line)
            continue

        if par.startswith("EPS"):
            log_line += "uniform between -1 and 1"
            logps.append(_flat_logprior(-1, 1))
        elif par.startswith("Phase"):
            # parameters_with_unc[par][0] is the template-derived phase-zero offset
            # (ell1fit._prepare_templates_and_phase_priors runs, and writes it
            # here, before priors are assigned). One cycle wide, centered on
            # that offset, so the raw local coordinate stays on a single
            # branch instead of drifting across repeated cycles.
            center = parameters_with_unc[par][0]
            log_line += f"uniform within one cycle of {center:.4f}"
            logps.append(_flat_logprior(center - 0.5, center + 0.5))
        elif (
            np.isnan(parameters_with_unc[par][1]) and par == "PBDOT"
        ):  # For now the uniform distribution is from/to +-np.inf.
            log_line += "uniform between -1 and 1"
            logps.append(_flat_logprior(-1, 1))
        elif np.isnan(parameters_with_unc[par][1]) and par[:2] in ["F0", "PB"]:
            log_line += "uniform between 1/2 and 2 times the mean value"
            value = parameters_with_unc[par][0]
            logps.append(_flat_logprior(value / 2, value * 2))
        elif np.isnan(parameters_with_unc[par][1]) and par == "A1":
            log_line += "uniform between 0 and 2 times the mean value"
            logps.append(_flat_logprior(0, parameters_with_unc[par][0] * 2))
        elif np.isnan(parameters_with_unc[par][1]):
            log_line += "uniform between -inf and inf"
            logps.append(_flat_logprior(-np.inf, np.inf))
        else:
            value, uncertainty = parameters_with_unc[par][0], abs(parameters_with_unc[par][1])
            log_line += f"normal with mean {value} and std {uncertainty:.2e}"
            logps.append(norm(loc=value, scale=uncertainty).logpdf)
        logging.info(log_line)

    return logps
