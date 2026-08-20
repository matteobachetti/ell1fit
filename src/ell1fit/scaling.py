"""Parameter scaling and uncertainty heuristics for ell1fit."""

import logging

import numpy as np

from .phase_utils import simple_freq_re


def order_of_magnitude(value):
    """Return a scale factor one decade below ``abs(value)``.

    Returns
    -------
    float
        Approximate order-of-magnitude scale used for parameter normalization.
    """
    return 10 ** int(np.log10(np.abs(value)) - 1)


def estimate_uncertainties_from_model(model, parameter_names, observation_length, optimistic=False):
    r"""Estimate heuristic 1-sigma scales for selected fit parameters.

    This helper derives approximate uncertainty magnitudes from the binary and
    spin scales of the input model(s), not from covariance propagation or a
    timing fit. It is intended to provide coarse parameter scales for
    initialization/tuning.

    Parameters
    ----------
    model : list
        List of PINT timing models (one per file). The first model provides
        orbital parameters ``PB``, ``PBDOT``, ``A1`` and the maximum ``F0``
        across models is used in the estimates.
    parameter_names : list of str
        Parameter names to estimate. Supported entries are ``PB``, ``A1``,
        ``TASC``, and frequency-derivative names matching ``F<n>_<i>``.
    observation_length : array-like
        Per-file observation durations in seconds. The maximum value is used.

    Returns
    -------
    dict
        Mapping ``{parameter_name: estimated_uncertainty}``.
        Returned units follow parameter units used by this module:
        ``PB`` (s), ``A1`` (light-seconds), ``TASC`` (s, relative to epoch),
        and ``F<n>_<i>`` in the native derivative units.

    Notes
    -----
    The implemented heuristics are:

        - ``PB``: :math:`\sigma_{PB} \approx \frac{\sqrt{3}}{\pi}`
            :math:`\frac{1}{2\pi F0}\frac{PB^2}{A1\,T_{obs}}`
    - ``A1``: :math:`\sigma_{A1} \approx \frac{1}{2\pi F0}`
    - ``TASC``: :math:`\sigma_{TASC} \approx \frac{1}{2\pi F0}\frac{PB}{2\pi A1}`
    - ``F_k``: :math:`\sigma_{F_k} \approx \max(A1\,\Omega^{k+1}F0,\;10/T_{obs}^{k+1})`,
      with :math:`\Omega=2\pi/PB`.
    """
    n_files = len(observation_length)

    P = model[0].PB.value * 86400
    X = model[0].A1.value
    F = np.max([model[i].F0.value for i in range(n_files)])
    twopi = 2 * np.pi
    omega = twopi / P

    common_factor = 1 / twopi / F

    obs_length = np.max(observation_length)

    parameter_uncertainties = {}
    for name in parameter_names:
        if name == "PB":
            parameter_uncertainties["PB"] = (
                np.sqrt(3) / np.pi * common_factor * P**2 / X / obs_length
            )
        elif name == "A1":
            parameter_uncertainties["A1"] = common_factor
        elif name == "TASC":
            parameter_uncertainties["TASC"] = common_factor * P / 86400 / twopi / X
        elif simple_freq_re.match(name):
            order = int(simple_freq_re.match(name).group(1))
            if optimistic:
                parameter_uncertainties[name] = 1 / obs_length ** (order + 1)
            else:
                parameter_uncertainties[name] = max(
                    X * omega ** (order + 1) * F, 10 / obs_length ** (order + 1)
                )

    return parameter_uncertainties


def get_factors(fit_parameter_names, model, observation_length, parvalunc=None):
    """Compute parameter scaling factors for numerically stable local fitting.

    The factors set the size of local parameter variations sampled by the
    optimizer/MCMC, based on spin/orbital sensitivity heuristics.
    """
    zoom = []
    Pd = model[0].PBDOT.value

    # Fixed local walker jitter in safe_run_sampler is 1e-6. Multiplying
    # uncertainties by this value should give physically meaningful initial
    # perturbations while remaining conservative.
    unc_to_factor_scale = 1e6

    approximate_uncertainties = estimate_uncertainties_from_model(
        model, fit_parameter_names, observation_length, optimistic=True
    )

    def _scaled_zoom_from_uncertainty(uncertainty):
        """Convert an uncertainty estimate into a positive local scale."""
        if not np.isfinite(uncertainty) or uncertainty <= 0:
            return None
        zoom_from_unc = order_of_magnitude(uncertainty * unc_to_factor_scale)
        return max(zoom_from_unc, 1e-12)

    for par in fit_parameter_names:
        zoom_factor = None
        source = None

        possible_uncertainties = []
        sources = []

        # For zoom purposes, we prefer the most optimistic uncertainty (to avoid high
        # rejection ratios). For prior purposes, we prefer the most conservative
        # uncertainty (to avoid overconfidence).
        if parvalunc is not None and par in parvalunc:
            possible_uncertainties.append(parvalunc[par][1])
            sources.append("uncertainty")
        if par in approximate_uncertainties:
            possible_uncertainties.append(approximate_uncertainties[par])
            sources.append("model")

        unc_idx = np.argmin(possible_uncertainties) if possible_uncertainties else None
        if unc_idx is not None:
            unc = possible_uncertainties[unc_idx]
            zoom_factor = _scaled_zoom_from_uncertainty(unc)
            source = sources[unc_idx]

        if zoom_factor is None:
            logging.debug("Using default zoom factors")
            if par.startswith("EPS"):
                zoom_factor = 0.001
            elif par == "PBDOT" and np.isfinite(Pd) and Pd != 0:
                zoom_factor = order_of_magnitude(Pd)
            else:
                zoom_factor = 1.0
            source = "default"

        zoom.append(zoom_factor)

        if source.startswith("uncertainty"):
            logging.info(
                f"Zoom factor for {par} from uncertainty: {zoom_factor} "
                f"(unc={unc}, local_jitter=1e-6)"
            )
        elif source.startswith("model"):
            logging.info(f"Zoom factor for {par} from model: {zoom_factor} " f"(approx_unc={unc})")
        else:
            logging.info(f"Zoom factor for {par}: {zoom_factor} (default)")

    return zoom
