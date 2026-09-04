"""Parameter scaling and uncertainty heuristics for ell1fit."""

import logging

import numpy as np

from .phase_utils import simple_freq_re


__all__ = [
    "estimate_uncertainties_from_model",
    "get_factors",
    "OPTIMIZER_EPS",
    "order_of_magnitude",
    "precondition_factors",
]


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
        ``A1DOT``, ``TASC``, and frequency-derivative names matching
        ``F<n>_<i>``.
    observation_length : array-like
        Per-file observation durations in seconds. The maximum value is used.

    Returns
    -------
    dict
        Mapping ``{parameter_name: estimated_uncertainty}``.
        Returned units follow parameter units used by this module:
        ``PB`` (s), ``A1`` (light-seconds), ``A1DOT`` (light-seconds per
        second), ``TASC`` (s, relative to epoch), and ``F<n>_<i>`` in the
        native derivative units.

    Notes
    -----
    The implemented heuristics are:

    - ``PB``: :math:`\sigma_{PB} \approx \frac{\sqrt{3}}{\pi}
      \frac{1}{2\pi F0}\frac{PB^2}{A1\,T_{obs}}`
    - ``A1``: :math:`\sigma_{A1} \approx \frac{1}{2\pi F0}`
    - ``A1DOT``: :math:`\sigma_{\dot{A1}} \approx \sigma_{A1} / T_{span}`, with
      :math:`T_{span}` the spread of the files' ``PEPOCH``. Omitted when a
      single epoch leaves no lever arm.
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

    # Epoch spacing, in seconds: the lever arm any orbital derivative is
    # measured over. Zero for a single file.
    epochs = np.array([model[i].PEPOCH.value for i in range(n_files)], dtype=float)
    baseline = float(epochs.max() - epochs.min()) * 86400.0

    parameter_uncertainties = {}
    for name in parameter_names:
        if name == "PB":
            parameter_uncertainties["PB"] = (
                np.sqrt(3) / np.pi * common_factor * P**2 / X / obs_length
            )
        elif name == "A1":
            parameter_uncertainties["A1"] = common_factor
        elif name == "A1DOT":
            # A drift is one epoch's A1 precision divided by the lever arm the
            # epochs give it, so this is the only scale here that depends on
            # the *spacing* of the files rather than on their length. With a
            # single epoch there is no lever arm and no estimate: the entry is
            # omitted, and the pipeline refuses the fit outright rather than
            # let a flat direction return its prior.
            if baseline > 0:
                parameter_uncertainties["A1DOT"] = common_factor / baseline
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


#: The local-coordinate convention: a step of this size should correspond to
#: roughly one standard deviation of the parameter. ``safe_run_sampler`` spreads
#: its initial walkers by exactly this amount, so honouring it makes the MCMC
#: start with a sensibly-sized ball in *every* direction.
TARGET_LOCAL_SIGMA = 1e-6


#: How far the point-estimate optimizer steps when it probes the gradient by
#: finite differences, in local units.
#:
#: ``scipy.optimize.minimize`` defaults to ``eps=1e-8``, an *absolute* number
#: with no idea what a local unit means here -- so the probe size was silently
#: tied to :data:`TARGET_LOCAL_SIGMA`, and changing that constant would have
#: quietly broken the fit. Writing it as a fraction of a sigma removes the
#: coupling. A hundredth of a sigma is exactly what the default was already
#: giving at ``TARGET_LOCAL_SIGMA = 1e-6``, so this is a no-op at that setting.
#:
#: It has to be this coarse. The posterior is numerically far noisier than
#: rounding: measured on bench problem ``P1``, a log-posterior of order 105 with
#: a rounding floor of 1.4e-14 still jitters by **7.5e-9** over a 1e-8
#: displacement, because the phase calculation carries its own
#: ``tolerance=1e-8``. A probe much finer than this would measure that jitter
#: instead of the slope, and the optimizer would stop wherever it happened to
#: stand.
OPTIMIZER_EPS = 1e-2 * TARGET_LOCAL_SIGMA


#: A drop smaller than this is rounding noise rather than curvature.
_MEASURABLE_DROP = 1e-3


def _curvature_drop(posterior_func, base, n_parameters, index, step):
    """Measure how far the log-posterior curves away over ``step``, without its slope.

    The quantity returned is ``0.5 * (step / sigma) ** 2``: what the posterior
    would fall by at ``step`` if the starting point were the peak. Taking the
    plain one-sided fall instead is what the earlier implementation did, and it
    is only the same thing when the starting point *is* the peak. Off the peak
    by ``d``, the one-sided fall is dominated by the linear term and the scale
    it implies collapses roughly as ``sigma / d`` -- measured on a two-epoch
    fixture, ``Phase_i`` sits 0.33 and 0.25 sigma off the peak (it is centred on
    a grid whose cells are a full sigma wide) and its scale came out **14x too
    small**, while ``A1`` and ``F0``, which start at the peak because the
    parfile is exact, came out right. Displacing those two deliberately
    reproduces the same collapse: 0.39x at 0.3 sigma, 0.22x at 1 sigma, 0.13x at
    3 sigma. The symmetric form below returns 1.00x at every one of those
    offsets.

    Returns
    -------
    float or None
        ``None`` when a hard prior bound blocks the measurement at this step,
        which tells the caller to try a smaller one.
    """

    def at(offset):
        probe = np.zeros(n_parameters)
        probe[index] = offset
        return posterior_func(probe)

    up = at(step)
    down = at(-step)
    if np.isfinite(up) and np.isfinite(down):
        # f(0) - (f(s) + f(-s))/2 = -h s^2 / 2 for f = f0 + g x + h x^2 / 2:
        # the linear term cancels identically, whatever g is.
        return base - 0.5 * (up + down)

    # One side is outside a hard-bounded prior (``EPS`` is confined to +-1,
    # ``A1`` to twice its value), so the symmetric stencil is unavailable.
    # Three points on the feasible side remove the linear term just as exactly:
    # f(0) - 2 f(s) + f(2s) = h s^2. Only the reach is worse, which matters
    # solely for a parameter starting near a bound.
    forward, sign = (up, 1.0) if np.isfinite(up) else (down, -1.0)
    if not np.isfinite(forward):
        return None
    far = at(2.0 * sign * step)
    if not np.isfinite(far):
        return None
    return -0.5 * (base - 2.0 * forward + far)


def precondition_factors(posterior_func, factors, n_parameters, target=TARGET_LOCAL_SIGMA):
    """Rescale parameter factors so every direction has a comparable local scale.

    :func:`get_factors` derives each parameter's scale from whatever uncertainty
    information happens to be available, and for some parameters there is none.
    ``Phase_i`` is the clear case: :func:`ell1fit.models._get_par_dict` records
    its uncertainty as ``0``, no model-based estimate covers it, and it falls
    through to the default factor of ``1``. Every other parameter gets a
    data-derived scale, so the directions end up wildly mismatched -- measured at
    a **1000x spread** on a routine two-epoch fit.

    That matters because L-BFGS-B maintains a single Hessian approximation: no
    step size suits directions three orders of magnitude apart, and it stalls on
    the shallow ones. Measured on a two-epoch fit, rescaling took the optimizer
    from reaching the global optimum in 7 of 12 starts to **12 of 12**, and
    collapsed a 3.2-nat spread in the achieved log-posterior to zero. It also
    fixes the MCMC's starting ball, which was 0.0006 sigma wide in ``Phase`` and
    0.58 sigma wide in ``A1``.

    The scale is measured from the posterior itself rather than derived
    per-parameter, so it needs no formula for each new parameter type and adapts
    to the actual data. It is measured as a *curvature*, by
    :func:`_curvature_drop`, and not as the fall in the log-posterior over one
    step -- see that function for why the difference is worth several extra
    evaluations, and for what it cost when it was not.

    Parameters
    ----------
    posterior_func : callable
        Log-posterior in local coordinates.
    factors : sequence of float
        Current per-parameter factors.
    n_parameters : int
        Number of free parameters.
    target : float, optional
        Local step that should correspond to one standard deviation.

    Returns
    -------
    list of float
        Rescaled factors. A parameter whose scale cannot be measured -- a flat
        or non-finite direction -- keeps the factor it came in with.
    """
    origin = np.zeros(n_parameters)
    base = posterior_func(origin)
    if not np.isfinite(base):
        logging.warning(
            "Cannot precondition parameter scales: posterior is not finite at the "
            "starting point. Keeping the original factors."
        )
        return list(factors)

    rescaled = list(factors)
    for i in range(n_parameters):
        step = target

        # Shrink until the geometry can be measured at all: a hard prior bound
        # nearer than the step blocks both stencils.
        for _ in range(60):
            if _curvature_drop(posterior_func, base, n_parameters, i, step) is not None:
                break
            step *= 0.5

        # Then step outward until the posterior curves by enough to be more
        # than rounding noise, stopping if growing runs into a bound.
        drop = None
        for _ in range(60):
            measured = _curvature_drop(posterior_func, base, n_parameters, i, step)
            if measured is None:
                break
            if measured > _MEASURABLE_DROP:
                drop = measured
                break
            step *= 2.0

        if drop is None:
            logging.debug(f"Parameter {i} has no measurable curvature; keeping its factor")
            continue

        # A quadratic peak drops by 0.5 at one sigma: drop = 0.5 (step/sigma)^2.
        sigma_local = step * np.sqrt(0.5 / drop)
        if not np.isfinite(sigma_local) or sigma_local <= 0:
            continue
        rescaled[i] = factors[i] * (sigma_local / target)

    return rescaled


def get_factors(fit_parameter_names, model, observation_length, parameters_with_unc=None):
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
        if parameters_with_unc is not None and par in parameters_with_unc:
            possible_uncertainties.append(parameters_with_unc[par][1])
            sources.append("uncertainty")
        if par in approximate_uncertainties:
            possible_uncertainties.append(approximate_uncertainties[par])
            sources.append("model")

        # A parfile that quotes no uncertainty leaves a NaN here, and NaN wins
        # ``argmin`` -- so a missing value used to beat a perfectly good model
        # estimate and drop the parameter to the default scale. Discard the
        # unusable candidates before choosing rather than after.
        usable = [
            (unc, src)
            for unc, src in zip(possible_uncertainties, sources)
            if np.isfinite(unc) and unc > 0
        ]
        if usable:
            unc, source = min(usable)
            zoom_factor = _scaled_zoom_from_uncertainty(unc)

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
            logging.info(f"Zoom factor for {par} from model: {zoom_factor} (approx_unc={unc})")
        else:
            logging.info(f"Zoom factor for {par}: {zoom_factor} (default)")

    return zoom
