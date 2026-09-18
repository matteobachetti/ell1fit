"""Prior helper utilities for ell1fit parameter inference.

Priors are rule-based by default: :func:`assign_logpriors` picks a shape per
parameter from its name and from whether the parfile quoted an uncertainty. A
caller can override any of those rules with a :class:`PriorSpec`, which is what
``ell1fit --prior`` builds; see :func:`parse_prior_spec` for the syntax.
"""

import logging

import numpy as np
from scipy.stats import norm


__all__ = [
    "assign_logpriors",
    "parse_prior_spec",
    "parse_prior_specs",
    "PriorSpec",
]


class _FlatLogPrior:
    """Uniform log-prior between two bounds.

    A class rather than the closure this used to be: ``ell1fit.nuts_sampling``
    and ``ell1fit.prior_transform`` both need to read ``bound0``/``bound1``
    back out to rebuild this prior as a JAX expression or a unit-cube
    transform, and a production nested-sampling pool needs the whole prior
    list -- and hence ``FitSetup`` -- to survive a pickle to a worker
    process. Both are attributes on an ordinary object; neither is available
    on a closure without reading its cell contents by free-variable name,
    and no closure pickles at all.

    Carries its ``(bound0, bound1)`` as a ``phys_bounds`` attribute too, so
    callers that need a hard search-space bound (e.g. a bounded local
    optimizer) can find it without re-deriving or duplicating the bound rules
    used to build each prior.
    """

    def __init__(self, bound0, bound1):
        self.bound0 = bound0
        self.bound1 = bound1
        self.phys_bounds = (bound0, bound1)

    def __call__(self, x):
        if x < self.bound0 or x > self.bound1:
            return -np.inf
        return 0


def _flat_logprior(bound0, bound1):
    """Create a uniform log-prior between two bounds. See :class:`_FlatLogPrior`."""
    return _FlatLogPrior(bound0, bound1)


class _PeriodicUniformLogPrior:
    """Periodic uniform log-prior around a center value.

    See :class:`_FlatLogPrior` for why this is a class rather than a closure.
    """

    def __init__(self, center, period, half_width):
        self.center = center
        self.period = period
        self.half_width = half_width
        self.phys_bounds = (center - 0.5 * period, center + 0.5 * period)

    def __call__(self, x):
        dx = ((x - self.center + 0.5 * self.period) % self.period) - 0.5 * self.period
        if np.abs(dx) > self.half_width:
            return -np.inf
        return 0


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
    return _PeriodicUniformLogPrior(center, period, half_width)


class _PeriodicNormalLogPrior:
    """Periodic Gaussian log-prior around a center value.

    See :class:`_FlatLogPrior` for why this is a class rather than a closure.
    """

    def __init__(self, center, sigma, period):
        self.center = center
        self.sigma = sigma
        self.period = period
        self.norm_const = -0.5 * np.log(2 * np.pi) - np.log(sigma)
        # Periodic, so one period around the centre covers every distinct value.
        self.phys_bounds = (center - 0.5 * period, center + 0.5 * period)

    def __call__(self, x):
        dx = ((x - self.center + 0.5 * self.period) % self.period) - 0.5 * self.period
        return self.norm_const - 0.5 * (dx / self.sigma) ** 2


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
    return _PeriodicNormalLogPrior(center, sigma, period)


#: Prior shapes an override may ask for, and the aliases accepted for each.
_SHAPE_ALIASES = {
    "uniform": "uniform",
    "flat": "uniform",
    "normal": "normal",
    "gaussian": "normal",
}

#: What a relative specification (``+-w``) is introduced by.
_RELATIVE_PREFIXES = ("+-", "\u00b1")

_SPEC_SYNTAX = (
    "Expected NAME:SHAPE:ARGS, with SHAPE one of "
    f"{', '.join(sorted(set(_SHAPE_ALIASES)))} and ARGS either two "
    "comma-separated numbers (bounds for uniform, mean and sigma for normal) "
    "or +-WIDTH to centre the prior on the parameter's input value."
)


class PriorSpec:
    """One command-line prior override, parsed but not yet built.

    Kept separate from the log-prior objects themselves because a spec is
    resolved against the fit *before* it becomes a prior: ``F1`` has to be
    matched to the per-file ``F1_0``, ``F1_1``, ... that are actually fitted,
    and a relative width needs the parameter's input value, which this object
    does not carry.

    Attributes
    ----------
    name : str
        Parameter the override applies to. A bare per-file name (``F1``)
        applies to every expansion of it (``F1_0``, ``F1_1``); a full name
        (``F1_0``) applies to that one only and wins over the bare form.
    shape : {'uniform', 'normal'}
        Normalised shape name.
    relative : bool
        True for the ``+-WIDTH`` form, which is centred on the parameter's
        input value rather than stating absolute numbers.
    args : tuple of float
        ``(low, high)`` or ``(mean, sigma)`` when absolute, ``(width,)`` when
        relative.
    text : str
        The original string, quoted back in error messages.
    """

    def __init__(self, name, shape, relative, args, text=None):
        self.name = name
        self.shape = shape
        self.relative = relative
        self.args = tuple(float(a) for a in args)
        self.text = text if text is not None else f"{name}:{shape}"

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"PriorSpec({self.text!r})"

    def matches(self, par):
        """Whether this spec applies to fit parameter ``par``.

        Uses the same bare-name convention as
        :func:`ell1fit.pipeline._collect_parameter_names`: ``F1`` covers
        ``F1_0``, ``F1_1``, ... as well as a global ``F1``.
        """
        return par == self.name or par.startswith(f"{self.name}_")

    def is_exact(self, par):
        """Whether this spec names ``par`` outright rather than by its bare name."""
        return par == self.name


def parse_prior_spec(text):
    """Parse one ``NAME:SHAPE:ARGS`` prior override.

    Parameters
    ----------
    text : str
        For example ``F1:uniform:-1e-10,1e-10`` (absolute bounds),
        ``F1:uniform:+-1e-10`` (input value plus or minus a width),
        ``PB:normal:16003.2,0.5`` or ``TASC:normal:+-1e-6``.

    Returns
    -------
    PriorSpec

    Raises
    ------
    ValueError
        For any malformed spec. Nothing is guessed: a prior that silently
        became something other than what was asked for would change the answer
        and leave no trace in the output.
    """
    original = text
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Cannot parse prior {original!r}: {_SPEC_SYNTAX}")

    name, shape_text, args_text = (part.strip() for part in parts)
    if not name:
        raise ValueError(f"Cannot parse prior {original!r}: no parameter name. {_SPEC_SYNTAX}")

    shape = _SHAPE_ALIASES.get(shape_text.lower())
    if shape is None:
        raise ValueError(
            f"Cannot parse prior {original!r}: unknown shape {shape_text!r}. {_SPEC_SYNTAX}"
        )

    relative = args_text.startswith(_RELATIVE_PREFIXES)
    if relative:
        for prefix in _RELATIVE_PREFIXES:
            if args_text.startswith(prefix):
                args_text = args_text[len(prefix) :]
                break

    try:
        values = [float(chunk) for chunk in args_text.split(",")]
    except ValueError:
        raise ValueError(
            f"Cannot parse prior {original!r}: {args_text!r} is not numeric. {_SPEC_SYNTAX}"
        ) from None

    expected = 1 if relative else 2
    if len(values) != expected:
        raise ValueError(
            f"Cannot parse prior {original!r}: expected {expected} number(s) after the "
            f"shape, got {len(values)}. {_SPEC_SYNTAX}"
        )
    if not all(np.isfinite(values)):
        raise ValueError(f"Cannot parse prior {original!r}: bounds must be finite. {_SPEC_SYNTAX}")

    if relative and values[0] <= 0:
        raise ValueError(f"Cannot parse prior {original!r}: the +- width must be positive.")
    if not relative and shape == "uniform" and values[0] >= values[1]:
        raise ValueError(
            f"Cannot parse prior {original!r}: the lower bound must be below the upper one."
        )
    if not relative and shape == "normal" and values[1] <= 0:
        raise ValueError(
            f"Cannot parse prior {original!r}: the standard deviation must be positive."
        )

    return PriorSpec(name, shape, relative, values, text=original)


def parse_prior_specs(texts):
    """Parse a list of prior overrides, rejecting duplicate parameter names.

    A repeated name is an error rather than a last-one-wins: two ``--prior``
    options for the same parameter mean the command line disagrees with itself,
    and picking one silently would make the fit answer a question nobody asked.
    """
    if not texts:
        return []

    specs = [parse_prior_spec(text) for text in texts]
    seen = {}
    for spec in specs:
        if spec.name in seen:
            raise ValueError(
                f"Two priors given for {spec.name}: {seen[spec.name].text!r} and {spec.text!r}."
            )
        seen[spec.name] = spec
    return specs


def _resolve_user_prior(par, specs):
    """Find the override that applies to ``par``, preferring an exact name."""
    matching = [spec for spec in specs if spec.matches(par)]
    if not matching:
        return None
    for spec in matching:
        if spec.is_exact(par):
            return spec
    return matching[0]


def _user_prior_arguments(spec, center):
    """Turn a spec into ``(low, high)`` for a uniform, or ``(mean, sigma)`` for a normal."""
    if not spec.relative:
        return spec.args
    width = spec.args[0]
    if spec.shape == "uniform":
        return (center - width, center + width)
    return (center, width)


def _build_user_logprior(spec, par, parameters_with_unc):
    """Build the log-prior an override asks for, and a line describing it.

    ``TASC`` keeps its periodic wrapping whatever the override says: the
    parameter is an epoch defined modulo one orbit, and a prior that does not
    wrap lets a walker settle a whole orbit away from where it was aimed.
    """
    center = parameters_with_unc[par][0]
    first, second = _user_prior_arguments(spec, center)

    is_tasc = par == "TASC"
    period = parameters_with_unc["PB"][0] / 86400.0 if is_tasc else None

    if spec.shape == "uniform":
        low, high = first, second
        if is_tasc:
            tasc_center = 0.5 * (low + high)
            half_width = 0.5 * (high - low)
            if half_width > 0.5 * period:
                logging.warning(
                    f"Prior {spec.text!r} is wider than one orbital period; "
                    f"clipping it to one full cycle ({period:.6g} d)."
                )
                half_width = 0.5 * period
            return (
                _periodic_uniform_logprior(tasc_center, period, half_width=half_width),
                f"periodic uniform within +-{half_width:.6g} d of {tasc_center}",
            )
        return _flat_logprior(low, high), f"uniform between {low:.6g} and {high:.6g}"

    mean, sigma = first, second
    if is_tasc:
        return (
            _periodic_normal_logprior(mean, sigma, period),
            f"periodic normal with mean {mean} d, std {sigma:.6g} d, period {period:.6g} d",
        )
    return (
        norm(loc=mean, scale=sigma).logpdf,
        f"normal with mean {mean} and std {sigma:.6g}",
    )


def _check_specs_are_used(fit_parameter_names, specs):
    """Reject an override that names a parameter the fit does not contain.

    Silently dropping it is the same failure mode ``-P TSAC`` used to have: the
    fit stays internally consistent, runs to completion, and answers a
    different question than the one asked.
    """
    unused = [
        spec.text for spec in specs if not any(spec.matches(par) for par in fit_parameter_names)
    ]
    if unused:
        raise ValueError(
            f"Prior(s) {', '.join(repr(u) for u in unused)} name parameters that are not "
            f"being fitted. Fitted parameters: {', '.join(fit_parameter_names)}."
        )


def assign_logpriors(fit_parameter_names, parameters_with_unc, obs_length=1, user_priors=None):
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
    user_priors : list of PriorSpec, optional
        Overrides from ``ell1fit --prior``. Each one replaces the rule that
        would otherwise apply to the parameter it names, and a spec naming a
        parameter that is not being fitted raises rather than being ignored.

    Returns
    -------
    list of callable
        One log-prior per entry of ``fit_parameter_names``, evaluated in
        physical units. Those with hard support also carry a ``phys_bounds``
        attribute; see :func:`_flat_logprior`.
    """
    logps = []
    logging.info("Setting up priors")

    user_priors = list(user_priors) if user_priors else []
    _check_specs_are_used(fit_parameter_names, user_priors)

    for par in fit_parameter_names:
        log_line = f"{par}: "

        spec = _resolve_user_prior(par, user_priors)
        if spec is not None:
            logprior, description = _build_user_logprior(spec, par, parameters_with_unc)
            logps.append(logprior)
            logging.info(log_line + description + " (from --prior)")
            continue

        if par == "TASC":
            period = parameters_with_unc["PB"][0] / 86400.0
            tasc_center = parameters_with_unc["TASC"][0]
            tasc_unc = parameters_with_unc["TASC"][1]

            if np.isnan(tasc_unc):
                log_line += f"periodic uniform prior over one orbital cycle (period={period:.6g} d)"
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
        elif np.isnan(parameters_with_unc[par][1]) and par == "A1DOT":
            # Flat across the whole physically allowed range. The orbit cannot
            # change size faster than the projected orbital velocity
            # |A1| * 2pi / PB -- in units of c, the very quantity
            # ``orbit_is_invertible`` screens on -- and nothing in between is
            # preferred. That is ~1e7 times wider than any credible drift,
            # which is deliberate: an upper limit is only worth quoting if the
            # prior bound is not what sets it. Finite, so the prior stays
            # proper and nested sampling can integrate against it -- at the
            # cost of an Occam factor that a Bayes factor on A1DOT would feel.
            center = parameters_with_unc[par][0]
            half_width = (
                np.abs(parameters_with_unc["A1"][0]) * 2 * np.pi / parameters_with_unc["PB"][0]
            )
            log_line += f"uniform within +-{half_width:.3e} lt-s/s of {center:.3e}"
            logps.append(_flat_logprior(center - half_width, center + half_width))
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
