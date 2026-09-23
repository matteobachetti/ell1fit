r"""Summarize the posterior of a *signed* parameter as a value or as a limit.

:mod:`ell1fit.eccentricity` already does this for the eccentricity, but that
quantity is positive by construction (:math:`e = \sqrt{\epsilon_1^2 +
\epsilon_2^2}`), so its 95th percentile *is* an upper limit and its "is it
zero?" test is a question about the origin of a plane. The orbital-period
derivatives ``PBDOT`` and ``PBDDOT`` are neither: each is a single signed
number that can legitimately come out either way, so "upper limit" has to be
defined rather than read off.

The convention used here
------------------------
The headline limit is the ``upper_limit_level`` percentile of
:math:`|x|`. It says exactly what an upper limit should say -- that fraction
of the posterior mass has a magnitude below the quoted number -- and it
collapses to one number, matching the single ``ECC_upper_limit`` that
:mod:`ell1fit.eccentricity` reports. It is deliberately *not* the more extreme
end of the two-sided interval, which would be a different (and, for an
off-centre posterior, larger) statement.

A second limit is always reported at 99.73%, the mass a Gaussian carries
within three standard deviations, since a three-sigma bound is what a
non-detection is often quoted as. Note what it costs: a 99.73% quantile is
estimated from the outermost 0.27% of the chain, so unlike the 95% one it
genuinely depends on how well the tail is sampled. ``{name}_nsamples`` says
how many samples stood behind it -- a few thousand puts only a handful of
samples beyond the limit.

Because a magnitude limit throws the sign away, two-sided credible intervals
are always reported alongside it, at the credible levels a Gaussian one, two
and three sigma carry (68.27%, 95.45% and 99.73%) rather than the rounded
16/84 and 2.5/97.5. A reader who wants to know whether the drift leans
positive or negative reads the interval; a reader who wants one number for a
table reads the limit.

Whether a measurement or a limit is the thing to quote is *not* decided here.
The caller passes ``detected``. In ``ell1decay`` that decision comes from a
nested-sampling Bayes factor against a model that omits the term entirely,
which is a better answer to "does the data need this parameter?" than any
statistic computed from a single model's posterior.

The significance this module *does* report is a diagnostic, quoted so the
numbers are comparable with ``ell1ecc``'s: the distance of zero from the
posterior in units of its own standard deviation, i.e. a Gaussian
approximation, the one-dimensional counterpart of the Mahalanobis distance
:func:`ell1fit.eccentricity.zero_eccentricity_exclusion` uses. A visibly
skewed or multi-modal posterior makes it an approximation rather than a fact;
nothing here decides anything on it.
"""

import numpy as np
from scipy.special import erf


__all__ = [
    "ONE_SIGMA_LEVEL",
    "THREE_SIGMA_LEVEL",
    "TWO_SIGMA_LEVEL",
    "signed_parameter_summary",
]

#: Credible levels of the reported two-sided intervals: the mass a Gaussian
#: carries within one, two and three standard deviations (68.27%, 95.45%,
#: 99.73%).
ONE_SIGMA_LEVEL = float(erf(1.0 / np.sqrt(2.0)))
TWO_SIGMA_LEVEL = float(erf(2.0 / np.sqrt(2.0)))
THREE_SIGMA_LEVEL = float(erf(3.0 / np.sqrt(2.0)))

#: Credible level of the quoted magnitude upper limit, matching
#: :data:`ell1fit.eccentricity.DEFAULT_UPPER_LIMIT_LEVEL`.
DEFAULT_UPPER_LIMIT_LEVEL = 0.95


def _interval(samples, level):
    """Equal-tailed credible interval containing ``level`` of the mass."""
    tail = 50.0 * (1.0 - level)
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def signed_parameter_summary(
    samples,
    name,
    detected,
    upper_limit_level=DEFAULT_UPPER_LIMIT_LEVEL,
    unit=None,
):
    r"""Summarize one signed parameter's posterior samples.

    Parameters
    ----------
    samples : array-like
        Posterior samples of the parameter, in physical units.
    name : str
        Parameter name, used as the prefix of every returned key and in the
        summary line, e.g. ``"PBDOT"``.
    detected : bool
        Whether to quote a measurement (``True``) or an upper limit
        (``False``). The caller owns this decision -- see the module
        docstring.
    upper_limit_level : float
        Credible level of the headline magnitude upper limit. Default 0.95.
        The additional three-sigma limit is always at
        :data:`THREE_SIGMA_LEVEL` and is not affected by this.
    unit : str, optional
        Appended to the numbers in the summary line. ``None`` (the default)
        means dimensionless, as ``PBDOT`` is.

    Returns
    -------
    dict
        ``{name}_50``, the median; ``{name}_1sigma_lo``/``_hi``,
        ``{name}_2sigma_lo``/``_hi`` and ``{name}_3sigma_lo``/``_hi``, the
        two-sided intervals at :data:`ONE_SIGMA_LEVEL`,
        :data:`TWO_SIGMA_LEVEL` and :data:`THREE_SIGMA_LEVEL`;
        ``{name}_upper_limit`` and ``{name}_upper_limit_3sigma`` (both ``nan``
        when ``detected``, since a limit is then not the thing to quote), with
        ``{name}_upper_limit_level`` and
        ``{name}_upper_limit_3sigma_level``;
        ``{name}_zero_credibility`` and ``{name}_significance_sigma``, the
        Gaussian-approximation exclusion of zero; ``{name}_detected``;
        ``{name}_nsamples``; and ``{name}_summary``, the one-line form to
        paste into a paper draft.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> summary = signed_parameter_summary(rng.normal(0, 1, 100000), "PBDOT", detected=False)
    >>> bool(np.isclose(summary["PBDOT_upper_limit"], 1.96, rtol=0.02))
    True
    """
    samples = np.asarray(samples, dtype=float).ravel()
    if samples.size < 3:
        raise ValueError(f"{name}: at least three samples are needed to summarize a posterior.")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name}: every sample must be finite.")

    median = float(np.median(samples))
    one_lo, one_hi = _interval(samples, ONE_SIGMA_LEVEL)
    two_lo, two_hi = _interval(samples, TWO_SIGMA_LEVEL)
    three_lo, three_hi = _interval(samples, THREE_SIGMA_LEVEL)
    magnitude = np.abs(samples)
    limit = float(np.percentile(magnitude, 100.0 * upper_limit_level))
    limit_3sigma = float(np.percentile(magnitude, 100.0 * THREE_SIGMA_LEVEL))

    # One-dimensional counterpart of eccentricity.zero_eccentricity_exclusion:
    # there the origin's Mahalanobis distance is turned into a credibility
    # through the two-degree-of-freedom chi-square tail, here through the
    # one-degree-of-freedom one, which makes the equivalent Gaussian
    # significance simply the distance itself.
    spread = float(np.std(samples))
    sigma = abs(float(np.mean(samples))) / spread if spread > 0 else 0.0
    credibility = float(erf(sigma / np.sqrt(2.0)))

    detected = bool(detected)
    unit_suffix = f" {unit}" if unit else ""

    if detected:
        line = (
            f"{name} = {median:.4g}{unit_suffix} "
            f"(+{one_hi - median:.3g} -{median - one_lo:.3g}, 1 sigma); "
            f"zero excluded at {sigma:.1f} sigma"
        )
    else:
        line = (
            f"|{name}| < {limit:.3g}{unit_suffix} "
            f"({100 * upper_limit_level:g}% upper limit), "
            f"< {limit_3sigma:.3g}{unit_suffix} (99.73%, 3 sigma); "
            f"2 sigma interval {two_lo:.3g} to {two_hi:.3g}{unit_suffix}; "
            f"zero excluded only at {sigma:.2g} sigma, "
            "so this is a limit and not a measurement"
        )

    return {
        f"{name}_50": median,
        f"{name}_1sigma_lo": one_lo,
        f"{name}_1sigma_hi": one_hi,
        f"{name}_2sigma_lo": two_lo,
        f"{name}_2sigma_hi": two_hi,
        f"{name}_3sigma_lo": three_lo,
        f"{name}_3sigma_hi": three_hi,
        f"{name}_upper_limit": float("nan") if detected else limit,
        f"{name}_upper_limit_level": float(upper_limit_level),
        f"{name}_upper_limit_3sigma": float("nan") if detected else limit_3sigma,
        f"{name}_upper_limit_3sigma_level": THREE_SIGMA_LEVEL,
        f"{name}_zero_credibility": credibility,
        f"{name}_significance_sigma": sigma,
        f"{name}_detected": detected,
        f"{name}_nsamples": int(samples.size),
        f"{name}_summary": line,
    }
