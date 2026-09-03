"""The tau-parametrized delta_tasc(t) model, and its physical-unit conversions.

Physics
-------
If the true orbital period drifts as ``P(t) = PB0 + PBDOT*t + (1/2)*PBDDOT*t**2
+ ...`` (t measured from the reference epoch), the number of orbital cycles
elapsed by time t is ``N(t) = integral(dt'/P(t'))``. Inverting for the true
time of the Nth ascending node against the time a *fixed*-period (PBDOT=0)
ephemeris would predict for the same N gives, order by order,

    delta_tasc(t) = PBDOT*t**2/(2*PB0) + PBDDOT*t**3/(6*PB0) + ...

i.e. the coefficient of the t**n term (n >= 2) is
``D[n-1] / (n! * PB0)``, where ``D[n-1]`` is the (n-1)-th time derivative of
PB (``D[1] = PBDOT``, ``D[2] = PBDDOT``, ...). Verified against an exact
phase-integral simulation (numerically integrating 1/P(t) and inverting):
recovers injected PBDOT/PBDDOT to 1e-5/1e-4 relative precision, the residual
being exactly the expected truncation of this closed-form expansion at cubic
order, not a formula error.

n=0 and n=1 are not derivatives of PB at all: n=0 is a plain TASC offset
(the reference TASC was not exactly the data's own mean), and n=1 is a linear
drift that a plain PB *miscalibration* (not a PB derivative) would produce --
nuisance terms the fit absorbs but that :func:`physical_from_beta` reports
only as a sanity value, not a headline result.

The tau parametrization
------------------------
Fitting ``PBDOT``/``PBDDOT`` (or their t**n coefficients) directly is exactly
the ill-conditioned optimization problem the original script's ``1e-6`` scale
factor patched by hand, order by order. Substituting ``tau = t/T`` (T = the
data's own time baseline in days) turns the model into a plain polynomial in
tau,

    delta_tasc(t)[sec] = sum_n beta[n] * (t/T)**n

Every ``beta[n]`` is then naturally the same order of magnitude as the data's
own residual amplitude (tens to hundreds of seconds), at *every* polynomial
order, because tau ranges over roughly [-1, 1] across the dataset -- no
per-order scale to derive or hand-tune, and extending to a quartic term later
needs nothing beyond adding ``beta[4]``.

T must be the same for M0 and M1 (see orbital_decay.py): beta[0], beta[1],
beta[2] must mean the identical physical quantity in both models for the
M0-vs-M1 posterior overlay to compare like with like, not renormalize the
axes out from under itself between the two fits.
"""

import math

import numpy as np


__all__ = [
    "delta_tasc_model",
    "log_likelihood_asymmetric_errors",
    "physical_from_beta",
    "spurious_tasc_from_pbdot_mismatch",
]


def delta_tasc_model(beta, x, baseline_days):
    """``sum(beta[n] * (x/baseline_days)**n for n in range(len(beta)))``.

    Parameters
    ----------
    beta : array-like
        Polynomial coefficients in tau, seconds. ``len(beta) == 3`` is M0
        (order 2: TASC offset, linear drift, PBDOT); ``len(beta) == 4`` is M1
        (order 3: adds PBDDOT).
    x : array-like
        Days since the reference epoch.
    baseline_days : float
        Shared time baseline T (see module docstring) -- not recomputed from
        ``x`` here, so M0 and M1 are guaranteed to use the same one.
    """
    beta = np.asarray(beta)
    tau = np.asarray(x) / baseline_days
    return sum(beta[n] * tau**n for n in range(len(beta)))


def physical_from_beta(beta, baseline_days, pb0_days):
    """Convert fitted tau-polynomial coefficients to physical quantities.

    Returns
    -------
    dict
        ``tasc_offset_sec`` (beta[0], direct); ``pb_offset_sec`` (linear
        nuisance term, not a PB derivative -- see module docstring);
        ``PBDOT`` (dimensionless, from beta[2] if present); ``PBDDOT``
        (yr**-1, from beta[3] if present); further orders as
        ``D{n-1}_per_yr{n-2}`` if ``beta`` is ever extended past order 3.
        Keys for orders not present in ``beta`` are omitted, not zero-filled.
    """
    beta = np.asarray(beta, dtype=float)
    result = {}

    if len(beta) > 0:
        result["tasc_offset_sec"] = float(beta[0])
    if len(beta) > 1:
        result["pb_offset_sec"] = float(beta[1] * pb0_days / baseline_days)

    derivative_names = {2: "PBDOT", 3: "PBDDOT"}
    for n in range(2, len(beta)):
        derivative = (
            math.factorial(n)
            * beta[n]
            * pb0_days
            * 365.25 ** (n - 2)
            / (86400.0 * baseline_days**n)
        )
        name = derivative_names.get(n, f"D{n - 1}_per_yr{n - 2}")
        result[name] = float(derivative)

    return result


def spurious_tasc_from_pbdot_mismatch(delta_pbdot, dt_days, pb0_days):
    """Seconds of spurious ``delta_tasc`` a PBDOT mismatch of ``delta_pbdot``
    would inject at an epoch ``dt_days`` from the reference epoch.

    Same physics as :func:`physical_from_beta`'s ``n=2`` conversion, used in
    reverse: if two input files were generated assuming ``PBDOT`` values
    differing by ``delta_pbdot`` (e.g. two processing batches using slightly
    different upstream ephemerides), treating them as one shared-PBDOT
    dataset -- as ell1decay's reference model does -- silently injects a
    spurious quadratic term of exactly this size into the epoch that
    disagrees. Used by
    :func:`ell1fit.orbital_decay_data.check_compatibility` to weigh a PBDOT
    disagreement against that epoch's own TASC uncertainty, rather than
    against an arbitrary numeric tolerance on PBDOT itself: a disagreement
    that would bias the fit by much less than its own statistical precision
    is not worth aborting over.
    """
    return abs(delta_pbdot) * dt_days**2 / (2.0 * pb0_days) * 86400.0


def log_likelihood_asymmetric_errors(beta, x, y, yerrn, yerrp, baseline_days):
    """Split-normal log-likelihood: picks ``yerrn``/``yerrp`` per point by
    which side of the model it falls on.

    Moved from the downstream script this replaces, now calling
    :func:`delta_tasc_model`. **Known limitation, deliberately not fixed
    here**: this is missing the proper split-normal joint normalization
    constant (``sqrt(2/pi) / (yerrn + yerrp)`` per point, not the symmetric
    Gaussian's ``1/sqrt(2*pi*sigma**2)``), so it is not a properly normalized
    likelihood -- every ``log Z`` (and hence every Bayes factor) this module
    produces inherits that. Left as-is because fixing it changes every
    evidence value this command has ever produced, and should be one
    deliberate, separately-reviewed change, not bundled into this feature.
    """
    model = delta_tasc_model(beta, x, baseline_days)
    resid = y - model
    sigma = np.where(resid > 0, yerrp, yerrn)
    return float(np.sum(-0.5 * (resid / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))))
