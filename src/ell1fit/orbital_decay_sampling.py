"""Self-contained nested-sampling wrapper for the M0/M1 delta_tasc fits.

Deliberately does not import :mod:`ell1fit.nested_sampling` or
:mod:`ell1fit.prior_transform`: both are built around the main pipeline's
``observations``/``setup`` (``FitSetup``) objects and an unnormalized mixture
prior that needs a logprior/logpost split to hand dynesty a bare likelihood.
Here the model has 3-4 parameters, a plain bounded-uniform prior (trivially
self-normalizing), and no pipeline objects at all -- forcing it through that
machinery would add coupling without buying anything. What *is* reused is the
pattern of two lessons measured elsewhere in this codebase (see
:mod:`ell1fit.nested_sampling`): a peak-shortfall convergence gate, since
nested sampling can silently miss a mode and still report a confident,
wrong ``log Z``; and seed-scatter over dynesty's own quoted error, since that
error can be far too small on a correlated posterior.
"""

import logging

import numpy as np
from scipy.optimize import minimize


__all__ = [
    "PEAK_SHORTFALL_GATE",
    "bayes_factor",
    "default_bounds",
    "laplace_cross_check",
    "run_one_nested_fit",
    "run_seed_scatter",
]

#: Same value and rationale as ell1fit.nested_sampling.PEAK_SHORTFALL_GATE:
#: nats the sampler's best likelihood may fall short of an independent
#: optimizer's MAP before log Z is flagged unreliable. Duplicated rather than
#: imported -- see the module docstring for why this module avoids that
#: import.
PEAK_SHORTFALL_GATE = 1.0


def default_bounds(y, order):
    """Bounded-uniform prior box for the order-``order`` tau-polynomial fit.

    Every ``beta[n]`` is already the same order of magnitude as the data's
    own residual amplitude regardless of ``n`` (that is the point of the tau
    parametrization -- see :mod:`ell1fit.orbital_decay_model`), so one bound
    scheme covers every order. ``beta[0]``/``beta[1]`` get a generous box
    around the data's own spread; ``beta[n >= 2]`` get a *wider* one, since
    those are exactly the terms being tested for existence (PBDOT, PBDDOT,
    ...) -- tightening them from the data being fit would bias the evidence
    towards "detected".

    Parameters
    ----------
    y : array-like
        The delta_tasc data, seconds.
    order : int
        Highest tau power in the model (2 for M0, 3 for M1).

    Returns
    -------
    list of (float, float)
        Length ``order + 1``.
    """
    y = np.asarray(y, dtype=float)
    spread = float(np.subtract(*np.percentile(y, [75, 25])))
    spread = max(abs(spread), np.max(np.abs(y)) if y.size else 1.0, 1.0)

    low_order_width = 20.0 * spread
    high_order_width = 200.0 * spread

    bounds = []
    for n in range(order + 1):
        width = low_order_width if n <= 1 else high_order_width
        bounds.append((-width, width))
    return bounds


def _prior_transform(bounds):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def transform(cube):
        return lo + np.asarray(cube) * (hi - lo)

    return transform


def _find_map(loglikelihood, bounds, seed=None):
    """Maximum-likelihood point within ``bounds`` (= MAP, prior is flat)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = lo + rng.random(len(bounds)) * (hi - lo)
    result = minimize(
        lambda beta: -loglikelihood(beta),
        x0,
        bounds=bounds,
        method="L-BFGS-B",
    )
    return result.x, float(-result.fun)


def run_one_nested_fit(loglikelihood, bounds, labels, nlive=500, dlogz=0.1, seed=None, outroot=None):
    """One dynesty run over a bounded-uniform prior.

    Returns
    -------
    dict
        ``log_evidence``, ``log_evidence_err`` (dynesty's own, see
        :func:`run_seed_scatter` for the more trustworthy scatter-based
        version), ``map_position``, ``map_loglikelihood``, ``peak_shortfall``,
        ``converged``, ``flat_samples``, ``labels``, ``bounds``.
    """
    import dynesty
    from dynesty.utils import resample_equal

    ndim = len(bounds)
    map_position, map_loglikelihood = _find_map(loglikelihood, bounds, seed=seed)

    transform = _prior_transform(bounds)
    rstate = np.random.default_rng(seed)
    sampler = dynesty.NestedSampler(
        loglikelihood, transform, ndim, nlive=nlive, rstate=rstate
    )
    sampler.run_nested(print_progress=False, dlogz=dlogz)
    results = sampler.results

    peak_shortfall = float(map_loglikelihood - np.max(results.logl))
    converged = bool(results.logzerr[-1] < 1.0 and peak_shortfall < PEAK_SHORTFALL_GATE)
    if not converged:
        logging.warning(
            f"Nested sampling of {labels} did not converge: "
            f"peak_shortfall={peak_shortfall:.2f} nats, "
            f"log_evidence_err={float(results.logzerr[-1]):.3f}. Raise --nlive."
        )

    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / weights.sum()
    kish = float(1.0 / np.sum(weights**2))
    flat_samples = resample_equal(results.samples, weights, rstate=np.random.default_rng(int(rstate.integers(2**31))))[
        : max(int(kish), 2)
    ]

    if outroot is not None:
        from .mcmc_utils import plot_mcmc_results

        plot_mcmc_results(flat_samples=flat_samples, labels=labels, fname=outroot + "_corner.jpg")

    return {
        "log_evidence": float(results.logz[-1]),
        "log_evidence_err": float(results.logzerr[-1]),
        "map_position": map_position,
        "map_loglikelihood": map_loglikelihood,
        "peak_shortfall": peak_shortfall,
        "converged": converged,
        "flat_samples": flat_samples,
        "labels": labels,
        "bounds": bounds,
    }


def laplace_cross_check(loglikelihood, bounds, map_position, step=1e-3):
    """Numerical-Hessian Laplace approximation of log Z at ``map_position``.

    ``log Z ~= log L(theta*) + log pi(theta*) + (d/2) log(2 pi) - (1/2) log|det(-H)|``,
    where ``H`` is the Hessian of the log-likelihood (the prior is flat within
    ``bounds``, so it contributes only its constant ``-log(volume)`` and
    nothing to the Hessian) and ``theta*`` is ``map_position``.

    Central-difference Hessian, adapted from this codebase's own
    ``_laplace_sigma`` pattern (``test_eccentricity.py``) but returning
    ``log Z`` itself rather than marginal sigmas. Independent, fast sanity
    check on the sampler's evidence -- meant to catch a gross bug, not to
    validate dynesty's accuracy.

    Returns
    -------
    float
        NaN (with a warning logged) if the Hessian is not negative-definite
        at ``map_position`` -- can happen if the optimizer landed at a
        boundary or on a poorly-conditioned likelihood.
    """
    n = len(map_position)
    hessian = np.zeros((n, n))
    centre = loglikelihood(map_position)

    for i in range(n):
        for j in range(i, n):
            step_i, step_j = np.zeros(n), np.zeros(n)
            step_i[i] = step_j[j] = step
            if i == j:
                hessian[i, i] = (
                    loglikelihood(map_position + step_i)
                    - 2 * centre
                    + loglikelihood(map_position - step_i)
                ) / step**2
            else:
                hessian[i, j] = hessian[j, i] = (
                    loglikelihood(map_position + step_i + step_j)
                    - loglikelihood(map_position + step_i - step_j)
                    - loglikelihood(map_position - step_i + step_j)
                    + loglikelihood(map_position - step_i - step_j)
                ) / (4 * step**2)

    sign, logdet = np.linalg.slogdet(-hessian)
    if sign <= 0:
        logging.warning(
            "Laplace cross-check: Hessian is not negative-definite at the MAP "
            "(sign={sign}) -- returning NaN instead of a wrong log Z.".format(sign=sign)
        )
        return float("nan")

    log_prior_volume = float(np.sum([np.log(hi - lo) for lo, hi in bounds]))
    return float(centre - log_prior_volume + 0.5 * n * np.log(2 * np.pi) - 0.5 * logdet)


def run_seed_scatter(loglikelihood, bounds, labels, n_seeds=3, nlive=500, dlogz=0.1, outroot=None):
    """Repeat :func:`run_one_nested_fit` across seeds; the real uncertainty is
    the scatter across seeds, not any one run's own quoted error.

    Matches the documented finding in :mod:`ell1fit.nested_sampling` that
    dynesty's own ``log_evidence_err`` can be far too small on a correlated
    posterior -- measured there, not re-measured here, but the same
    conditioning risk applies to any nested-sampling run on a non-trivial
    posterior.

    Returns
    -------
    dict
        ``log_evidence`` (mean across seeds), ``log_evidence_err`` (std
        across seeds if ``n_seeds > 1``, else that one run's own dynesty
        error), ``log_evidence_dynesty_err`` (mean of each run's own quoted
        error, for reference), ``runs`` (list of each seed's full result
        dict), plus ``map_position``, ``peak_shortfall``, ``converged``,
        ``flat_samples``, ``labels`` from the first run (for plotting/MAP
        reporting -- the seeds target the same posterior, so any one run's
        samples are a representative draw from it).
    """
    seeds = np.random.default_rng(12345).integers(0, 2**31, size=n_seeds)
    runs = [
        run_one_nested_fit(loglikelihood, bounds, labels, nlive=nlive, dlogz=dlogz, seed=int(seed))
        for seed in seeds
    ]

    evidences = np.array([r["log_evidence"] for r in runs])
    dynesty_errs = np.array([r["log_evidence_err"] for r in runs])
    scatter_err = float(np.std(evidences, ddof=1)) if n_seeds > 1 else float(dynesty_errs[0])

    if outroot is not None:
        from .mcmc_utils import plot_mcmc_results

        plot_mcmc_results(
            flat_samples=runs[0]["flat_samples"], labels=labels, fname=outroot + "_corner.jpg"
        )

    return {
        "log_evidence": float(np.mean(evidences)),
        "log_evidence_err": scatter_err,
        "log_evidence_dynesty_err": float(np.mean(dynesty_errs)),
        "runs": runs,
        "map_position": runs[0]["map_position"],
        "map_loglikelihood": runs[0]["map_loglikelihood"],
        "peak_shortfall": max(r["peak_shortfall"] for r in runs),
        "converged": all(r["converged"] for r in runs),
        "flat_samples": runs[0]["flat_samples"],
        "labels": labels,
        "bounds": bounds,
    }


def bayes_factor(result_m0, result_m1):
    """ln BF (M1 over M0) from two :func:`run_seed_scatter` results, with a
    Jeffreys'-scale text interpretation (Kass & Raftery 1995 grading of
    ``2 * ln BF``).

    Returns
    -------
    dict
        ``ln_bf``, ``ln_bf_err`` (seed-scatter errors combined in
        quadrature), ``interpretation``.
    """
    ln_bf = result_m1["log_evidence"] - result_m0["log_evidence"]
    ln_bf_err = float(np.hypot(result_m0["log_evidence_err"], result_m1["log_evidence_err"]))

    grade = 2 * abs(ln_bf)
    if grade < 2:
        strength = "not worth more than a bare mention"
    elif grade < 6:
        strength = "positive"
    elif grade < 10:
        strength = "strong"
    else:
        strength = "very strong"
    favored = "M1 (PBDOT+PBDDOT)" if ln_bf > 0 else "M0 (PBDOT only)"
    interpretation = f"{strength} evidence for {favored}" if grade >= 2 else "inconclusive"

    return {"ln_bf": float(ln_bf), "ln_bf_err": ln_bf_err, "interpretation": interpretation}
