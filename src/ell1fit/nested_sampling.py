"""Nested sampling for ell1fit: the evidence, and the Bayes factor it enables.

Why this exists
----------------
Neither the ensemble sampler nor NUTS produces ``log Z`` -- both only ever look
at *differences* of the log-posterior, never its integral. An eccentricity
*detection* needs exactly that integral: fit once with ``EPS1``/``EPS2`` free
and once with them fixed at zero, and the difference of the two ``log_evidence``
fields is the Bayes factor. This module is what ``--sampler nested`` calls;
:func:`run_nested` is the production entry point.

Nothing here is imported at package load time -- :func:`run_nested` imports
``dynesty`` lazily and raises a clear ``ImportError`` pointing at
``pip install ell1fit[nested]`` if it is missing. :mod:`ell1fit.prior_transform`
needs only numpy and scipy, already core dependencies.

The evidence needs the *likelihood* alone, not the posterior
--------------------------------------------------------------
Every other sampler in this package takes ``logprior + loglikelihood`` together,
because an MCMC only ever looks at differences of it. Nested sampling cannot:
it draws from the prior through :mod:`ell1fit.prior_transform` and integrates
the likelihood against it, so handing it the posterior would count the prior
twice -- and would count the *unnormalised* version of it at that, since
``ell1fit.priors._flat_logprior`` returns 0 rather than ``-log(width)``.
:func:`split_loglikelihood` recovers the likelihood by subtracting the
package's own log-prior from its own posterior, rather than rebuilding the
likelihood from :func:`ell1fit.likelihoods.pletsch_clarke_likelihood` directly
-- so there is no second copy of that composition to drift.
"""

import logging

import numpy as np


#: Nats the sampler's best likelihood is allowed to fall short of the
#: optimizer's before ``log Z`` is flagged as unreliable. Measured need: at
#: nlive=200 a ten-parameter fit missed its mode by 112 nats and reported a
#: confident, wrong ``log Z = -32.0 +- 0.30`` with nothing in dynesty's own
#: diagnostics to say so -- this is the guard that turns that silent failure
#: loud. nlive=1000 still missed the mode in 2 of 3 attempts on that problem;
#: nlive=4000 found it in 5 of 5. Raise ``--nlive`` well past the default on
#: anything with more than a handful of free parameters.
PEAK_SHORTFALL_GATE = 1.0


def split_loglikelihood(observations, setup, logpost):
    """Separate the log-likelihood from the log-prior.

    Parameters
    ----------
    logpost : callable
        The posterior being split -- ``func_to_maximize`` from
        :func:`ell1fit.posterior._build_posterior_functions`.

    Returns
    -------
    loglikelihood : callable
    logprior : callable
    """
    from .posterior import _build_posterior_functions

    logprior, _, _ = _build_posterior_functions(observations, setup)

    def loglikelihood(position):
        prior = logprior(position)
        if not np.isfinite(prior):
            # Outside the prior's support the likelihood is never consulted, and
            # ``-inf - -inf`` would be a NaN that dynesty cannot interpret.
            return -np.inf
        posterior = logpost(position)
        if not np.isfinite(posterior):
            # A non-invertible orbit, which the posterior rejects outright.
            return -np.inf
        return posterior - prior

    return loglikelihood, logprior


def check_loglikelihood_split(
    setup, start, logpost, loglikelihood, logprior, n_probes=32, seed=90212
):
    """Confirm the split recomposes into the posterior it came from.

    Probes are drawn from the prior itself rather than around the peak: that is
    where nested sampling spends most of its evaluations, and it is the region a
    check centred on the MAP would never visit.
    """
    from .prior_transform import build_prior_transform

    transform, _ = build_prior_transform(setup, check=False)
    ndim = len(start)
    rng = np.random.default_rng(seed)
    worst = 0.0
    compared = 0
    for _ in range(n_probes):
        position = transform(rng.random(ndim))
        recomposed = loglikelihood(position) + logprior(position)
        reference = logpost(position)
        if not np.isfinite(reference):
            continue
        compared += 1
        worst = max(worst, abs(recomposed - reference))
    if compared == 0:
        raise AssertionError("Every probe drawn from the prior had an infinite posterior")
    if worst > 1e-9:
        raise AssertionError(f"Log-likelihood split does not recompose: worst {worst:.3e}")
    return {"probes_compared": compared, "worst_recomposition_error": worst}


#: Set once per worker process by :func:`_init_nested_worker`, read by
#: :func:`_worker_loglikelihood`. A pool under macOS's ``spawn`` start method
#: pickles whatever it sends a worker, and a closure over ``observations``/
#: ``setup`` cannot be pickled -- so nothing captured is sent. Instead each
#: worker gets the two picklable objects once, at start-up, and rebuilds its
#: own local closure from them; only this module-level trampoline, which
#: closes over nothing, ever needs to survive a pickle.
_WORKER_LOGLIKELIHOOD = None


def _init_nested_worker(observations, setup):
    """Rebuild this worker's own log-likelihood from picklable inputs.

    Run once per worker by the pool's own ``initializer`` mechanism. Cheap:
    this only wires together numbers and templates ``observations``/``setup``
    already hold -- the same assembly
    :func:`ell1fit.posterior._build_posterior_functions` does in the main
    process -- not the event-file loading or template refinement that built
    them in the first place.
    """
    global _WORKER_LOGLIKELIHOOD
    from .posterior import _build_posterior_functions

    _, _, logpost = _build_posterior_functions(observations, setup)
    _WORKER_LOGLIKELIHOOD, _ = split_loglikelihood(observations, setup, logpost)


def _worker_loglikelihood(position):
    """Trampoline dynesty can pickle to a worker: no captured state, just a global read."""
    return _WORKER_LOGLIKELIHOOD(position)


def run_nested(
    observations,
    setup,
    func_to_maximize,
    starting_pars,
    outroot="chain_results",
    labels=None,
    corner_labels=None,
    nlive=1000,
    dlogz=0.1,
    bound="multi",
    nested_sample="auto",
    workers=0,
    seed=None,
):
    """Run ``dynesty`` nested sampling, and summarize like the other samplers.

    The production counterpart of :func:`ell1fit.mcmc_utils.safe_run_sampler`
    and :func:`ell1fit.nuts_sampling.run_nuts`, called when
    ``optimize_solution(..., sampler="nested")``.

    **This is a capability, not a speed play.** It is the only sampler here
    that produces ``log_evidence``, and that is the only reason to choose it
    -- it will lose on every rate against the other two, ``workers`` or not.

    ``nlive`` matters more than it looks like it should: nested sampling can
    miss a narrow mode entirely and still report a tidy error bar, with
    nothing in dynesty's own diagnostics to say so. See
    :data:`PEAK_SHORTFALL_GATE`. Raise ``nlive`` on anything with more than a
    handful of free parameters.

    Parameters
    ----------
    workers : int, optional
        Worker processes to spread likelihood evaluations across. ``0`` (the
        default) runs single-process. Unlike the benchmark harness, which
        rebuilds a whole synthetic problem in each worker because that is
        cheap for a seeded fixture, a worker here is handed the already-built
        ``observations``/``setup`` and only rebuilds the likelihood closure
        from them -- no event-file I/O, no template refinement repeated per
        worker. The prior transform stays evaluated in the main process: it
        is already picklable data (see :mod:`ell1fit.prior_transform`) and a
        handful of microseconds of numpy, cheaper to run here than to ship a
        cube point to a worker and back.

        Not free to turn on. Measured on a small two-parameter fixture,
        4 workers were a net *slowdown* (23 s against 17 s single-process at
        ``nlive=500``): shipping a position to a worker and a likelihood back
        costs more than the fixture's likelihood call did. Reach for
        ``workers`` on a fit expensive enough that one likelihood call is
        milliseconds, not microseconds -- a large event count or several free
        parameters -- not by default.

    Returns
    -------
    dict
        Same shape as
        :func:`ell1fit.mcmc_utils.calculate_result_array_from_samples`
        (``label_p`` percentiles, ``date``, ``nsamples``), plus
        ``log_evidence``, ``log_evidence_err``, ``information_nats``,
        ``kish_effective_samples``, ``peak_shortfall``, and ``converged``.
    """
    try:
        import dynesty
        from dynesty.utils import resample_equal
    except ImportError as exc:
        raise ImportError("sampler='nested' needs dynesty: pip install ell1fit[nested]") from exc

    from .mcmc_utils import plot_mcmc_results
    from .prior_transform import build_prior_transform

    starting_pars = np.asarray(starting_pars)
    ndim = len(starting_pars)
    if labels is None:
        labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))

    transform, omitted_log_normalisation = build_prior_transform(setup)
    loglikelihood, logprior = split_loglikelihood(observations, setup, func_to_maximize)
    split_check = check_loglikelihood_split(
        setup, starting_pars, func_to_maximize, loglikelihood, logprior
    )
    map_loglikelihood = float(loglikelihood(starting_pars))

    seed = int(np.random.default_rng().integers(2**31)) if seed is None else seed

    pool = None
    sampler_loglikelihood = loglikelihood
    if workers:
        import multiprocessing

        pool = multiprocessing.get_context("spawn").Pool(
            processes=workers,
            initializer=_init_nested_worker,
            initargs=(observations, setup),
        )
        sampler_loglikelihood = _worker_loglikelihood

    try:
        sampler = dynesty.NestedSampler(
            sampler_loglikelihood,
            transform,
            ndim,
            nlive=nlive,
            bound=bound,
            sample=nested_sample,
            rstate=np.random.default_rng(seed),
            pool=pool,
            queue_size=workers if workers else None,
            use_pool={"prior_transform": False} if workers else None,
        )
        sampler.run_nested(print_progress=False, dlogz=dlogz)
        results = sampler.results
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    # The guard that matters: see PEAK_SHORTFALL_GATE.
    peak_shortfall = float(map_loglikelihood - np.max(results.logl))
    converged = bool(results.logzerr[-1] < 1.0 and peak_shortfall < PEAK_SHORTFALL_GATE)
    if not converged:
        logging.warning(
            f"Nested sampling did not converge: peak_shortfall={peak_shortfall:.2f} nats, "
            f"log_evidence_err={float(results.logzerr[-1]):.3f}. Raise --nlive before "
            "trusting log_evidence from this run."
        )

    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / weights.sum()
    kish = float(1.0 / np.sum(weights**2))
    flat_samples = resample_equal(results.samples, weights, rstate=np.random.default_rng(seed + 1))[
        : max(int(kish), 2)
    ]

    result_dict = {}
    percs = [1, 10, 16, 50, 84, 90, 99]
    for i in range(ndim):
        mcmc_percentiles = np.percentile(flat_samples[:, i], percs)
        for i_p, p in enumerate(percs):
            result_dict[labels[i] + f"_{p:g}"] = mcmc_percentiles[i_p]

    from astropy.time import Time

    result_dict["date"] = Time.now().mjd
    result_dict["nsamples"] = flat_samples.shape[0]
    result_dict["log_evidence"] = float(results.logz[-1])
    result_dict["log_evidence_err"] = float(results.logzerr[-1])
    result_dict["information_nats"] = float(results.information[-1])
    result_dict["kish_effective_samples"] = kish
    result_dict["peak_shortfall"] = peak_shortfall
    result_dict["converged"] = converged
    result_dict["omitted_log_normalisation"] = omitted_log_normalisation
    result_dict["loglikelihood_split_check"] = split_check

    plot_mcmc_results(
        flat_samples=flat_samples,
        labels=corner_labels or labels,
        fname=outroot + "_corner.jpg",
    )

    return result_dict
