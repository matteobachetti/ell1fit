"""MCMC sampling and posterior-summary utilities for ell1fit."""

import logging
import os

import corner
import emcee
import numpy as np
from astropy.time import Time

from .plotting import plot_style_context


__all__ = [
    "calculate_result_array_from_samples",
    "default_moves",
    "get_flat_samples",
    "plot_mcmc_results",
    "safe_run_sampler",
]


def get_flat_samples(sampler):
    """Extract flattened post-burn-in MCMC samples from an emcee sampler.

    Burn-in and thinning are derived from the maximum estimated autocorrelation
    time.

    Returns
    -------
    flat_samples : np.ndarray
        Flattened chain with shape ``(nsamples, ndim)``.
    maxtau : float
        Maximum integrated autocorrelation time across parameters.
    """
    tau = sampler.get_autocorr_time(quiet=True)
    maxtau = np.max(tau)
    burnin = int(2 * maxtau)
    thin = int(0.5 * maxtau)
    flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    # log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)
    logging.info("burn-in: {0}".format(burnin))
    logging.info("thin: {0}".format(thin))
    logging.info("flat chain shape: {0}".format(flat_samples.shape))
    logging.info("flat log prob shape: {0}".format(log_prob_samples.shape))
    return flat_samples, maxtau


def calculate_result_array_from_samples(sampler, labels):
    """Summarize posterior samples into percentile-based result fields.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler or emcee.backends.HDFBackend
        Sampler/backend exposing ``get_chain`` and ``get_log_prob`` APIs.
    labels : list of str
        Parameter labels used as prefixes in output keys.

    Returns
    -------
    result_dict : dict
        Dictionary of parameter percentiles and sampling metadata.
    flat_samples : np.ndarray
        Flattened posterior samples.
    """
    flat_samples, maxtau = get_flat_samples(sampler)
    result_dict = {}
    ndim = flat_samples.shape[1]
    percs = [1, 10, 16, 50, 84, 90, 99]
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], percs)
        for i_p, p in enumerate(percs):
            result_dict[labels[i] + f"_{p:g}"] = mcmc[i_p]

    result_dict["date"] = Time.now().mjd
    result_dict["nsamples"] = flat_samples.shape[0]
    result_dict["maxtau"] = maxtau
    result_dict["burnin"] = maxtau
    result_dict["thin"] = maxtau

    return result_dict, flat_samples


def plot_mcmc_results(
    sampler=None,
    backend=None,
    flat_samples=None,
    labels=None,
    fname="results.jpg",
    **plot_kwargs,
):
    """Create a corner plot from posterior samples.

    Samples can be supplied directly, via a live sampler, or from an emcee HDF5
    backend file.
    """
    assert np.any([a is not None for a in [sampler, backend, flat_samples]]), (
        "At least one between backend, sampler, or flat_samples, should be specified, in",
        "increasing order of priority",
    )

    if flat_samples is None:
        if sampler is None:
            assert os.path.exists(backend), "Backend file does not exist"
            sampler = emcee.backends.HDFBackend(backend)
            assert sampler.iteration > 0, "Backend is empty"

        flat_samples, _ = get_flat_samples(sampler)

    with plot_style_context():
        fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], **plot_kwargs)
        fig.savefig(fname, dpi=300)


def default_moves():
    """The proposal mix this sampler uses, and why it is not emcee's default.

    Differential evolution proposes along the difference between two other
    walkers, so its steps line up with whatever direction the ensemble is
    currently spread over. On a correlated posterior that is the ridge itself,
    where the stretch move -- which only walks toward one other walker -- makes
    far less use of the same information. The 0.8/0.2 mix with the snooker
    variant is emcee's own recommendation for correlated targets.

    Measured on three benchmark posteriors with
    ``tools/sampler_bench.py``, three seeds each, effective samples **per
    step**: 0.212 to 0.689 at fixture scale, 0.158 to 0.425 at production
    scale, and 0.075 to 0.307 on a ten-parameter fit with eccentricity free.
    The cost per step did not move -- both moves make exactly one posterior
    evaluation per walker per step -- so the gain is the proposals, not
    arithmetic. Credible intervals agreed with the stretch move's within the
    Monte Carlo error on every parameter.

    Acceptance runs lower than the stretch move's, around 0.32 against 0.57.
    That is what a bolder proposal looks like and not a fault; it stays well
    clear of the thresholds that report a struggling chain below.
    """
    return [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]


def safe_run_sampler(
    func_to_maximize,
    starting_pars,
    max_n=100_000,
    outroot="chain_results",
    labels=None,
    corner_labels=None,
    n_autocorr=50,
    moves=None,
):
    """Run emcee with checkpointing, restart support, and convergence checks.

    The chain is stored in an HDF5 backend (``outroot + '.h5'``). If a previous
    chain exists, sampling resumes from the stored state. Convergence is checked
    from the integrated autocorrelation time.

    Parameters
    ----------
    moves : list or None, optional
        emcee move specification. ``None`` selects :func:`default_moves`, which
        is *not* emcee's own default -- pass ``emcee.moves.StretchMove()`` to
        get that back. A resumed chain is sampled with whatever is passed now,
        regardless of what produced the stored part; both target the same
        posterior, but an autocorrelation time measured across the join
        describes neither half.

    Returns
    -------
    dict
        Posterior summary dictionary from
        :func:`calculate_result_array_from_samples`.
    """
    # https://emcee.readthedocs.io/en/stable/tutorials/monitor/?highlight=run_mcmc#saving-monitoring-progress
    # We'll track how the average autocorrelation time estimate changes
    starting_pars = np.asarray(starting_pars)
    ndim = len(starting_pars)
    initial_jitter = 1e-6

    def _parameter_damage_report(coords, log_probs, param_labels, top_n=3):
        """Heuristic report of parameters most associated with poor walkers."""
        coords = np.asarray(coords)
        log_probs = np.asarray(log_probs)

        finite_lp = np.isfinite(log_probs)
        finite_coords = np.all(np.isfinite(coords), axis=1)
        valid = finite_lp & finite_coords

        n_total = coords.shape[0]
        n_valid = int(np.sum(valid))
        if n_valid < max(10, ndim + 2):
            return (
                f"valid_walkers={n_valid}/{n_total}; "
                "insufficient finite walkers for parameter damage diagnostics"
            )

        local_coords = coords[valid]
        local_lp = log_probs[valid]

        q25 = np.percentile(local_lp, 25)
        q75 = np.percentile(local_lp, 75)
        worst = local_lp <= q25
        best = local_lp >= q75

        if np.sum(worst) < 3 or np.sum(best) < 3:
            return (
                f"valid_walkers={n_valid}/{n_total}; "
                "insufficient best/worst walker split for diagnostics"
            )

        lp_std = np.std(local_lp)
        diagnostics = []
        for j, lbl in enumerate(param_labels):
            col = local_coords[:, j]
            spread = np.std(col)
            if not np.isfinite(spread) or spread <= 0:
                continue

            dmed = np.abs(np.median(col[worst]) - np.median(col[best]))
            score = dmed / (spread + 1e-12)

            if lp_std > 0:
                corr = np.corrcoef(col, local_lp)[0, 1]
                if not np.isfinite(corr):
                    corr = 0.0
            else:
                corr = 0.0

            diagnostics.append((score, np.abs(corr), lbl, dmed, spread, corr))

        diagnostics.sort(key=lambda x: (x[0], x[1]), reverse=True)

        if len(diagnostics) == 0:
            return f"valid_walkers={n_valid}/{n_total}; no finite parameter diagnostics"

        worst_labels = []
        for score, _, lbl, dmed, spread, corr in diagnostics[:top_n]:
            worst_labels.append(
                f"{lbl}(score={score:.2f},corr={corr:+.2f},dmed={dmed:.2e},std={spread:.2e})"
            )

        nonfinite_frac = 1.0 - n_valid / n_total
        return (
            f"valid_walkers={n_valid}/{n_total}, nonfinite_frac={nonfinite_frac:.2%}; "
            f"top_damage={'; '.join(worst_labels)}"
        )

    if labels is None:
        labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))
    if corner_labels is None:
        corner_labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))

    backend_filename = outroot + ".h5"
    backend = emcee.backends.HDFBackend(backend_filename)
    initial_size = 0
    if os.path.exists(backend_filename):
        initial_size = backend.iteration

    logging.info("Initial size: {0}".format(initial_size))
    # backend.reset(nwalkers, ndim)
    nwalkers = max(32, starting_pars.size * 2)
    if initial_size < 100:
        logging.info("Starting from zero")

        pos = np.array(starting_pars) + np.random.normal(
            np.zeros((nwalkers, starting_pars.size)), initial_jitter
        )
        _, ndim = pos.shape
        backend.reset(nwalkers, ndim)
    elif initial_size < max_n:
        logging.info("Starting from where we left")
        reader = emcee.backends.HDFBackend(backend_filename)
        samples = reader.get_chain(discard=initial_size // 2, flat=True)

        pos = samples[-nwalkers:, :]

        nwalkers, ndim = pos.shape

        max_n = max_n - initial_size
    else:
        reader = emcee.backends.HDFBackend(backend_filename)

        result_dict, flat_samples = calculate_result_array_from_samples(reader, labels)
        logging.info("Nothing to be done here")
        return result_dict

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        func_to_maximize,
        backend=backend,
        moves=default_moves() if moves is None else moves,
    )

    index = 0
    autocorr = np.empty(max_n)
    recent_tau_means = []
    acceptance_history = []
    low_acceptance_streak = 0
    last_tau_relative_change = np.inf
    last_plateau_relative_spread = np.inf
    converged = False

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(pos, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        mean_tau = np.mean(tau)
        max_tau = np.max(tau)
        autocorr[index] = mean_tau
        recent_tau_means.append(mean_tau)
        if len(recent_tau_means) > 5:
            recent_tau_means.pop(0)
        index += 1

        acceptance_frac = float(np.mean(sampler.acceptance_fraction))
        acceptance_history.append(acceptance_frac)
        if acceptance_frac < 0.05:
            low_acceptance_streak += 1
        else:
            low_acceptance_streak = 0

        tau_relative_change = np.inf
        if np.all(np.isfinite(old_tau)):
            tau_relative_change = np.max(np.abs(old_tau - tau) / tau)

        plateau_relative_spread = np.inf
        if len(recent_tau_means) >= 3 and np.min(recent_tau_means) > 0:
            plateau_relative_spread = (
                np.max(recent_tau_means) - np.min(recent_tau_means)
            ) / np.mean(recent_tau_means)

        last_tau_relative_change = tau_relative_change
        last_plateau_relative_spread = plateau_relative_spread

        # Check convergence
        converged = np.all(tau * n_autocorr < sampler.iteration)
        converged &= tau_relative_change < 0.05

        required_steps_estimate = float(n_autocorr * max_tau)
        remaining_steps_estimate = max(0.0, required_steps_estimate - sampler.iteration)
        convergence_progress = min(1.0, sampler.iteration / required_steps_estimate)

        logging.info(
            f"Iteration {sampler.iteration}: mean tau = {mean_tau:.3f}, "
            f"max tau = {max_tau:.3f}, tau_rel_change = {tau_relative_change:.3e}, "
            f"tau_plateau_spread5 = {plateau_relative_spread:.3e}, "
            f"acceptance = {acceptance_frac:.3f}, "
            f"target_iter~{required_steps_estimate:.0f}, remaining~{remaining_steps_estimate:.0f}, "
            f"progress={convergence_progress:.1%}, converged = {converged}"
        )

        if acceptance_frac < 0.1:
            param_damage_info = _parameter_damage_report(
                sample.coords, sample.log_prob, labels, top_n=3
            )
            logging.warning(f"Low acceptance detected: {param_damage_info}")

        if low_acceptance_streak >= 3:
            logging.warning(
                "Acceptance has been <0.05 for 3 consecutive checks. "
                "Consider tightening parameter scales (get_factors), "
                "running a minimization warm start, or resetting the backend."
            )
        if sampler.iteration % 1000 == 0:
            result_dict, flat_samples = calculate_result_array_from_samples(sampler, labels)
            logging.info(
                f"Checkpointing intermediate results to {outroot + '_corner_incomplete.jpg'}"
            )
            plot_mcmc_results(
                flat_samples=flat_samples,
                labels=labels,
                fname=outroot + "_corner_incomplete.jpg",
                backend=backend,
            )
        if converged:
            break
        old_tau = tau

    if len(acceptance_history) > 0:
        acceptance_min = float(np.min(acceptance_history))
        acceptance_max = float(np.max(acceptance_history))
    else:
        acceptance_min = np.nan
        acceptance_max = np.nan

    if np.isfinite(last_tau_relative_change) and np.isfinite(last_plateau_relative_spread):
        if last_tau_relative_change < 0.01 and last_plateau_relative_spread < 0.05:
            tau_trend = "stable"
        elif last_tau_relative_change < 0.05 and last_plateau_relative_spread < 0.20:
            tau_trend = "approaching-plateau"
        else:
            tau_trend = "still-evolving"
    else:
        tau_trend = "insufficient-checks"

    logging.info(
        f"Final convergence summary: iterations = {sampler.iteration}, "
        f"checks = {index}, converged = {converged}, tau_trend = {tau_trend}, "
        f"last_tau_rel_change = {last_tau_relative_change:.3e}, "
        f"last_tau_plateau_spread5 = {last_plateau_relative_spread:.3e}, "
        f"acceptance_range = [{acceptance_min:.3f}, {acceptance_max:.3f}]"
    )

    result_dict, flat_samples = calculate_result_array_from_samples(sampler, labels)
    plot_mcmc_results(
        flat_samples=flat_samples,
        labels=labels,
        fname=outroot + "_corner.jpg",
        backend=backend,
    )

    return result_dict
