"""Likelihood utilities for phase-based pulsar timing fits."""

import numpy as np
from numba import njit
from stingray.pulse.pulsar import z_n_events


@njit()
def _pc_like_weight(probs, weights):
    """Compute weighted Pletsch-Clarke log-likelihood contribution.

    Parameters
    ----------
    probs : np.ndarray
        Template probabilities at event phases.
    weights : np.ndarray
        Per-event weights in [0, 1].

    Returns
    -------
    float
        Summed weighted log-likelihood.
    """
    like = 0.0
    for i in range(probs.size):
        like += np.log(weights[i] * probs[i] + (1 - weights[i]))
    return like


@njit()
def _pc_like(probs):
    """Compute unweighted Pletsch-Clarke log-likelihood contribution.

    Parameters
    ----------
    probs : np.ndarray
        Template probabilities at event phases.

    Returns
    -------
    float
        Summed log-likelihood.
    """
    like = 0.0
    for i in range(probs.size):
        like += np.log(probs[i])
    return like


def pletsch_clarke_likelihood(phases, template_func, weights=None):
    """Evaluate the Pletsch-Clarke profile likelihood for event phases.

    Parameters
    ----------
    phases : np.ndarray
        Event phases.
    template_func : callable
        Pulse-template function returning probability density at each phase.
    weights : np.ndarray or None, optional
        Event weights in [0, 1]. If provided, uses weighted log-likelihood.

    Returns
    -------
    float
        Total log-likelihood.
    """
    # Templates that can score events themselves do it in one fused pass, which
    # avoids three full-length temporaries and uses compensated summation. The
    # generic path below stays for any other callable -- notably the analytic
    # single-harmonic template and anything a test supplies.
    fused = getattr(template_func, "loglike", None)
    if fused is not None:
        return fused(phases, weights=weights)

    probs = template_func(phases)
    probs = np.asarray(probs, dtype=float)
    probs = np.nan_to_num(probs, nan=1e-12, posinf=1e12, neginf=1e-12)
    probs = np.clip(probs, 1e-12, None)

    if weights is None:
        return _pc_like(probs)

    local_weights = np.asarray(weights, dtype=float)
    local_weights = np.nan_to_num(local_weights, nan=0.0, posinf=1.0, neginf=0.0)
    local_weights = np.clip(local_weights, 0.0, 1.0)
    return _pc_like_weight(probs, local_weights)


def rayleigh_as_likelihood(phases, template_func=None, weights=None):
    """Use the Rayleigh statistic :math:`Z_1^2` as a surrogate for a likelihood.

    Selected by ``--likelihood Rayleigh``. It measures how concentrated the
    event phases are at the fundamental frequency, and depends on **nothing but
    the phases** -- which is both its appeal and its limitation.

    Warnings
    --------
    This is a *statistic*, not a log-likelihood, and the difference has
    consequences a user should know about before trusting the output:

    - **Only the fundamental harmonic is used.** The pulse template is not
      consulted at all, so ``-N``/``nharm`` has no effect here. A sharply peaked
      pulse carries information in its higher harmonics that this discards.
    - **Per-event weights are ignored**, so ``--use-weight`` does nothing.
    - **It is not on a log-density scale.** The pipeline adds a log-prior to
      whatever this returns, and since :math:`Z_1^2` is not a log-probability,
      the relative weight of prior against data is not meaningful. On a typical
      dataset :math:`Z_1^2` is of order +1700 where the Pletsch-Clarke
      log-likelihood is around -970: different scale and different sign
      convention.

    Consequently the MCMC run on top of this does not sample a posterior in the
    usual sense, and its credible intervals should not be read as such. Prefer
    ``--likelihood PC`` unless there is a specific reason to want a
    template-free statistic.

    :func:`ell1fit.ell1fit.ell1fit` warns when this is combined with options it
    cannot honour, rather than letting them be silently discarded.

    Parameters
    ----------
    phases : np.ndarray
        Event phases.
    template_func : callable or None, optional
        Accepted for interface compatibility and deliberately unused.
    weights : np.ndarray or None, optional
        Accepted for interface compatibility and deliberately unused.

    Returns
    -------
    float
        :math:`Z_1^2` for the input phases.
    """
    return z_n_events(phases, 1)
