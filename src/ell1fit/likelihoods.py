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


def rayleigh_as_likelihood(phases, *args, **kwargs):
    """Use the Rayleigh test statistic as a surrogate likelihood.

    Parameters
    ----------
    phases : np.ndarray
        Event phases.

    Returns
    -------
    float
        Z1^2 value for the input phases.
    """
    prob = z_n_events(phases, 1)
    return prob
