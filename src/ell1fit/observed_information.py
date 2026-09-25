r"""Observed information of each observation about the shared orbital parameters.

:mod:`ell1fit.information` does the linear algebra; this module supplies the
matrices, by differentiating the very log likelihood the fit maximises. Nothing
is re-derived analytically, so whatever the fit actually optimises -- the
weights, the template, the deorbiting tolerance -- is what gets differentiated.

The shape of the problem
------------------------

A multi-file run fits one binary and many spins. ``setup.parameter_names``
mixes both: ``A1``, ``EPS1`` and ``EPS2`` carry no index and belong to
everybody, while ``F0_3``, ``F1_3`` and ``Phase_3`` belong to file 3 and appear
nowhere else in the likelihood. :func:`split_shared_and_local` separates them on
that trailing index, which is the only thing distinguishing them.

Because file 3's spin parameters enter only file 3's term, the full Hessian is
an arrow matrix and each observation can be handled entirely on its own: build
the small joint Hessian over ``(shared, that file's spin)``, hand it to
:func:`ell1fit.information.schur_information`, and add the results up. One
6x6 Hessian per observation replaces one 48x48 Hessian over everything, and the
answer is the same.

Where the derivatives are taken
-------------------------------

At the point the parfiles describe, with each observation's own spin parameters
first profiled to their best value there (``profile=True``, the default). The
shared parameters are *not* re-optimised: the budget is about curvature at the
solution it is handed, and moving each file to its own private best ``A1``
would be a different -- and incoherent -- question.

Steps come from ``setup.factors``, the preconditioned scales the sampler
already uses, so one step is a fixed fraction of the expected uncertainty
whatever the parameter's units. That keeps the central differences well away
from both round-off and curvature change.

Priors count, and that is worth knowing
---------------------------------------

With ``include_priors=True`` (the default) the curvature of the priors is
included, which is what the posterior actually has. It matters more than it
looks. A uniform prior contributes nothing, but a *narrow* Gaussian on
``F0_i`` and ``F1_i`` stops those parameters absorbing the orbital signature,
and the information that would have been eaten survives into ``A1`` instead.
Running with ``include_priors=False`` prices that: the difference between the
two is the part of the orbital constraint that comes from the spin priors
rather than from the photons.
"""

import copy
import re

import numpy as np
from scipy.optimize import minimize

from .information import schur_information
from .phase_utils import _calculate_phases


__all__ = [
    "segment_information",
    "segment_log_likelihood",
    "split_shared_and_local",
]


#: Central-difference step as a fraction of each parameter's preconditioned
#: scale. Large enough that the likelihood's own numerical noise is negligible
#: against the difference, small enough to stay inside the quadratic region:
#: a third of a standard deviation satisfies both by a wide margin.
DEFAULT_STEP = 0.3

_TRAILING_INDEX = re.compile(r"_(\d+)$")


def split_shared_and_local(parameter_names):
    """Separate globally shared parameters from per-file ones.

    Parameters
    ----------
    parameter_names : sequence of str
        Free parameter names as the fit setup lists them.

    Returns
    -------
    shared : list of str
        Names with no trailing file index, in their original order.
    local : dict
        ``{file_index: [names]}`` for the rest.
    """
    shared = []
    local = {}
    for name in parameter_names:
        match = _TRAILING_INDEX.search(name)
        if match is None:
            shared.append(name)
        else:
            local.setdefault(int(match.group(1)), []).append(name)
    return shared, local


def _single_file_parameters(parameters, index):
    """Re-index one file's parameters to position 0, dropping every other file's.

    :func:`ell1fit.phase_utils._calculate_phases` reads per-file values by a
    trailing ``_i`` matching the position in the list of time arrays it is
    given. Handing it one file means that file has to become file 0.
    """
    single = {}
    for key, value in parameters.items():
        match = _TRAILING_INDEX.search(key)
        if match is None:
            single[key] = value
        elif int(match.group(1)) == index:
            single[key[: match.start()] + "_0"] = value
    return single


def segment_log_likelihood(observations, setup, index):
    """Build the log likelihood of one observation, as a function of overrides.

    Parameters
    ----------
    observations : ObservationSet
    setup : FitSetup
    index : int
        Which file.

    Returns
    -------
    callable
        Takes a ``{name: value}`` mapping of parameters to change from the
        setup's own values, and returns that file's log likelihood. Names
        belonging to another file are accepted and ignored, since they cannot
        affect this term.
    """
    times = [observations.times_from_pepoch[index]]
    template = setup.template_funcs[index]
    weights = None if setup.weights is None else setup.weights[index]
    likelihood_func = setup.likelihood_func
    tolerance = setup.tolerance
    base = dict(setup.parameters)

    def log_likelihood(overrides=None):
        parameters = base if not overrides else {**base, **overrides}
        single = _single_file_parameters(parameters, index)
        phases = _calculate_phases(times, single, tolerance=tolerance)[0]
        return likelihood_func(phases, template, weights=weights)

    return log_likelihood


def _prior_terms(setup, names, include_priors):
    """Log-prior functions for the named parameters, or zeros if they are excluded."""
    if not include_priors:
        return {}
    lookup = dict(zip(setup.parameter_names, setup.logprior_funcs))
    return {name: lookup[name] for name in names if name in lookup}


def _target_function(observations, setup, index, names, include_priors):
    """Log posterior (or likelihood) of one file as a function of a value vector."""
    log_likelihood = segment_log_likelihood(observations, setup, index)
    priors = _prior_terms(setup, names, include_priors)

    def target(values):
        overrides = dict(zip(names, values))
        total = log_likelihood(overrides)
        for name, logp in priors.items():
            total += logp(overrides[name])
        return total

    return target


def _hessian(target, centre, steps):
    """Central-difference Hessian of ``target`` at ``centre``."""
    n = len(centre)
    centre = np.asarray(centre, dtype=float)
    steps = np.asarray(steps, dtype=float)
    hessian = np.zeros((n, n))

    at_centre = target(centre)
    plus = np.empty(n)
    minus = np.empty(n)
    for i in range(n):
        shift = np.zeros(n)
        shift[i] = steps[i]
        plus[i] = target(centre + shift)
        minus[i] = target(centre - shift)
        hessian[i, i] = (plus[i] - 2.0 * at_centre + minus[i]) / steps[i] ** 2

    for i in range(n):
        for j in range(i + 1, n):
            shift_i = np.zeros(n)
            shift_i[i] = steps[i]
            shift_j = np.zeros(n)
            shift_j[j] = steps[j]
            mixed = (
                target(centre + shift_i + shift_j)
                - target(centre + shift_i - shift_j)
                - target(centre - shift_i + shift_j)
                + target(centre - shift_i - shift_j)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = hessian[j, i] = mixed

    return hessian


def _profile_local(target, centre, steps, n_shared):
    """Optimise the per-file parameters with the shared ones held fixed."""
    n_local = len(centre) - n_shared
    if n_local == 0:
        return centre
    local_steps = steps[n_shared:]

    def to_minimize(scaled):
        trial = np.array(centre, dtype=float)
        trial[n_shared:] = centre[n_shared:] + scaled * local_steps
        return -target(trial)

    result = minimize(to_minimize, np.zeros(n_local), method="Nelder-Mead")
    profiled = np.array(centre, dtype=float)
    profiled[n_shared:] = centre[n_shared:] + result.x * local_steps
    return profiled


def segment_information(
    observations,
    setup,
    shared=None,
    include_priors=True,
    step=DEFAULT_STEP,
    profile=True,
):
    """Observed information of every observation about the shared parameters.

    Parameters
    ----------
    observations : ObservationSet
    setup : FitSetup
    shared : sequence of str or None, optional
        Which shared parameters to report on. Defaults to every free parameter
        without a file index.
    include_priors : bool, optional
        Add the curvature of the priors, as the posterior has it. See the
        module docstring for why turning this off is informative.
    step : float, optional
        Central-difference step, as a fraction of each parameter's
        preconditioned scale.
    profile : bool, optional
        Optimise each file's own spin parameters before differentiating.

    Returns
    -------
    list of SegmentInformation
        One per input file, named after it, in input order.
    """
    all_shared, local = split_shared_and_local(setup.parameter_names)
    if shared is None:
        shared = all_shared
    else:
        unknown = [name for name in shared if name not in all_shared]
        if unknown:
            raise ValueError(
                f"Parameters {unknown} are not shared free parameters of this fit; "
                f"it shares {all_shared}"
            )
    if not shared:
        raise ValueError(
            "This fit has no shared parameters: every free parameter carries a file "
            "index, so there is nothing for an information budget to be about."
        )

    factors = dict(zip(setup.parameter_names, setup.factors))
    parameters = copy.deepcopy(setup.parameters)

    segments = []
    for index, path in enumerate(observations.files):
        names = list(shared) + local.get(index, [])
        centre = np.array([parameters[name] for name in names], dtype=float)
        steps = np.array([step * abs(factors[name]) for name in names], dtype=float)

        target = _target_function(observations, setup, index, names, include_priors)
        if profile:
            centre = _profile_local(target, centre, steps, len(shared))

        information = -_hessian(target, centre, steps)
        segments.append(schur_information(information, len(shared), name=str(path)))

    return segments
