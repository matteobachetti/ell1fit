"""Iterative refinement of the pulse template and the timing solution.

The problem
-----------
The pulse template is built by folding events with the timing solution from the
input parfile. If that solution is imperfect, the fold is smeared -- and the
smearing is not symmetric. Orbital-parameter errors produce *structured* phase
residuals across an observation, so the resulting template comes out skewed,
not merely broadened.

That distinction matters. A symmetric blur costs only precision, because the
free nuisance parameter ``Phase_i`` absorbs a pure offset. A skew does not get
absorbed: the fit is then comparing events against a template whose shape is
itself wrong in a direction correlated with the error being fitted, which is a
self-consistency **bias**, not just a loss of significance.

The fix
-------
Refold with the improved solution and rebuild the template, repeating until the
solution stops moving. Each pass costs one deorbit-and-fold plus one bounded
point-estimate fit; the expensive MCMC still runs only once, afterwards, seeded
at the converged position with the sharper template.

The opposite risk
-----------------
Iterating a template against the same data it was fitted to can *overfit noise*:
the template starts absorbing noise features, which then pull the solution --
a bias in the opposite direction from the one being corrected. This is mild at
low ``nharm`` and real at high ``nharm``, so the harmonic count is held fixed
while iterating, and :func:`ell1fit.tests.test_refinement` measures the residual
bias against an injected truth rather than assuming it is negligible.

Two safeguards follow from that:

* **Convergence is judged in parameter space**, as the size of the point
  estimate's step in local coordinates, where one standard deviation is
  ``1e-6`` units for every parameter. That common scale is what makes a single
  threshold meaningful across ``F0``, ``PB``, ``A1`` and ``TASC`` at once -- and
  it holds only because :func:`ell1fit.scaling.precondition_factors` enforces
  it; the raw factors from :func:`~ell1fit.scaling.get_factors` differ between
  directions by a factor of a thousand.
* **The best iterate is kept, not the last.** Refinement is not guaranteed to be
  monotonic: a point-estimate step can wander and produce a worse fold than the
  one before. Scoring each pass by the summed profile :math:`Z^2_n` and keeping
  the best costs nothing and removes the failure mode entirely.
"""

import dataclasses
import logging

import numpy as np
from stingray.pulse.pulsar import z_n_binned_events

from .fitting import point_estimate_fit
from .scaling import TARGET_LOCAL_SIGMA
from .phase_utils import folded_profile
from .templates import create_template_from_profile_harm, get_template_func

#: Convergence threshold on the point estimate's step, in local coordinates.
#:
#: The units matter and are easy to get wrong. Local coordinates follow the
#: convention that one standard deviation is
#: :data:`~ell1fit.scaling.TARGET_LOCAL_SIGMA` = 1e-6 local units, **not** 1. An
#: earlier version of this used 0.1, believing the factors normalised each
#: parameter to order unity; that is 1e5 sigma, so convergence was declared on
#: essentially every first pass regardless of how far the solution had moved.
#:
#: A tenth of a sigma is small enough that a further pass cannot meaningfully
#: change the answer, and large enough not to chase numerical noise.
CONVERGENCE_TOLERANCE = 0.1 * TARGET_LOCAL_SIGMA


def _profile_score(profiles, nharm):
    """Summed :math:`Z^2_n` across files: how well a fold concentrates the pulse."""
    return float(np.sum([z_n_binned_events(p, nharm) for p in profiles]))


def _templates_from_profiles(profiles, nharm, final_nbin=200):
    """Build one template function per file, without writing diagnostic figures."""
    template_funcs = []
    phase_offsets = []
    for profile in profiles:
        template, additional_phase = create_template_from_profile_harm(
            profile,
            nharm=nharm,
            final_nbin=final_nbin,
            plot=False,
        )
        template_funcs.append(get_template_func(template))
        phase_offsets.append(additional_phase)
    return template_funcs, phase_offsets


def refine_templates_and_solution(
    observations,
    setup,
    nbin,
    nharm,
    max_iterations=1,
    tolerance=CONVERGENCE_TOLERANCE,
):
    """Alternate between rebuilding the template and refitting the solution.

    Parameters
    ----------
    observations : ObservationSet
        The event data.
    setup : FitSetup
        Starting configuration, carrying the templates built from the input
        parfile's solution.
    nbin : int
        Phase bins used when refolding.
    nharm : int
        Harmonics retained in the template. Held fixed across iterations, to
        limit the noise-overfitting risk described in the module docstring.
    max_iterations : int, optional
        Maximum refinement passes. ``1`` means a single fit against the initial
        template -- identical to not refining at all.
    tolerance : float, optional
        Convergence threshold on ``max_k |dtheta_k| / factor_k``.

    Returns
    -------
    setup : FitSetup
        The best configuration found, with refined templates and a re-centred
        baseline. Ready to hand to :func:`ell1fit.fitting.optimize_solution`.
    history : list of dict
        One entry per pass, recording ``score``, ``max_shift`` and
        ``converged``, so that convergence can be inspected rather than assumed.
    """
    history = []

    if max_iterations <= 1:
        return setup, history

    times = observations.times_from_pepoch

    best_setup = setup
    best_score = _profile_score(
        folded_profile(times, setup.parameters, nbin=nbin, tolerance=setup.tolerance),
        nharm,
    )
    logging.info(f"Template refinement: initial profile score Z^2 = {best_score:.1f}")

    current = setup
    for iteration in range(1, max_iterations + 1):
        fit_pars, fitted_parameters, _ = point_estimate_fit(observations, current)

        # How far the solution moved, in units of each parameter's own scale.
        max_shift = float(np.max(np.abs(fit_pars))) if len(fit_pars) else 0.0

        profiles = folded_profile(
            times, fitted_parameters, nbin=nbin, tolerance=current.tolerance
        )
        score = _profile_score(profiles, nharm)

        template_funcs, _ = _templates_from_profiles(profiles, nharm)

        candidate = dataclasses.replace(
            current,
            parameters=fitted_parameters,
            baseline_values=[fitted_parameters[p] for p in current.parameter_names],
            template_funcs=template_funcs,
        )

        converged = max_shift < tolerance
        history.append({"iteration": iteration, "score": score, "max_shift": max_shift,
                        "converged": converged})
        logging.info(
            f"Template refinement pass {iteration}/{max_iterations}: "
            f"Z^2 = {score:.1f}, max|dtheta|/factor = {max_shift:.3g}, "
            f"converged = {converged}"
        )

        # Keep the best iterate, not the last: refinement is not guaranteed to
        # improve monotonically.
        if score > best_score:
            best_score = score
            best_setup = candidate

        current = candidate

        if converged:
            break
    else:
        logging.warning(
            f"Template refinement did not converge in {max_iterations} passes "
            f"(last max|dtheta|/factor = {history[-1]['max_shift']:.3g}, "
            f"tolerance = {tolerance}). Using the best iterate found; consider "
            "raising --template-iterations or checking the input solution."
        )

    logging.info(f"Template refinement: best profile score Z^2 = {best_score:.1f}")
    return best_setup, history
