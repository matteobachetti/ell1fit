"""The end-to-end ELL1 timing-fit pipeline.

Stages, in the order they run and the order they depend on each other:

1. **Load** the timing models and the event files
   (:mod:`ell1fit.models`, :mod:`ell1fit.events`).
2. **Fold** events into pulse profiles and, optionally, derive energy-dependent
   event weights (:mod:`ell1fit.weighting`).
3. **Build templates** from those profiles (:mod:`ell1fit.templates`). This must
   precede the next step: the per-file ``Phase_i`` offset the templates
   determine is what the phase prior is centred on.
4. **Assemble the fit**: expand the requested parameter names, attach priors and
   scaling (:mod:`ell1fit.priors`, :mod:`ell1fit.scaling`).
5. **Condition** the parameter scales against the actual posterior, so that
   every fitted direction has a comparable local step size.
6. **Refine** templates and solution together, if asked
   (:mod:`ell1fit.refinement`).
7. **Optimize and sample** (:mod:`ell1fit.fitting`), then write result tables,
   updated parfiles and diagnostic plots (:mod:`ell1fit.results_io`).

The ordering in steps 3-5 is not arbitrary and has been got wrong before: see
the comments at each call site in :func:`ell1fit`.

:func:`ell1fit` is the entry point; the command-line interface that wraps it is
in :mod:`ell1fit.cli`.
"""

import dataclasses
import logging
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from stingray.pulse.pulsar import z_n_binned_events, z_n_gauss

from . import version
from .create_parfile import update_model
from .events import _load_events_for_all_files
from .fitting import optimize_solution
from .likelihoods import pletsch_clarke_likelihood
from .likelihoods import rayleigh_as_likelihood
from .models import _build_parameters_from_models
from .models import _load_and_validate_models
from .outputs import _get_outroots
from .outputs import _make_outroot_getter
from .phase_utils import _calculate_phases
from .phase_utils import ell1_truncation_error
from .phase_utils import folded_profile
from .phase_utils import phases_around_zero
from .plotting import plot_style_context as _plot_style_context
from .posterior import _build_posterior_functions
from .posterior import _trace_phase_0_likelihood
from .priors import assign_logpriors
from .templates import create_template_from_profile_harm
from .templates import estimate_weighted_profile_std
from .templates import get_template_func
from .refinement import refine_templates_and_solution
from .results_io import safe_save
from .results_io import split_output_results
from .scaling import get_factors
from .scaling import precondition_factors
from .setup_types import FitSetup
from .setup_types import ObservationSet
from .weighting import pf_weight_versus_energy

__all__ = [
    "ELL1_TRUNCATION_WARNING_FRACTION",
    "UNFITTABLE_PARAMETERS",
    "_prepare_fit_setup",
    "ell1fit",
]


freq_re = re.compile(r"^d?F([0-9]+)_([0-9]+)$")


#: Parameters that appear in the parameter dictionary but that the phase model
#: is flat in, mapped to the reason. Fitting one of these would add a dimension
#: the likelihood does not depend on, so the chain would simply return the
#: prior -- and the result table would report that prior as a measurement.
UNFITTABLE_PARAMETERS = {
    "PBDOT": (
        "the phase model holds the orbital period constant. PINT applies PBDOT "
        "once, when each parfile's binary epoch is aligned to its PEPOCH, and "
        "nothing downstream of that depends on it"
    ),
}


def _reject_unfittable_parameters(requested_parameter_names):
    """Refuse to fit parameters the likelihood is flat in.

    Raises
    ------
    ValueError
        If any requested parameter appears in :data:`UNFITTABLE_PARAMETERS`.
    """
    for par in requested_parameter_names:
        if par in UNFITTABLE_PARAMETERS:
            raise ValueError(
                f"{par} cannot be fitted: {UNFITTABLE_PARAMETERS[par]}. "
                f"Fitting it would sample the prior rather than the data. "
                f"Set {par} in the parfile instead."
            )


def _reject_unmeasurable_derivatives(parameters, fit_parameter_names):
    """Refuse to fit an orbital derivative the epochs give no lever arm for.

    :data:`UNFITTABLE_PARAMETERS` rejects what the *model* is flat in.
    ``A1DOT`` is different: the model depends on it perfectly well, but only
    through ``A1DOT * binary_dt_i``, so a dataset whose files all share one
    epoch -- a single file, most obviously -- makes the likelihood flat in it
    anyway, and the chain would report the prior as a measurement.

    Raises
    ------
    ValueError
        If ``A1DOT`` is requested and every lever arm is zero.
    """
    if "A1DOT" not in fit_parameter_names:
        return

    levers = [value for name, value in parameters.items() if name.startswith("binary_dt_")]
    if not any(levers):
        raise ValueError(
            "A1DOT cannot be fitted from a single epoch: it enters the phase model only "
            "as A1DOT times the separation between each file's epoch and the reference, "
            "which is zero here, so the likelihood is flat in it and the fit would return "
            "the prior. Give it files from at least two different epochs."
        )


#: Warn once the ELL1 truncation reaches this fraction of the phase precision
#: the data support. A systematic at one third of the statistical error inflates
#: the total by 5%, which is about where it stops being ignorable.
ELL1_TRUNCATION_WARNING_FRACTION = 1 / 3


def _warn_on_eccentric_orbit(parameters, profiles, nharm):
    """Warn when the orbit is too eccentric for ELL1 to describe at this precision.

    ELL1 exists for nearly circular orbits and expands the Roemer delay in
    eccentricity, so it stops being a faithful description at large ``e``. Where
    exactly depends on the data, not on ``e`` alone: the truncation matters only
    once it is comparable to the precision the observation supports.

    The comparison here is
    :func:`ell1fit.phase_utils.ell1_truncation_error` against
    ``1 / (2 pi sqrt(sum Z^2_n))``, the phase precision implied by the folded
    profiles. That estimator was checked against the fitted ``Phase_i``
    uncertainty over a sixfold range of event counts and two pulse shapes and
    tracks it to within 30%, always on the low side -- so this errs toward
    warning early rather than late.

    Nothing is rejected. Exceeding the limit degrades sensitivity rather than
    biasing the eccentricity, so the fit remains meaningful; the user just needs
    to know the model is working outside its range.
    """
    truncation = ell1_truncation_error(
        parameters["EPS1"], parameters["EPS2"], parameters["A1"], parameters["F0_0"]
    )
    if truncation <= 0:
        return

    total_z2 = float(np.sum([z_n_binned_events(profile, nharm) for profile in profiles]))
    if not np.isfinite(total_z2) or total_z2 <= 0:
        return
    phase_precision = 1 / (2 * np.pi * np.sqrt(total_z2))

    if truncation > ELL1_TRUNCATION_WARNING_FRACTION * phase_precision:
        eccentricity = np.hypot(parameters["EPS1"], parameters["EPS2"])
        warnings.warn(
            f"Eccentricity {eccentricity:.3g} is large enough that the ELL1 "
            f"expansion limits this fit: its truncation leaves {truncation:.2g} "
            f"cycles against a phase precision of {phase_precision:.2g} cycles. "
            "ELL1 is a small-eccentricity model; the unmodelled residual sits in "
            "the third harmonic of the orbit, so it costs sensitivity rather "
            "than biasing the recovered eccentricity. A full Keplerian model "
            "(BT, DD) would be the right description here.",
            stacklevel=2,
        )


def _collect_parameter_names(parameters, requested_parameter_names, likelihood_func):
    """Expand the user-requested parameter tokens into per-file fit parameter names.

    Raises
    ------
    ValueError
        If a requested name matches nothing in the parameter dictionary. It used
        to be dropped in silence, so ``-P F0,A1DOT`` fitted ``F0`` alone and said
        nothing -- and so did the typo ``-P TSAC``. The resulting fit is
        internally consistent and simply answers a different question than the
        one asked, which nothing downstream can detect.
    """
    fit_parameter_names = []
    for f in parameters:
        if f.startswith("Phase") and likelihood_func == pletsch_clarke_likelihood:
            fit_parameter_names.append(f)
            continue

        for g in requested_parameter_names:
            # Startswith alone was confusing PBDOT for PB
            if f == g or (f.startswith(g) and freq_re.match(f)):
                fit_parameter_names.append(f)

    # A per-file parameter is requested by its bare name (``F0`` for ``F0_3``,
    # ``Phase`` for ``Phase_3``), so a name is known if it *or* any of its
    # per-file expansions is in the dictionary -- not by exact key alone.
    unknown = [
        g
        for g in requested_parameter_names
        if not any(f == g or f.startswith(f"{g}_") for f in parameters)
    ]
    if unknown:
        raise ValueError(
            f"Cannot fit {', '.join(unknown)}: not in the timing model. "
            f"Available parameters: {', '.join(sorted(parameters))}."
        )

    return fit_parameter_names


def _enrich_results_with_observation_metadata(
    results,
    model,
    times_from_pepoch,
    pepoch,
    files,
    expo,
    pulsed_frac,
    profile,
    nharm,
    energy_range,
    nsteps,
):
    """Attach observation-level metadata fields to fit results."""
    n_files = len(files)

    for i in range(n_files):
        if getattr(model[i], "START", None) is not None and model[i].START.value is not None:
            results[f"Start_{i}"] = model[i].START.value
        else:
            results[f"Start_{i}"] = times_from_pepoch[i][0] / 86400 + pepoch[i]

        if getattr(model[i], "STOP", None) is not None and model[i].STOP.value is not None:
            results[f"Stop_{i}"] = model[i].STOP.value
        else:
            results[f"Stop_{i}"] = times_from_pepoch[i][-1] / 86400 + pepoch[i]

        results[f"PEPOCH_{i}"] = pepoch[i]
        results[f"fname_{i}"] = files[i]

    results["nharm"] = nharm
    results["emin"] = 0 if energy_range is None else energy_range[0]
    results["emax"] = np.inf if energy_range is None else energy_range[1]
    results["nsteps"] = nsteps

    for i in range(n_files):
        results[f"pf_{i}"] = pulsed_frac[i]
        results[f"Z2{nharm}_{i}"] = z_n_binned_events(profile[i], nharm)
        results[f"ctrate_{i}"] = times_from_pepoch[i].size / expo[i]

    results["ell1fit_version"] = version.version
    return results


def _write_results_products(results, n_files, get_outroot, requested_parameter_names, model):
    """Write combined and per-file result tables plus updated parfiles."""
    table_results = Table(rows=[results])
    output_file = get_outroot(None) + "_results.ecsv"
    safe_save(table_results, output_file)

    split_tables = split_output_results(table_results, n_files, requested_parameter_names)
    for i, table in enumerate(split_tables):
        outfile = get_outroot(i) + "_results.ecsv"
        table.write(outfile, overwrite=True)
        logging.info(f"Writing {outfile}")
        logging.info(table)

        outpar = get_outroot(i) + "_results.par"
        new_model = update_model(model[i], table[-1])
        logging.info(f"Writing model to {outpar}")
        with open(outpar, "w") as fobj:
            print(new_model.as_parfile(include_info=os.name != "nt"), file=fobj)

    return output_file


def _undilute_template(template, weights):
    """Rescale a weighted fold into the profile a fully weighted event follows.

    The weighted fold is the profile of the weighted *ensemble*, but the
    likelihood's model for event ``i`` is ``1 + w_i (T - 1)`` -- a copy of the
    template diluted by that event's own weight. ``T`` must therefore be the
    *undiluted* profile, the one a ``w = 1`` event follows. That profile's
    modulation is ``sum(w a) / sum(w^2)``, against the fold's
    ``sum(w a) / sum(w)``, so the fold is short by ``sum(w) / sum(w^2)`` and the
    likelihood would otherwise dilute it a second time. Left uncorrected, the
    fitted phase uncertainties come out well over 50% too wide.

    Only the deviation from the mean is scaled, so the result does not depend on
    how :func:`create_template_from_profile_harm` normalizes its output.
    """
    local_weights = np.asarray(weights, dtype=float)
    total_square = float(np.sum(local_weights**2))
    if not np.isfinite(total_square) or total_square <= 0:
        return template
    undilute = float(np.sum(local_weights)) / total_square
    level = template.mean()
    return level + undilute * (template - level)


def _prepare_templates_and_phase_priors(
    profile,
    profile_weight,
    use_weight,
    nharm,
    get_outroot,
    files,
    weights,
    nbin,
    parameters,
    parameters_with_unc,
):
    """Create templates, log diagnostics, and record each file's phase-zero offset.

    Must run before priors/scaling are assigned (:func:`_prepare_fit_setup`):
    the per-file ``Phase_i`` offset computed here is the value
    :func:`ell1fit.priors.assign_logpriors` centers that parameter's prior on.
    """
    n_files = len(profile)
    template_func = []
    pulsed_frac = []

    for i in range(n_files):
        template_raw, additional_phase_raw = create_template_from_profile_harm(
            profile[i],
            nharm=nharm,
            final_nbin=200,
            imagefile=get_outroot(i) + "_template_raw.jpg",
        )

        if use_weight:
            template, additional_phase = create_template_from_profile_harm(
                profile_weight[i],
                nharm=nharm,
                final_nbin=200,
                imagefile=get_outroot(i) + "_template.jpg",
            )
            template = _undilute_template(template, weights[i])
        else:
            template = template_raw
            additional_phase = additional_phase_raw

        logging.info(f"File {files[i]}: ")
        logging.info("  Profile:")
        logging.info(f"  + phase = {additional_phase_raw:.4f}")

        z2 = z_n_binned_events(profile[i], nharm)
        logging.info(f"  + Z^2_{nharm} = {z2:.1f}")
        pulsed_fraction = (
            (template_raw.max() - template_raw.min())
            / (template_raw.max() + template_raw.min())
            * 100
        )
        logging.info(f"  + pulsed fraction = {pulsed_fraction:.1f}%")
        pulsed_fraction_z2 = np.sqrt(2 * z2 / np.sum(profile[i])) * 100
        logging.info(f"  + pulsed fraction from Z^2_{nharm} = {pulsed_fraction_z2:.1f}%")

        if use_weight:
            logging.info("  Weighted profile:")
            logging.info(f"  + phase = {additional_phase:.4f}")

            err = estimate_weighted_profile_std(weights[i], nbin=nbin, ntrials=400)
            weighted_z2 = z_n_gauss(profile_weight[i], err=err, n=nharm)
            logging.info(f"  + Z^2_{nharm} = {weighted_z2:.1f}")
            weighted_pulsed_fraction = (
                (template.max() - template.min()) / (template.max() + template.min()) * 100
            )
            logging.info(f"  + pulsed fraction (weighted) = {weighted_pulsed_fraction:.1f}%")

        template_func.append(get_template_func(template))
        mint = template.min()
        maxt = template.max()
        pulsed_frac.append((maxt - mint) / (maxt + mint))

        ph0 = -phases_around_zero(additional_phase)
        parameters[f"Phase_{i}"] = ph0
        parameters_with_unc[f"Phase_{i}"][0] = ph0

    return template_func, pulsed_frac, parameters, parameters_with_unc


def _build_profiles_and_weights(
    times_from_pepoch,
    parameters,
    energies,
    n_files,
    get_outroot,
    use_weight,
    nbin,
    tolerance,
):
    """Fold profiles and optionally compute energy-based event weights."""
    profile = folded_profile(times_from_pepoch, parameters, nbin=nbin, tolerance=tolerance)

    weights = None
    if use_weight:
        weights = pf_weight_versus_energy(
            times_from_pepoch,
            energies,
            parameters,
            nharm=1,
            tolerance=tolerance,
            plot_root_file_name=[get_outroot(i) + "_pf_weight_spectrum" for i in range(n_files)],
        )

        profile_weight = folded_profile(
            times_from_pepoch, parameters, weights, nbin=nbin, tolerance=tolerance
        )
        # This comparison was previously drawn but never saved or closed, so it
        # produced nothing and leaked a figure per file. Saving it makes it the
        # diagnostic it was evidently meant to be; closing it matters because
        # iterative refinement calls this repeatedly.
        with _plot_style_context():
            for i, (p, pw) in enumerate(zip(profile, profile_weight)):
                fig = plt.figure(figsize=(3.5, 2.65))
                plt.plot(np.concatenate((p, p)) / p.max(), label="unweighted")
                plt.plot(np.concatenate((pw, pw)) / pw.max(), label="weighted")
                plt.xlabel("Phase bin (two cycles)")
                plt.ylabel("Normalized counts")
                plt.legend()
                plt.savefig(get_outroot(i) + "_weighted_profile_comparison.jpg")
                plt.close(fig)
    else:
        profile_weight = profile

    return profile, profile_weight, weights


def _prepare_fit_setup(
    parameters,
    requested_parameter_names,
    likelihood_func,
    parameters_with_unc,
    observation_length,
    model,
    template_funcs=None,
    weights=None,
    tolerance=1e-8,
):
    """Collect fit parameters, priors, factors, and initial fit values.

    Returns
    -------
    FitSetup
        The bundle defining what is being fitted. Must be built *after* the
        templates exist: the per-file ``Phase_i`` offset they determine is what
        :func:`ell1fit.priors.assign_logpriors` centres that parameter's prior
        on.
    """
    fit_parameter_names = _collect_parameter_names(
        parameters,
        requested_parameter_names,
        likelihood_func=likelihood_func,
    )
    _reject_unmeasurable_derivatives(parameters, fit_parameter_names)
    logprior_funcs = assign_logpriors(
        fit_parameter_names,
        parameters_with_unc,
        obs_length=observation_length,
    )
    factors = get_factors(
        fit_parameter_names,
        model,
        observation_length,
        parameters_with_unc=parameters_with_unc,
    )

    try:
        input_mean_fit_pars = [parameters[par] for par in fit_parameter_names]
    except KeyError as exc:
        raise ValueError("One or more parameters are missing from the parameter file") from exc

    return FitSetup(
        parameter_names=fit_parameter_names,
        baseline_values=input_mean_fit_pars,
        logprior_funcs=logprior_funcs,
        factors=factors,
        template_funcs=template_funcs,
        parameters=parameters,
        likelihood_func=likelihood_func,
        weights=weights,
        tolerance=tolerance,
    )


def ell1fit(
    files,
    parfiles,
    nsteps=100,
    nharm=1,
    tolerance=1e-8,
    energy_range=None,
    fit_parameters=["F0"],
    minimize_first=False,
    general_outroot=None,
    likelihood_func=pletsch_clarke_likelihood,
    use_weight=False,
    use_pi=False,
    ignore_uncertainties=False,
    template_iterations=1,
    sampler="emcee",
    nlive=1000,
    dlogz=0.1,
    workers=0,
):
    """Fit spin and ELL1 orbital parameters from one or more event files.

    This is the high-level pipeline used by the CLI:

    1. Load timing models and event files.
    2. Fold events and build pulse templates (optionally energy-weighted).
    3. Build priors and parameter scaling.
    4. Perform posterior optimization and MCMC sampling.
    5. Save ECSV summaries, diagnostic plots, and updated ``.par`` files.

    Parameters
    ----------
    files : list of str
        Event files to analyze.
    parfiles : list of str
        PINT-compatible ELL1 parameter files, one per event file.
    nsteps : int, optional
        Maximum MCMC steps.
    nharm : int, optional
        Number of harmonics used to model pulse profiles.
    tolerance : float, optional
        Deorbiting tolerance in seconds.
    energy_range : tuple or None, optional
        Energy selection ``(emin, emax)`` applied to events.
    fit_parameters : list of str, optional
        Parameters to fit (e.g. ``["F0", "F1", "PB"]``).
    minimize_first : bool, optional
        If True, run a local minimization before MCMC.
    general_outroot : str or None, optional
        Base output root; otherwise inferred from input filename.
    likelihood_func : callable, optional
        Likelihood/statistic function evaluated on phases.
    use_weight : bool, optional
        If True, apply energy-dependent event weighting.
    use_pi : bool, optional
        If True, base that weighting on PI channels instead of calibrated
        energy. Has no effect unless ``use_weight`` is also True; the
        ``energy_range`` selection above is always applied in calibrated
        energy regardless of this flag.
    ignore_uncertainties : bool, optional
        If True, ignore uncertainties from input parfiles when building priors.
    template_iterations : int, optional
        Maximum passes of template refinement. The template is initially built
        by folding with the input parfile's solution; if that solution is
        imperfect the fold is smeared and, because orbital errors produce
        structured rather than random phase residuals, the template comes out
        skewed. Refolding with the improved solution and rebuilding removes
        that. ``1`` (the default) disables refinement entirely and is
        bit-identical to not having the feature. See
        :mod:`ell1fit.refinement`.
    sampler : {"emcee", "nuts", "nested"}, optional
        Posterior-exploration backend -- see
        :func:`ell1fit.fitting.optimize_solution`.
    nlive, dlogz, workers : optional
        Only consulted when ``sampler="nested"`` -- see
        :func:`ell1fit.fitting.optimize_solution`.

    Returns
    -------
    str
        Path to the combined output ECSV file.
    """
    n_files = len(files)
    assert len(parfiles) == len(files), (
        "The number of parameter files must match that of event files."
    )

    # The Rayleigh statistic depends only on the phases: it consults neither the
    # pulse template nor per-event weights. Silently dropping options the user
    # explicitly asked for is worse than not offering them, so say so.
    if likelihood_func is rayleigh_as_likelihood:
        if use_weight:
            warnings.warn(
                "--use-weight has no effect with --likelihood Rayleigh: the "
                "Rayleigh statistic ignores per-event weights. Use --likelihood PC "
                "to make use of energy weighting.",
                stacklevel=2,
            )
        if nharm > 1:
            warnings.warn(
                f"-N/--nharm {nharm} has no effect on the fit with --likelihood "
                "Rayleigh: the Rayleigh statistic uses only the fundamental "
                "harmonic. It still sets the binning of the diagnostic profiles.",
                stacklevel=2,
            )
    model, pepoch, ref_model = _load_and_validate_models(parfiles)

    nbin = max(32, nharm * 8)

    requested_parameter_names = sorted(fit_parameters)
    _reject_unfittable_parameters(requested_parameter_names)
    get_outroot = _make_outroot_getter(
        files,
        requested_parameter_names,
        energy_range,
        nharm,
        likelihood_func,
        use_weight,
        use_pi=use_pi,
        general_outroot=general_outroot,
    )

    times_from_pepoch, observation_length, energies, expo = _load_events_for_all_files(
        files,
        energy_range,
        pepoch,
        get_outroot,
        use_pi=use_pi,
    )

    observations = ObservationSet(
        files=files,
        models=model,
        ref_model=ref_model,
        pepoch=pepoch,
        times_from_pepoch=times_from_pepoch,
        energies=energies,
        exposures=expo,
        observation_length=observation_length,
    )

    parameters_with_unc, parameters = _build_parameters_from_models(
        model,
        ref_model,
        observation_length,
        ignore_uncertainties=ignore_uncertainties,
    )

    profile, profile_weight, weights = _build_profiles_and_weights(
        times_from_pepoch,
        parameters,
        energies,
        n_files,
        get_outroot,
        use_weight,
        nbin,
        tolerance,
    )

    # Needs the folded profiles: how much eccentricity ELL1 can carry depends on
    # the precision the data support, not on a fixed threshold.
    _warn_on_eccentric_orbit(parameters, profile, nharm)

    # Must run before _prepare_fit_setup: it writes each file's real Phase_i
    # offset into parameters/parameters_with_unc, which assign_logpriors then
    # centers that parameter's prior on.
    (
        template_func,
        pulsed_frac,
        parameters,
        parameters_with_unc,
    ) = _prepare_templates_and_phase_priors(
        profile,
        profile_weight,
        use_weight,
        nharm,
        get_outroot,
        files,
        weights,
        nbin,
        parameters,
        parameters_with_unc,
    )

    setup = _prepare_fit_setup(
        parameters,
        requested_parameter_names,
        likelihood_func,
        parameters_with_unc,
        observation_length,
        model,
        template_funcs=template_func,
        weights=weights if use_weight else None,
        tolerance=tolerance,
    )

    outroots = _get_outroots(get_outroot, n_files)

    _trace_phase_0_likelihood(observations, setup, outroot=outroots[-1])
    # _trace_phase_0_likelihood moves each Phase_i to its best-fit value, so
    # re-centre the local coordinate system before handing it to the optimizer.
    setup = setup.with_baseline_from(parameters)

    # Put every fitted direction on a comparable local scale before anything
    # tries to optimize or sample. Done after the Phase_i trace, which scans in
    # units of the pre-existing factors, and before refinement, so that both the
    # point estimates and the MCMC see a well-conditioned problem.
    setup = dataclasses.replace(
        setup,
        factors=precondition_factors(
            _build_posterior_functions(observations, setup)[2],
            setup.factors,
            setup.n_parameters,
        ),
    )
    logging.info(
        "Preconditioned parameter scales: "
        + ", ".join(f"{n}={f:.4g}" for n, f in zip(setup.parameter_names, setup.factors))
    )

    # Capture the "before" phases now, while the baseline still describes the
    # solution this run started from. Refinement re-centres the baseline on its
    # own result, so deriving these later -- as evaluating the posterior at
    # local zero would -- yields the refined solution instead, and the
    # comparison phaseograms end up showing it against itself.
    reference_phases = _calculate_phases(
        observations.times_from_pepoch, setup.parameters, tolerance=setup.tolerance
    )

    # Refold and rebuild the template against the improved solution, so the
    # template the MCMC uses is not the one smeared by the input parfile's
    # errors. A single iteration is a no-op by construction.
    setup, refinement_history = refine_templates_and_solution(
        observations,
        setup,
        nbin=nbin,
        nharm=nharm,
        max_iterations=template_iterations,
    )

    results = optimize_solution(
        observations,
        setup,
        nsteps=nsteps,
        minimize_first=minimize_first,
        outroots=outroots,
        reference_phases=reference_phases,
        sampler=sampler,
        nlive=nlive,
        dlogz=dlogz,
        workers=workers,
    )
    results["template_iterations"] = template_iterations
    results["template_passes_run"] = len(refinement_history)
    if refinement_history:
        results["template_final_shift"] = refinement_history[-1]["max_shift"]
        results["template_converged"] = refinement_history[-1]["converged"]

    results = _enrich_results_with_observation_metadata(
        results,
        model,
        times_from_pepoch,
        pepoch,
        files,
        expo,
        pulsed_frac,
        profile,
        nharm,
        energy_range,
        nsteps,
    )

    return _write_results_products(results, n_files, get_outroot, requested_parameter_names, model)
