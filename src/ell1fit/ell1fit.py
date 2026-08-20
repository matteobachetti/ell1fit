"""Utilities and pipeline to fit ELL1 pulsar timing parameters from event data.

This module implements an end-to-end workflow for X-ray pulsar timing in
binary systems modeled with ELL1 orbital parameters.

Main stages are:

1. Load and pre-process event lists.
2. Deorbit event times and compute pulse phases.
3. Build pulse templates and evaluate profile likelihoods.
4. Fit selected spin/orbital parameters with optional minimization.
5. Explore the posterior with MCMC and save diagnostics/results.

The high-level entry point is :func:`ell1fit`; :func:`main` exposes the CLI.
"""

import logging
import os
import re

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
from .logging import configure_logging
from .models import _build_parameters_from_models
from .models import _load_and_validate_models
from .outputs import _get_outroots
from .outputs import _make_outroot_getter
from .phase_utils import folded_profile
from .phase_utils import phases_around_zero
from .plotting import plot_style_context as _plot_style_context
from .posterior import _trace_phase_0_likelihood
from .priors import assign_logpriors
from .templates import create_template_from_profile_harm
from .templates import estimate_weighted_profile_std
from .templates import get_template_func
from .results_io import safe_save
from .results_io import split_output_results
from .scaling import get_factors
from .setup_types import FitSetup
from .setup_types import ObservationSet
from .weighting import pf_weight_versus_energy

freq_re = re.compile(r"^d?F([0-9]+)_([0-9]+)$")


def _collect_parameter_names(parameters, requested_parameter_names, likelihood_func):
    """Expand the user-requested parameter tokens into per-file fit parameter names."""
    fit_parameter_names = []
    for f in parameters:
        if f.startswith("Phase") and likelihood_func == pletsch_clarke_likelihood:
            fit_parameter_names.append(f)
            continue

        for g in requested_parameter_names:
            # Startswith alone was confusing PBDOT for PB
            if f == g or (f.startswith(g) and freq_re.match(f)):
                fit_parameter_names.append(f)

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
            nbin=32,
            nharm=1,
            tolerance=tolerance,
            plot_root_file_name=[get_outroot(i) + "_pf_weight_spectrum" for i in range(n_files)],
        )

        profile_weight = folded_profile(
            times_from_pepoch, parameters, weights, nbin=nbin, tolerance=tolerance
        )
        with _plot_style_context():
            for p, pw in zip(profile, profile_weight):
                plt.figure()
                plt.plot(np.concatenate((p, p)) / p.max())
                plt.plot(np.concatenate((pw, pw)) / pw.max())
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

    Returns
    -------
    str
        Path to the combined output ECSV file.
    """
    n_files = len(files)
    assert len(parfiles) == len(
        files
    ), "The number of parameter files must match that of event files."
    model, pepoch, ref_model = _load_and_validate_models(parfiles)

    nbin = max(32, nharm * 8)

    requested_parameter_names = sorted(fit_parameters)
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

    results = optimize_solution(
        observations,
        setup,
        nsteps=nsteps,
        minimize_first=minimize_first,
        outroots=outroots,
    )
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


def main(args=None):
    """Main function called by the `ell1fit` script"""
    import argparse

    configure_logging()

    description = "Fit an ELL1 model and frequency derivatives to an X-ray " "pulsar observation."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "-p",
        "--parfile",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Input parameter files, one per event file. Must contain a simple ELL1 binary model, "
            "with no orbital derivatives, and a number of spin derivatives (F0, F1, ...). "
            "All other models will be ignored."
        ),
    )
    parser.add_argument("-o", "--outroot", type=str, default=None, help="Root of output file names")
    parser.add_argument(
        "-N",
        "--nharm",
        type=int,
        help="Number of harmonics to describe the pulse profile",
        default=1,
    )
    parser.add_argument(
        "--deorb-tolerance",
        type=float,
        help="Tolerance of deorbit operation, in seconds",
        default=1e-8,
    )
    parser.add_argument(
        "-E",
        "--erange",
        nargs=2,
        type=float,
        help="Energy range",
        default=None,
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        help="Maximum number of MCMC steps",
        default=100_000,
    )
    parser.add_argument(
        "-P",
        "--parameters",
        type=str,
        help="Comma-separated list of parameters to fit",
        default="F0,F1",
    )
    parser.add_argument(
        "--likelihood",
        type=str,
        help="Can be PC (Pletsch & Clarke, default) or Rayleigh",
        default="PC",
    )
    parser.add_argument(
        "--minimize-first",
        action="store_true",
        default=False,
        help="Minimize first, then MCMC (don't trust the solution in the par file)",
    )
    parser.add_argument(
        "--use-weight",
        action="store_true",
        default=False,
        help="Use pulse energy dependence of profile as weight",
    )
    parser.add_argument(
        "--use-pi",
        action="store_true",
        default=False,
        help=(
            "Base pulsed-fraction weighting (--use-weight) on PI channels instead of "
            "calibrated energy. No effect without --use-weight."
        ),
    )
    parser.add_argument("--ignore-uncertainties", action="store_true", default=False)

    args = parser.parse_args(args)
    files = args.files
    parfiles = args.parfile

    like = pletsch_clarke_likelihood
    if args.likelihood.lower() == "rayleigh":
        like = rayleigh_as_likelihood

    ell1fit(
        files,
        parfiles,
        nsteps=args.nsteps,
        nharm=args.nharm,
        tolerance=args.deorb_tolerance,
        energy_range=args.erange,
        fit_parameters=args.parameters.split(","),
        minimize_first=args.minimize_first,
        general_outroot=args.outroot,
        likelihood_func=like,
        use_weight=args.use_weight,
        use_pi=args.use_pi,
        ignore_uncertainties=args.ignore_uncertainties,
    )
