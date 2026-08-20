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

import warnings

import os
import copy
import re
import logging

import matplotlib.pyplot as plt
import numpy as np
from ell1fit import splitext_improved
from hendrics.io import load_events
from stingray.pulse.pulsar import z_n_binned_events, z_n_gauss
from stingray.stats import (
    a_from_ssig,
    z2_n_detection_level,
    power_confidence_limits,
)
from pint.models import get_model

from scipy.interpolate import interp1d

from astropy.table import Table
from scipy.optimize import minimize

from . import version
from .create_parfile import update_model
from .likelihoods import pletsch_clarke_likelihood
from .likelihoods import rayleigh_as_likelihood
from .logging import configure_logging
from .mcmc_utils import calculate_result_array_from_samples  # noqa: F401
from .mcmc_utils import get_flat_samples  # noqa: F401
from .mcmc_utils import plot_mcmc_results  # noqa: F401
from .mcmc_utils import safe_run_sampler
from .phase_utils import _calculate_phases
from .phase_utils import _mjd_to_sec
from .phase_utils import _sec_to_mjd  # noqa: F401
from .phase_utils import add_circular_orbit_numba  # noqa: F401
from .phase_utils import add_ell1_orbit_numba  # noqa: F401
from .phase_utils import fast_phase  # noqa: F401
from .phase_utils import folded_profile
from .phase_utils import interp_nb  # noqa: F401
from .phase_utils import phases_around_zero
from .phase_utils import phases_from_zero_to_one
from .phase_utils import simple_circular_deorbit_numba  # noqa: F401
from .phase_utils import simple_ell1_deorbit_numba  # noqa: F401
from .plotting import plot_style_context as _plot_style_context
from .priors import _flat_logprior, assign_logpriors
from .profile_plotting import _compare_phaseograms
from .profile_plotting import create_template_from_profile_harm
from .profile_plotting import estimate_weighted_profile_std
from .profile_plotting import get_template_func
from .profile_plotting import normalize_dyn_profile  # noqa: F401
from .results_io import _format_energy_string
from .results_io import safe_save
from .results_io import split_output_results
from .scaling import estimate_uncertainties_from_model  # noqa: F401
from .scaling import get_factors
from .scaling import order_of_magnitude  # noqa: F401

simple_freq_re = re.compile(r"^d?F([0-9]+)")
freq_re = re.compile(r"^d?F([0-9]+)_([0-9]+)$")


def pf_weight_versus_energy(
    times, energies, parameters, nbin=32, nharm=1, tolerance=1e-8, plot_root_file_name=None
):
    """Estimate per-event weights from pulse amplitude versus energy.

    For each input observation, this function computes phases with the current
    timing model, bins events in energy quantiles, and estimates pulsed
    amplitude in each energy bin from the :math:`Z_n^2` statistic. The resulting
    amplitude trend is interpolated and evaluated at each event energy to obtain
    per-event weights.

    Parameters
    ----------
    times : list of np.ndarray
        Event times (seconds from each file PEPOCH), one array per file.
    energies : list of np.ndarray
        Event energies (or PI channels if provided upstream), one array per file.
    parameters : dict
        Timing/orbital parameter dictionary consumed by
        :func:`_calculate_phases`.
    nbin : int, optional
        Number of phase bins used to evaluate :math:`Z_n^2` in each energy bin.
    nharm : int, optional
        Number of harmonics for :math:`Z_n^2` and pulsed amplitude estimation.
    tolerance : float, optional
        Convergence tolerance (seconds) for deorbiting iterations.
    plot_root_file_name : list of str or None, optional
        If provided, save one diagnostic amplitude-versus-energy plot per file
        using these roots.

    Returns
    -------
    list of np.ndarray
        Event weights for each file, aligned with ``times`` and ``energies``.
    """
    n_files = len(times)
    phases = _calculate_phases(times, parameters, tolerance=tolerance)

    weights = []
    for i in range(n_files):
        local_phases = np.array(phases[i])
        local_energies = np.array(energies[i])
        amps = []
        amp_errs = []
        limit_amps_50 = []
        limit_amps_90 = []

        est_n_bins = local_phases.size // 1000
        if est_n_bins < 15:
            est_n_bins = 15
        if est_n_bins > 25:
            est_n_bins = 25

        logging.info(
            f"Estimating the pulsed fraction in {est_n_bins} energy bins using {nharm} harmonics"
        )

        e_percentiles = np.percentile(local_energies, np.linspace(0, 100, est_n_bins + 1))
        energy_edges = np.array(list(zip(e_percentiles[:-1], e_percentiles[1:])))
        mid_energies = np.array([(e[0] + e[1]) / 2 for e in energy_edges])

        for emin, emax in energy_edges:
            filt_phases = local_phases[(local_energies >= emin) & (local_energies < emax)]

            prof = np.histogram(filt_phases, bins=np.linspace(0, 1, nbin + 1))[0]

            z_n = z_n_binned_events(prof, nharm)

            z_lims = power_confidence_limits(z_n, n=nharm, c=0.68, summed_flag=True)
            det_lev_05 = z2_n_detection_level(n=nharm, epsilon=0.5)
            det_lev_09 = z2_n_detection_level(n=nharm, epsilon=0.1)

            amp = a_from_ssig(z_n, ncounts=filt_phases.size)
            a_low = a_from_ssig(z_lims[0], ncounts=filt_phases.size)
            a_high = a_from_ssig(z_lims[1], ncounts=filt_phases.size)
            if a_low > amp or a_high / 2 > amp:
                a_low = 0

            amps.append(amp)
            amp_errs.append((amp - a_low, a_high - amp))
            limit_amps_50.append(a_from_ssig(det_lev_05, ncounts=filt_phases.size))
            limit_amps_90.append(a_from_ssig(det_lev_09, ncounts=filt_phases.size))

        amp = np.array(amps)
        amp_corr = np.copy(amp)
        amp_errs = np.array(amp_errs)

        amp_errs = [np.array(amp_errs)[:, 0], np.array(amp_errs)[:, 1]]

        limit_amps_50 = np.array(limit_amps_50)
        limit_amps_90 = np.array(limit_amps_90)
        amp_corr = np.concatenate([[0, amp_corr[0]], amp_corr, [amp_corr[-1], 0]])
        limit_amps_50 = np.concatenate(
            [[0, limit_amps_50[0]], limit_amps_50, [limit_amps_50[-1], 0]]
        )
        limit_amps_90 = np.concatenate(
            [[0, limit_amps_90[0]], limit_amps_90, [limit_amps_90[-1], 0]]
        )

        energy_points = np.concatenate(
            [
                [e_percentiles[0] - 1e-15, e_percentiles[0]],
                mid_energies,
                [e_percentiles[-1], e_percentiles[-1] + 1e-15],
            ]
        )
        # Never give less credibility than the amplitude that would be detected
        # with 50% probability from noise!
        low_amp = amp_corr < limit_amps_50
        amp_corr[low_amp] = limit_amps_50[low_amp]

        func = interp1d(energy_points, amp_corr, kind="linear", assume_sorted=True)

        fine_energy_range = np.linspace(energy_points[0], energy_points[-1], 1000)
        fine_amps = func(fine_energy_range)
        fine_amps_50 = interp1d(energy_points, limit_amps_50, kind="linear", assume_sorted=True)(
            fine_energy_range
        )
        fine_amps_90 = interp1d(energy_points, limit_amps_90, kind="linear", assume_sorted=True)(
            fine_energy_range
        )

        if plot_root_file_name is not None:
            with _plot_style_context():
                plt.figure(f"{plot_root_file_name[i]}")
                plt.errorbar(
                    mid_energies,
                    amp,
                    yerr=amp_errs,
                    xerr=[mid_energies - energy_edges[:, 0], energy_edges[:, 1] - mid_energies],
                    fmt="o",
                )
                plt.semilogx(fine_energy_range, fine_amps, color="black", label="Estimated weight")
                plt.plot(fine_energy_range, fine_amps_50, color="red", label="50% detection limit")
                plt.plot(fine_energy_range, fine_amps_90, color="grey", label="90% detection limit")
                plt.legend()
                plt.savefig(f"{plot_root_file_name[i]}.jpg")
                plt.close()

        # Normalize weights so that the maximum expected pulsed amplitude maps
        # to weight=1. This keeps the weighted likelihood well behaved.
        amp_norm = np.nanmax(fine_amps)
        if not np.isfinite(amp_norm) or amp_norm <= 0:
            warnings.warn(
                "Could not normalize pulsed-fraction weights; falling back to uniform weights."
            )
            fine_amps = np.ones_like(fine_amps)
        else:
            fine_amps = fine_amps / amp_norm
            fine_amps = np.clip(fine_amps, 0.0, 1.0)

        weight_func = interp1d(
            fine_energy_range,
            fine_amps,
            kind="linear",
            assume_sorted=True,
        )
        local_weights = np.asarray(weight_func(local_energies), dtype=float)
        local_weights = np.nan_to_num(local_weights, nan=0.0, posinf=1.0, neginf=0.0)
        local_weights = np.clip(local_weights, 0.0, 1.0)
        weights.append(local_weights)

    return weights


def _get_likelihood_suffix(likelihood_func):
    """Return output-name suffix for selected likelihood implementation."""
    if likelihood_func == rayleigh_as_likelihood:
        return "_rayleigh"
    return ""


def _get_weight_suffix(use_weight):
    """Return output-name suffix when energy weights are enabled."""
    if use_weight:
        return "_pf_weight"
    return ""


def _get_nharm_suffix(nharm):
    """Return output-name suffix for harmonic count when > 1."""
    if nharm > 1:
        return f"_N{nharm}"
    return ""


def _make_outroot_getter(
    files,
    list_parameter_names,
    energy_range,
    nharm,
    likelihood_func,
    use_weight,
    general_outroot=None,
):
    """Build a closure that returns the configured output root name."""
    energy_str = _format_energy_string(energy_range)
    nharm_str = _get_nharm_suffix(nharm)
    likelihood_str = _get_likelihood_suffix(likelihood_func)
    weight_str = _get_weight_suffix(use_weight)

    def get_outroot(file_n=None):
        if file_n is not None:
            initial_outroot = splitext_improved(files[file_n])[0]
        elif general_outroot is not None:
            initial_outroot = general_outroot
        else:
            initial_outroot = "out"

        outroot = (
            initial_outroot
            + "_"
            + "_".join(list_parameter_names)
            + energy_str
            + nharm_str
            + likelihood_str
            + weight_str
        )
        return outroot

    return get_outroot


def _collect_parameter_names(parameters, list_parameter_names, likelihood_func):
    """Collect fit parameter names matching selected tokens and phase rules."""
    parameter_names = []
    for f in parameters:
        if f.startswith("Phase") and likelihood_func == pletsch_clarke_likelihood:
            parameter_names.append(f)
            continue

        for g in list_parameter_names:
            # Startswith alone was confusing PBDOT for PB
            if f == g or (f.startswith(g) and freq_re.match(f)):
                parameter_names.append(f)

    return parameter_names


def _get_outroots(get_outroot, n_files):
    """Return per-file roots plus a final aggregate root."""
    outroots = [get_outroot(i) for i in range(n_files)]
    if n_files == 1:
        outroots += [get_outroot(0)]
    else:
        outroots += [get_outroot(None)]
    return outroots


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


def _write_results_products(results, n_files, get_outroot, list_parameter_names, model):
    """Write combined and per-file result tables plus updated parfiles."""
    table_results = Table(rows=[results])
    output_file = get_outroot(None) + "_results.ecsv"
    safe_save(table_results, output_file)

    split_tables = split_output_results(table_results, n_files, list_parameter_names)
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


def _load_and_validate_models(parfiles):
    """Load PINT models, validate ELL1 constraints, and align binary epochs."""
    model = []
    pepoch = []

    for i in range(len(parfiles)):
        model.append(get_model(parfiles[i]))
        pepoch.append(model[i].PEPOCH.value)

        if hasattr(model[i], "T0") or model[i].BINARY.value != "ELL1":
            raise ValueError("This script wants an ELL1 model, with TASC, not T0, defined")

        model[i].change_binary_epoch(pepoch[i])

    ref_model = copy.deepcopy(model[0])
    ref_model.change_binary_epoch(np.mean(pepoch))

    return model, pepoch, ref_model


def _load_events_for_all_files(files, energy_range, pepoch, get_outroot):
    """Load all event files and compute per-file exposure and duration."""
    n_files = len(files)
    times_from_pepoch = [[] for _ in range(n_files)]
    observation_length = np.zeros(n_files, dtype=float)
    energies = [[] for _ in range(n_files)]
    expo = np.zeros(n_files)

    for i in range(n_files):
        fname = files[i]
        times_from_pepoch[i], gtis, energies[i] = _load_and_format_events(
            fname,
            energy_range,
            pepoch[i],
            plotfile=get_outroot(i) + f"_lightcurve_{i}.jpg",
            return_energy=True,
            use_pi=False,
        )
        expo[i] += np.sum(np.diff(gtis, axis=1))
        observation_length[i] = times_from_pepoch[i][-1] - times_from_pepoch[i][0]

    return times_from_pepoch, observation_length, energies, expo


def _build_parameters_from_models(model, ref_model, observation_length, ignore_uncertainties=False):
    """Assemble global and per-file parameter dictionaries from timing models."""
    n_files = len(model)
    parameters_with_unc = _get_par_dict(
        ref_model,
        ignore_uncertainties=ignore_uncertainties,
        obs_length=np.min(observation_length),
    )
    del parameters_with_unc["PEPOCH"]

    for i in range(n_files):
        count = 0
        local_pars_uncs = _get_par_dict(
            model[i],
            ignore_uncertainties=ignore_uncertainties,
            obs_length=observation_length[i],
        )

        while f"F{count}" in local_pars_uncs:
            parameters_with_unc[f"F{count}_{i}"] = [
                local_pars_uncs[f"F{count}"][0],
                local_pars_uncs[f"F{count}"][1],
            ]
            if f"F{count}" in parameters_with_unc:
                del parameters_with_unc[f"F{count}"]
            count += 1

        parameters_with_unc[f"PEPOCH_{i}"] = [
            local_pars_uncs["PEPOCH"][0],
            local_pars_uncs["PEPOCH"][1],
        ]
        parameters_with_unc[f"Phase_{i}"] = [
            parameters_with_unc["Phase"][0],
            parameters_with_unc["Phase"][1],
        ]

    # _calculate_phases expects file-specific phase keys.
    del parameters_with_unc["Phase"]
    parameters = {f: parameters_with_unc[f][0] for f in parameters_with_unc}
    return parameters_with_unc, parameters


def _prepare_templates_and_phase_priors(
    profile,
    profile_weight,
    use_weight,
    nharm,
    get_outroot,
    files,
    weights,
    nbin,
    parameter_names,
    logprior_funcs,
    parameters,
):
    """Create templates, log diagnostics, and update phase priors."""
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

        for j, par in enumerate(parameter_names):
            if par == f"Phase_{i}":
                assert logprior_funcs[j] is None, (
                    f"Expected a placeholder prior for {par} from assign_logpriors; "
                    "found a real one. The Phase prior should be set exactly once, here."
                )
                logprior_funcs[j] = _flat_logprior(ph0 - 0.5, ph0 + 0.5)
                break

    return template_func, pulsed_frac, parameters, logprior_funcs


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
    list_parameter_names,
    likelihood_func,
    parameters_with_unc,
    observation_length,
    model,
):
    """Collect fit parameters, priors, factors, and initial fit values."""
    parameter_names = _collect_parameter_names(
        parameters,
        list_parameter_names,
        likelihood_func=likelihood_func,
    )
    logprior_funcs = assign_logpriors(
        parameter_names,
        parameters_with_unc,
        obs_length=observation_length,
    )
    factors = get_factors(
        parameter_names,
        model,
        observation_length,
        parvalunc=parameters_with_unc,
    )

    try:
        input_mean_fit_pars = [parameters[par] for par in parameter_names]
    except KeyError as exc:
        raise ValueError("One or more parameters are missing from the parameter file") from exc

    return parameter_names, logprior_funcs, factors, input_mean_fit_pars


def _trace_phase_0_likelihood(
    parameter_names,
    times_from_pepoch,
    parameters,
    input_mean_fit_pars,
    logprior_funcs,
    factors,
    template_func,
    likelihood_func,
    outroot,
    tolerance=1e-8,
):
    """Optionally trace the likelihood around Phase_0 for diagnostics."""
    for parameter in [p for p in parameter_names if p.startswith("Phase_")]:
        idx = parameter_names.index(parameter)
        results_trace = trace_likelihood_over_parameter(
            times_from_pepoch,
            parameters,
            parameter_names,
            input_mean_fit_pars,
            logprior_funcs,
            factors,
            template_func,
            parameter_name=parameter,
            parameter_values=np.linspace(
                parameters[parameter] - 6 * factors[idx],
                parameters[parameter] + 6 * factors[idx],
                100,
            ),
            likelihood_func=likelihood_func,
            tolerance=tolerance,
        )

        phase_values = list(results_trace.keys())
        ll_values = list(results_trace.values())
        best_phase = phase_values[np.nanargmax(ll_values)]

        with _plot_style_context():
            plt.figure("trace_" + parameter)
            plt.plot(phase_values, ll_values, color="black")
            plt.axvline(parameters[parameter], color="k", alpha=0.5, ls="--")
            plt.axvline(best_phase, color="r", ls="--")
            plt.xlabel(parameter)
            plt.ylabel("log likelihood")
            plt.savefig(outroot + f"_trace_{parameter}.jpg")

        ll_values_clean = [
            ll for ll in list(results_trace.values()) if not np.isnan(ll) and not np.isinf(ll)
        ]
        min_ll = np.nanmin(ll_values_clean)
        max_ll = np.nanmax(ll_values_clean)
        logging.info(f"Delta log likelihood for {parameter}: {max_ll - min_ll:.2f}")
        parameters[parameter] = phase_values[np.nanargmax(ll_values)]
    return parameters


def _get_par_dict(
    model,
    ignore_uncertainties=False,
    obs_length=1,
):  # The dictionary contains lists [parameter mean, parameter uncertainty]
    """Build a parameter/uncertainty dictionary from a PINT timing model.

    The returned mapping stores ``[value, uncertainty]`` for each parameter and
    fills missing uncertainties with heuristic defaults suitable for priors.
    """

    def return_unc(param):
        if param.uncertainty_value is None or param.uncertainty_value == 0:
            return np.nan
        return param.uncertainty_value.astype(float)

    parameters = {
        "Phase": [0, 0],
        "PB": [model.PB.value.astype(float) * 86400, return_unc(model.PB) * 86400],
        "TASC": [model.TASC.value.astype(float), return_unc(model.TASC)],
        "A1": [model.A1.value.astype(float), return_unc(model.A1)],
        "EPS1": [model.EPS1.value.astype(float), return_unc(model.EPS1)],
        "EPS2": [model.EPS2.value.astype(float), return_unc(model.EPS2)],
        "PBDOT": [model.PBDOT.value.astype(float), return_unc(model.PBDOT)],
        "PEPOCH": [
            model.PEPOCH.value.astype(float),
            return_unc(model.PEPOCH),
        ],  # I added Pepoch
    }

    count = 0
    while hasattr(model, f"F{count}"):
        parameters[f"F{count}"] = [
            getattr(model, f"F{count}").value.astype(float),
            return_unc(getattr(model, f"F{count}")),
        ]
        count += 1

    if ignore_uncertainties:
        # Start from a clean slate
        for par in parameters:
            parameters[par][1] = np.nan

    # Then, give sensible defaults for the uncertainties of some critical
    # parameters that are not set
    def check_uncertainty(par, default_uncertainty):
        if np.isnan(parameters[par][1]) or np.isinf(parameters[par][1]) or ignore_uncertainties:
            parameters[par][1] = default_uncertainty

    check_uncertainty("PB", parameters["PB"][0] / 2)

    Omega = 2 * np.pi / parameters["PB"][0]
    X = parameters["A1"][0]
    f = parameters["F0"][0]

    count = 0

    while hasattr(model, f"F{count}"):
        obs_length_change = 10 / obs_length ** (count + 1)
        max_orbital_change = X * Omega ** (count + 1) * f
        logging.debug(
            f"F{count}: max_orbital_change={max_orbital_change}, "
            f"obs_length_change={obs_length_change}"
        )
        default_unc = 10 * max_orbital_change + obs_length_change
        check_uncertainty(f"F{count}", default_unc)
        count += 1

    return parameters


def _load_and_format_events(
    event_file,
    energy_range,
    pepoch,
    plotlc=True,
    plotfile="lightcurve.jpg",
    return_energy=False,
    use_pi=False,
):
    """Load an event file, apply filtering, and express times from PEPOCH.

    Parameters
    ----------
    event_file : str
        Input event file readable by ``hendrics.io.load_events``.
    energy_range : tuple or None
        ``(emin, emax)`` range applied through ``filter_energy_range``.
    pepoch : float
        Reference epoch (MJD) used to compute ``times_from_pepoch``.
    plotlc : bool, optional
        If True, save a quick-look light curve.
    plotfile : str, optional
        Output filename for the light-curve plot.
    return_energy : bool, optional
        If True, also return event energies (or PI if ``use_pi=True``).
    use_pi : bool, optional
        Use PI channels instead of energy values.

    Returns
    -------
    tuple
        ``(times_from_pepoch, gtis_from_pepoch)`` or
        ``(times_from_pepoch, gtis_from_pepoch, energy)``.
    """
    events = load_events(event_file)
    events.apply_gtis(inplace=True)

    if plotlc:
        lc = events.to_lc(100)

        with _plot_style_context():
            fig = plt.figure("LC", figsize=(3.5, 2.65))
            lc.plot(ax=plt.gca())
            plt.savefig(plotfile)
            plt.close(fig)

    if energy_range is not None:
        events.filter_energy_range(energy_range, inplace=True)
    mjdref = events.mjdref
    pepoch_met = _mjd_to_sec(pepoch, mjdref)
    times_from_pepoch = (events.time - pepoch_met).astype(float)
    gtis_from_pepoch = (events.gti - pepoch_met).astype(float)
    if not use_pi:
        energy = events.energy
    else:
        energy = events.pi
    if return_energy:
        return times_from_pepoch, gtis_from_pepoch, energy
    return times_from_pepoch, gtis_from_pepoch


def trace_likelihood_over_parameter(
    times_from_pepoch,
    model_parameters,
    fit_parameters,
    values,
    logprior_funcs,
    factors,
    template_func,
    parameter_name,
    parameter_values,
    likelihood_func=pletsch_clarke_likelihood,
    tolerance=1e-8,
):
    """Trace the posterior (log-likelihood + log-prior) over one parameter.

    All fitted parameters are represented in a normalized local coordinate
    system used by :func:`optimize_solution`. This helper fixes every local
    parameter at zero except ``parameter_name``, which is scanned over
    ``parameter_values``.

    Returns
    -------
    dict
        Mapping from scanned local parameter value to posterior value.
    """

    _, _, func_to_maximize = _build_posterior_functions(
        times_from_pepoch=times_from_pepoch,
        model_parameters=model_parameters,
        fit_parameters=fit_parameters,
        values=values,
        logprior_funcs=logprior_funcs,
        factors=factors,
        template_func=template_func,
        likelihood_func=likelihood_func,
        tolerance=tolerance,
        weights=None,
        debug_local_phases=False,
        debug_func=False,
    )

    results = {}
    for val in parameter_values:
        idx = fit_parameters.index(parameter_name)
        pars = [0] * len(values)
        pars[idx] = val
        results[val] = func_to_maximize(pars)

    return results


def _build_posterior_functions(
    times_from_pepoch,
    model_parameters,
    fit_parameters,
    values,
    logprior_funcs,
    factors,
    template_func,
    likelihood_func,
    tolerance=1e-8,
    weights=None,
    debug_local_phases=False,
    debug_func=False,
):
    """Build shared posterior components used by trace and optimization flows."""

    def logprior(pars):
        if np.any(np.isnan(pars)):
            return -np.inf
        if np.any(np.isinf(pars)):
            return -np.inf

        logp = 0
        for parname, logp_func, initial, local_value, f in zip(
            fit_parameters, logprior_funcs, values, pars, factors
        ):
            value = local_value * f + initial
            logp += logp_func(value)
        return logp

    def local_phases(pars):
        allpars = copy.deepcopy(model_parameters)

        for par, initial, value, f in zip(fit_parameters, values, pars, factors):
            allpars[par] = value * f + initial
        if debug_local_phases:
            logging.debug(f"Local phases for parameters: {allpars}")

        return _calculate_phases(times_from_pepoch, allpars, tolerance=tolerance)

    def func_to_maximize(pars):
        lp = logprior(pars)
        if np.isinf(lp):
            return lp

        phases = local_phases(pars)

        ll = 0
        for i in range(len(phases)):
            ll += likelihood_func(
                phases[i], template_func[i], weights=weights[i] if weights is not None else None
            )

        if debug_func:
            logging.debug(f"pars: {pars}, func: {ll + lp}")

        return ll + lp

    return logprior, local_phases, func_to_maximize


def _plot_phaseogram_set(reference_phases, fitted_phases, times_from_pepoch, outroot, suffix=""):
    """Plot reference vs fitted phaseograms for each input file."""
    for i in range(len(times_from_pepoch)):
        _compare_phaseograms(
            reference_phases[i],
            phases_from_zero_to_one(fitted_phases[i]),
            times_from_pepoch[i],
            fname=outroot[i] + suffix + ".jpg",
        )


def _augment_results_with_fit_metadata(
    results,
    fit_parameters,
    fit_pars,
    values,
    factors,
    phase_source,
):
    """Store local-fit summaries and scaling metadata in sampler output."""
    rough_results = {}
    for par, value in zip(fit_parameters, fit_pars):
        rough_results["rough_d" + par] = value

    count = 0
    while f"Phase_{count}" in phase_source:
        results[f"additional_phase_{count}"] = phase_source[f"Phase_{count}"]
        count += 1

    results.update(rough_results)

    for par, initial, f in zip(fit_parameters, values, factors):
        results["d" + par + "_mean"] = results["d" + par + "_50"]
        results["d" + par + "_initial"] = initial
        results["d" + par + "_factor"] = f

    return results


def optimize_solution(
    times_from_pepoch,
    model_parameters,
    fit_parameters,
    values,
    logprior_funcs,
    factors,
    template_func,
    nsteps=1000,
    minimize_first=False,
    nharm=1,
    outroot="out",
    tolerance=1e-8,
    likelihood_func=pletsch_clarke_likelihood,
    weights=None,
):
    """Optimize and sample pulsar timing parameters for multiple event files.

    Workflow:
    1. Build a posterior from priors + profile likelihood.
    2. Optionally run deterministic minimization for a starting point.
    3. Run MCMC with :func:`safe_run_sampler`.
    4. Produce diagnostic phaseogram comparisons and return summary fields.

    Parameters are handled in local coordinates: for each fitted parameter,
    ``physical = local * factor + initial``.

    Returns
    -------
    dict
        Aggregated result dictionary containing posterior summaries, initial
        values, scaling factors, and copied model metadata.
    """
    _, local_phases, func_to_maximize = _build_posterior_functions(
        times_from_pepoch=times_from_pepoch,
        model_parameters=model_parameters,
        fit_parameters=fit_parameters,
        values=values,
        logprior_funcs=logprior_funcs,
        factors=factors,
        template_func=template_func,
        likelihood_func=likelihood_func,
        tolerance=tolerance,
        weights=weights,
        debug_local_phases=True,
        debug_func=True,
    )

    def func_to_minimize(pars):
        return -func_to_maximize(pars)

    logging.info("Initial parameters: ")
    for par in fit_parameters:
        logging.info(f"  {par}: {model_parameters[par]}")

    logging.info("Initial likelihood: " + str(func_to_maximize([0] * len(values))))
    all_zeros = [0] * len(values)
    if minimize_first:
        res = minimize(func_to_minimize, all_zeros)
        fit_pars = res.x
    else:
        fit_pars = all_zeros

    logging.info("Fitted (rescaled) parameters: " + str(fit_pars))

    pars_dict = copy.deepcopy(model_parameters)

    for par, initial, value, f in zip(fit_parameters, values, fit_pars, factors):
        pars_dict[par] = value * f + initial

    for key in fit_parameters:
        logging.info(
            f"  {key}: {pars_dict[key]} (difference from initial: {pars_dict[key] - model_parameters[key]})"
        )
    logging.info("Fitted likelihood: " + str(func_to_maximize(fit_pars)))
    phases = local_phases(fit_pars)
    phases_zero = local_phases(all_zeros)

    _plot_phaseogram_set(phases_zero, phases, times_from_pepoch, outroot, suffix="")

    corner_labels = [
        "d" + par + f"{np.log10(fac):+g}" for (par, fac) in zip(fit_parameters, factors)
    ]
    results = safe_run_sampler(
        func_to_maximize,
        fit_pars,
        max_n=nsteps,
        outroot=outroot[-1],
        labels=["d" + par for par in fit_parameters],
        corner_labels=corner_labels,
    )

    results.update(model_parameters)
    results = _augment_results_with_fit_metadata(
        results,
        fit_parameters,
        fit_pars,
        values,
        factors,
        phase_source=pars_dict,
    )

    fit_pars = [results["d" + par + "_50"] for par in fit_parameters]
    phases = local_phases(fit_pars)

    _plot_phaseogram_set(phases_zero, phases, times_from_pepoch, outroot, suffix="_final")

    return results


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

    list_parameter_names = sorted(fit_parameters)
    get_outroot = _make_outroot_getter(
        files,
        list_parameter_names,
        energy_range,
        nharm,
        likelihood_func,
        use_weight,
        general_outroot=general_outroot,
    )

    times_from_pepoch, observation_length, energies, expo = _load_events_for_all_files(
        files,
        energy_range,
        pepoch,
        get_outroot,
    )

    parameters_with_unc, parameters = _build_parameters_from_models(
        model,
        ref_model,
        observation_length,
        ignore_uncertainties=ignore_uncertainties,
    )

    parameter_names, logprior_funcs, factors, input_mean_fit_pars = _prepare_fit_setup(
        parameters,
        list_parameter_names,
        likelihood_func,
        parameters_with_unc,
        observation_length,
        model,
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

    template_func, pulsed_frac, parameters, logprior_funcs = _prepare_templates_and_phase_priors(
        profile,
        profile_weight,
        use_weight,
        nharm,
        get_outroot,
        files,
        weights,
        nbin,
        parameter_names,
        logprior_funcs,
        parameters,
    )

    outroots = _get_outroots(get_outroot, n_files)

    input_stuff = copy.deepcopy(
        (
            parameter_names,
            times_from_pepoch,
            parameters,
            input_mean_fit_pars,
            logprior_funcs,
            factors,
            template_func,
            likelihood_func,
        )
    )
    _trace_phase_0_likelihood(
        parameter_names,
        times_from_pepoch,
        parameters,
        input_mean_fit_pars,
        logprior_funcs,
        factors,
        template_func,
        likelihood_func,
        outroot=outroots[-1],
        tolerance=tolerance,
    )
    input_mean_fit_pars = [parameters[par] for par in parameter_names]
    output_stuff = (
        parameter_names,
        times_from_pepoch,
        parameters,
        input_mean_fit_pars,
        logprior_funcs,
        factors,
        template_func,
        likelihood_func,
    )

    results = optimize_solution(
        times_from_pepoch,
        parameters,
        parameter_names,
        input_mean_fit_pars,
        logprior_funcs,
        factors,
        template_func,
        nsteps=nsteps,
        minimize_first=minimize_first,
        nharm=nharm,
        outroot=outroots,
        tolerance=tolerance,
        likelihood_func=likelihood_func,
        weights=weights if use_weight else None,
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

    return _write_results_products(results, n_files, get_outroot, list_parameter_names, model)


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
        ignore_uncertainties=args.ignore_uncertainties,
    )
