"""Energy-dependent event weighting for ell1fit.

Pulsed fraction usually varies with photon energy, so events in energy bands
where the pulse is strong carry more timing information than events where it is
weak. Weighting by that trend recovers signal that an unweighted fit throws
away.

The weights returned here feed the weighted branch of
:func:`ell1fit.likelihoods.pletsch_clarke_likelihood`, which requires them to
lie in ``[0, 1]``: a weight of 1 means "trust this event's phase fully", 0 means
"treat it as unmodulated background". The normalization at the end of
:func:`pf_weight_versus_energy` enforces that by scaling the peak amplitude to
1, and falls back to uniform weights if the amplitude trend is degenerate.
"""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from stingray.pulse.pulsar import z_n_binned_events
from stingray.stats import (
    a_from_ssig,
    z2_n_detection_level,
    power_confidence_limits,
)

from .phase_utils import _calculate_phases
from .plotting import plot_style_context as _plot_style_context


__all__ = [
    "pf_weight_versus_energy",
]


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
