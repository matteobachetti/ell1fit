"""Event-file loading and preparation for ell1fit.

Turns event files on disk into the per-file arrays the rest of the pipeline
works with: arrival times expressed in seconds from each file's own ``PEPOCH``,
the good-time intervals, event energies, and the exposure implied by the GTIs.

Times are referenced to each file's own ``PEPOCH`` rather than to a common
epoch. That keeps their magnitude small, which matters: the spin phase is
``t * F0``, so a long lever arm from a distant reference epoch would eat into
the float64 significand that the deorbiting tolerance depends on.
"""

import logging  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from hendrics.io import load_events

from .phase_utils import _mjd_to_sec
from .plotting import plot_style_context as _plot_style_context


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
        ``(emin, emax)`` range applied through ``filter_energy_range``. This is
        always interpreted in calibrated energy (keV), regardless of ``use_pi``.
    pepoch : float
        Reference epoch (MJD) used to compute ``times_from_pepoch``.
    plotlc : bool, optional
        If True, save a quick-look light curve.
    plotfile : str, optional
        Output filename for the light-curve plot.
    return_energy : bool, optional
        If True, also return event energies (or PI if ``use_pi=True``).
    use_pi : bool, optional
        Return PI channels instead of calibrated energy values (only affects
        what is returned for weighting; the ``energy_range`` cut above is
        still applied in calibrated energy).

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
    energy = events.pi if use_pi else events.energy
    if return_energy:
        return times_from_pepoch, gtis_from_pepoch, energy
    return times_from_pepoch, gtis_from_pepoch


def _load_events_for_all_files(files, energy_range, pepoch, get_outroot, use_pi=False):
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
            use_pi=use_pi,
        )
        expo[i] += np.sum(np.diff(gtis, axis=1))
        observation_length[i] = times_from_pepoch[i][-1] - times_from_pepoch[i][0]

    return times_from_pepoch, observation_length, energies, expo
