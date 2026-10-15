"""Event-file loading and preparation for ell1fit.

Turns event files on disk into the per-file arrays the rest of the pipeline
works with: arrival times expressed in seconds from each file's own ``PEPOCH``,
the good-time intervals, event energies, and the exposure implied by the GTIs.

Times are referenced to each file's own ``PEPOCH`` rather than to a common
epoch. That keeps their magnitude small, which matters: the spin phase is
``t * F0``, so a long lever arm from a distant reference epoch would eat into
the float64 significand that the deorbiting tolerance depends on.
"""

import matplotlib.pyplot as plt
import numpy as np
from hendrics.io import get_file_format, load_events
from stingray import EventList

from .phase_utils import _mjd_to_sec
from .plotting import figure_size as _figure_size
from .plotting import plot_style_context as _plot_style_context
from .plotting import save_figure as _save_figure


__all__ = [
    "_event_column",
    "_load_events_for_all_files",
    "_read_event_list",
    "_times_from_pepoch",
]


def _read_event_list(event_file, additional_columns=None):
    """Read an event file with HENDRICS, optionally keeping extra columns.

    ``hendrics.io.load_events`` keeps everything stored in its own formats, but
    for OGIP FITS files it only reads the standard columns. Any
    ``additional_columns`` are then requested from stingray's OGIP reader
    explicitly; they arrive as lower-case attributes.
    """
    if additional_columns and get_file_format(event_file) == "ogip":
        return EventList.read(event_file, fmt="ogip", additional_columns=list(additional_columns))
    return load_events(event_file)


def _event_column(events, name):
    """Return the per-event array ``name`` of an event list, in either case.

    Raises
    ------
    ValueError
        If there is no per-event attribute by that name.
    """
    for attr in (name, name.lower()):
        value = getattr(events, attr, None)
        if value is not None and np.size(value) == np.size(events.time):
            return np.asarray(value)
    raise ValueError(
        f"No per-event column {name!r} in this event file. "
        f"Available: {', '.join(sorted(events.array_attrs()))}."
    )


def _times_from_pepoch(events, pepoch):
    """Event times in seconds from ``pepoch`` (MJD), plus the GTIs on the same scale."""
    pepoch_met = _mjd_to_sec(pepoch, events.mjdref)
    return (events.time - pepoch_met).astype(float), (events.gti - pepoch_met).astype(float)


def _load_and_format_events(
    event_file,
    energy_range,
    pepoch,
    plotlc=True,
    plotfile="lightcurve",
    return_energy=False,
    use_pi=False,
    extra_column=None,
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
    extra_column : str or None, optional
        Name of a further per-event column to return, filtered in step with
        the times. Appended to the returned tuple when given.

    Returns
    -------
    tuple
        ``(times_from_pepoch, gtis_from_pepoch)`` or
        ``(times_from_pepoch, gtis_from_pepoch, energy)``, followed by the
        ``extra_column`` values if one was requested.
    """
    events = _read_event_list(event_file, [extra_column] if extra_column else None)
    if extra_column is not None:
        # Fail before any work is done, and store it under the name the
        # filters below will carry along.
        events.fold_extra_column = _event_column(events, extra_column)
    events.apply_gtis(inplace=True)

    if plotlc:
        lc = events.to_lc(100)

        with _plot_style_context():
            fig = plt.figure("LC", figsize=_figure_size("column"), layout="constrained")
            lc.plot(ax=plt.gca())
            _save_figure(fig, plotfile)

    if energy_range is not None:
        events.filter_energy_range(energy_range, inplace=True)
    times_from_pepoch, gtis_from_pepoch = _times_from_pepoch(events, pepoch)
    energy = events.pi if use_pi else events.energy
    out = (times_from_pepoch, gtis_from_pepoch)
    if return_energy:
        out += (energy,)
    if extra_column is not None:
        out += (np.asarray(events.fold_extra_column, dtype=float),)
    return out


def _load_events_for_all_files(
    files, energy_range, pepoch, get_outroot, use_pi=False, plotlc=True, extra_column=None
):
    """Load all event files and compute per-file exposure and duration.

    ``plotlc=False`` skips the quick-look light curve of each file. With an
    ``extra_column``, a fifth element is returned: that column for each file,
    filtered in step with the times.
    """
    n_files = len(files)
    times_from_pepoch = [[] for _ in range(n_files)]
    observation_length = np.zeros(n_files, dtype=float)
    energies = [[] for _ in range(n_files)]
    extras = [[] for _ in range(n_files)]
    expo = np.zeros(n_files)

    for i in range(n_files):
        fname = files[i]
        loaded = _load_and_format_events(
            fname,
            energy_range,
            pepoch[i],
            plotlc=plotlc,
            plotfile=get_outroot(i) + f"_lightcurve_{i}",
            return_energy=True,
            use_pi=use_pi,
            extra_column=extra_column,
        )
        times_from_pepoch[i], gtis, energies[i] = loaded[:3]
        if extra_column is not None:
            extras[i] = loaded[3]
        expo[i] += np.sum(np.diff(gtis, axis=1))
        observation_length[i] = times_from_pepoch[i][-1] - times_from_pepoch[i][0]

    if extra_column is not None:
        return times_from_pepoch, observation_length, energies, expo, extras
    return times_from_pepoch, observation_length, energies, expo
