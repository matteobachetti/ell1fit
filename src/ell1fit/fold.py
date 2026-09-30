"""``ell1fold``: fold event files with an ELL1 solution, without fitting.

The first thing :func:`ell1fit.pipeline.ell1fit` does is load the parfiles and
events and fold them. This command stops there: it plots the folded profile and
phaseogram for each file and, optionally, writes each event's phases back to a
copy of the event file.

The phase columns
-----------------
``--save-phases`` writes ``<name>_phase.<ext>`` next to each input, in the same
format, with three columns added:

``PULSE_PHASE``
    Pulse phase in [0, 1), the same number ell1fit folds with.
``ABS_PHASE``
    The same phase unwrapped: cycles since the parfile's ``PEPOCH``. Its zero
    is the phase the spin model gives at ``PEPOCH`` after deorbiting, not the
    ``TZRMJD`` reference PINT would use.
``ORBIT_PHASE``
    ``(t - TASC) / PB`` in [0, 1), in observed time: 0 at the ascending node.

Every event in the file gets a phase: the energy selection and the good time
intervals apply to the plots only, so the new columns line up with the rows.

OGIP FITS files are copied with the columns appended to their ``EVENTS``
extension, keeping every header and extension. Other formats go through
``hendrics.io.save_events``; HDF5 and ECSV store the names in lower case.
"""

import argparse
import logging

import numpy as np
from astropy.io import fits
from hendrics.io import get_file_format, save_events

from . import splitext_improved
from .events import _load_events_for_all_files, _read_event_list, _times_from_pepoch
from .logging import configure_logging
from .models import _build_parameters_from_models, _load_and_validate_models
from .phase_utils import _calculate_absolute_phases, _calculate_phases
from .plotting import add_figure_format_argument, set_figure_format
from .profile_plotting import plot_phaseogram
from .results_io import _format_energy_string


__all__ = [
    "PHASE_COLUMNS",
    "fold_events",
    "main",
    "phase_file_name",
    "save_phase_columns",
]


#: The columns ``--save-phases`` adds, in the order they are written.
PHASE_COLUMNS = ("PULSE_PHASE", "ABS_PHASE", "ORBIT_PHASE")


def phase_file_name(fname):
    """``dir/name.evt.gz`` -> ``dir/name_phase.evt.gz``."""
    root, ext = splitext_improved(fname)
    return root + "_phase" + ext


def fold_events(files, parfiles, energy_range=None, weight_column=None, nbin=32, tolerance=1e-8):
    """Fold each event file with its parfile and plot profile and phaseogram.

    ``nbin`` is the number of phase bins per pulse cycle.

    Returns
    -------
    list of str
        The figures written, one per file.
    """
    if len(files) != len(parfiles):
        raise ValueError("The number of parameter files must match that of event files.")

    models, pepoch, ref_model = _load_and_validate_models(parfiles)
    roots = [splitext_improved(f)[0] for f in files]
    loaded = _load_events_for_all_files(
        files,
        energy_range,
        pepoch,
        lambda i: roots[i],
        plotlc=False,
        extra_column=weight_column,
    )
    times, observation_length = loaded[0], loaded[1]
    weights = loaded[4] if weight_column is not None else [None] * len(files)

    _, parameters = _build_parameters_from_models(models, ref_model, observation_length)
    phases = _calculate_phases(times, parameters, tolerance=tolerance)

    suffix = _format_energy_string(energy_range)
    if weight_column is not None:
        suffix += f"_{weight_column}"
    figures = []
    for i, root in enumerate(roots):
        fname = f"{root}{suffix}_phaseogram"
        logging.info(f"Writing phaseogram of {files[i]} to {fname}")
        figures.append(
            plot_phaseogram(
                phases[i], times[i], fname, weights=weights[i], nbin=nbin, pepoch=pepoch[i]
            )
        )
    return figures


def _append_fits_columns(fname, outfile, columns):
    """Copy an OGIP event file, appending ``columns`` to its EVENTS extension."""
    with fits.open(fname) as hdul:
        name = "EVENTS" if "EVENTS" in hdul else 1
        events = hdul[name]
        nrows = events.data.shape[0]
        for key, value in columns.items():
            if value.size != nrows:
                raise ValueError(
                    f"{fname}: {value.size} phases for {nrows} rows; the event reader "
                    f"dropped or added events, so {key} cannot be aligned with the file."
                )
        kept = [c for c in events.columns if c.name.upper() not in columns]
        new = [fits.Column(name=key, format="D", array=value) for key, value in columns.items()]
        hdul[name] = fits.BinTableHDU.from_columns(kept + new, header=events.header, name="EVENTS")
        hdul.writeto(outfile, overwrite=True)


def save_phase_columns(files, parfiles, tolerance=1e-8):
    """Write each event file back with ``PULSE_PHASE``, ``ABS_PHASE``, ``ORBIT_PHASE``.

    Every event is phased, with no energy or GTI selection. See the module
    docstring for what each column means.

    Returns
    -------
    list of str
        The files written.
    """
    event_lists = [_read_event_list(f) for f in files]
    models, pepoch, ref_model = _load_and_validate_models(parfiles)
    times = [_times_from_pepoch(ev, p)[0] for ev, p in zip(event_lists, pepoch)]
    observation_length = np.array([t[-1] - t[0] for t in times])
    _, parameters = _build_parameters_from_models(models, ref_model, observation_length)

    pulse, orbit = _calculate_absolute_phases(times, parameters, tolerance=tolerance)

    outfiles = []
    for i, (fname, events) in enumerate(zip(files, event_lists)):
        columns = dict(zip(PHASE_COLUMNS, (pulse[i] - np.floor(pulse[i]), pulse[i], orbit[i])))
        outfile = phase_file_name(fname)
        logging.info(f"Writing phases of {fname} to {outfile}")
        if get_file_format(fname) == "ogip":
            _append_fits_columns(fname, outfile, columns)
        else:
            for key, value in columns.items():
                setattr(events, key, value)
            save_events(events, outfile)
        outfiles.append(outfile)
    return outfiles


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fold event files with an ELL1 timing solution and plot the pulse "
            "profile and phaseogram, without fitting anything."
        )
    )
    parser.add_argument("files", help="Input event files (any format HENDRICS reads)", nargs="+")
    parser.add_argument(
        "-p", "--parfiles", nargs="+", required=True, help="ELL1 parfiles, one per event file"
    )
    parser.add_argument(
        "-e", "--energy-range", type=float, nargs=2, default=None, help="Energy range (keV)"
    )
    parser.add_argument(
        "-b", "--nbin", type=int, default=32, help="Phase bins per pulse cycle (default 32)"
    )
    parser.add_argument(
        "--weight-column",
        default=None,
        help="Per-event column used to weight the profile and phaseogram",
    )
    parser.add_argument(
        "--save-phases",
        action="store_true",
        default=False,
        help=(
            "Also write <name>_phase.<ext>, a copy of each event file with "
            "PULSE_PHASE, ABS_PHASE and ORBIT_PHASE columns"
        ),
    )
    add_figure_format_argument(parser)
    return parser


def main(args=None):
    """Entry point for ``ell1fold``."""
    configure_logging()
    args = _parser().parse_args(args)
    set_figure_format(args.figure_format)

    fold_events(
        args.files,
        args.parfiles,
        energy_range=args.energy_range,
        weight_column=args.weight_column,
        nbin=args.nbin,
    )
    if args.save_phases:
        save_phase_columns(args.files, args.parfiles)


if __name__ == "__main__":
    main()
