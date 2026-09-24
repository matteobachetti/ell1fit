"""Load per-epoch spin measurements -- ``F0``, ``F1`` and pulsed count rate --
for ``ell1stat``.

The inputs are the same single-epoch ``ell1fit`` ecsv results ``ell1decay``
reads, plus optional "extra" tables for epochs with no full ``ell1fit`` run
(literature values, quick-look timing). Unlike ``ell1decay``, no orbital
parameter is required: a file only needs a fitted ``F0``.

Which value is "the" F0
-----------------------
The bare ``F0``/``F1`` columns of an ``ell1fit`` result are the fit's
*starting* values, copied from the input parfile (they equal ``dF0_initial``).
The fitted value is the posterior median, ``dF0_50 * dF0_factor +
dF0_initial``, which is also what ``ell1fit`` writes into its output ``.par``.
This module always uses the latter, through the same
:func:`~ell1fit.orbital_decay_data.retrieve_value_and_error` ``ell1decay`` uses.

Pulsed count rate
-----------------
``pf`` is ``(max - min) / (max + min)`` of the pulse template and ``ctrate`` is
all events over exposure, so ``pf * ctrate`` is the pulse semi-amplitude
``(max - min) / 2`` in counts/s. A constant background raises ``max`` and
``min`` equally and cancels, which makes this a background-free proxy for the
pulsed luminosity -- within one instrument: count rates from different
instruments are not comparable. Neither column has an uncertainty.
"""

import os

import numpy as np
from astropy.table import Table

from .orbital_decay_data import (
    OrbitalModelCompatibilityError,
    _float128_to_float64_header,
    retrieve_value_and_error,
)

__all__ = ["SPIN_COLUMNS", "load_spin_table", "read_extra_table", "read_spin_epoch"]

#: Columns of the table :func:`load_spin_table` returns. Missing measurements are NaN.
SPIN_COLUMNS = (
    "label",
    "file",
    "mjd",
    "f0",
    "f0_err_neg",
    "f0_err_pos",
    "f1",
    "f1_err_neg",
    "f1_err_pos",
    "pulsed_rate",
)

#: Label given to every point read from an ell1fit result file.
ELL1FIT_LABEL = "ell1fit"


def _empty_epoch():
    return {col: np.nan for col in SPIN_COLUMNS}


def read_spin_epoch(fname):
    """Read one ``ell1fit`` result file's last row into a dict keyed by :data:`SPIN_COLUMNS`.

    Raises
    ------
    OrbitalModelCompatibilityError
        If ``fname`` is a joint multi-epoch fit's output, or has no fitted ``F0``.
    """
    with open(fname) as fobj:
        table = Table.read(_float128_to_float64_header(fobj.read()), format="ascii.ecsv")
    if "PEPOCH" not in table.colnames:
        if "PEPOCH_0" in table.colnames:
            raise OrbitalModelCompatibilityError(
                f"{fname} is a joint multi-epoch fit's output (has PEPOCH_0, not PEPOCH) -- "
                "ell1stat needs one single-epoch result file per epoch."
            )
        raise OrbitalModelCompatibilityError(f"{fname} has no PEPOCH column")
    row = table[-1]

    epoch = _empty_epoch()
    epoch.update(label=ELL1FIT_LABEL, file=os.path.basename(fname), mjd=float(row["PEPOCH"]))

    for par in ("F0", "F1"):
        value, err = retrieve_value_and_error(row, par)
        if value is None:
            if par == "F0":
                raise OrbitalModelCompatibilityError(f"{fname} has no fitted F0")
            continue
        key = par.lower()
        epoch[key] = float(value)
        epoch[f"{key}_err_neg"], epoch[f"{key}_err_pos"] = float(err[0]), float(err[1])

    if "pf" in row.colnames and "ctrate" in row.colnames:
        epoch["pulsed_rate"] = float(row["pf"]) * float(row["ctrate"])
    return epoch


def read_extra_table(fname):
    """Read a table of extra epochs, in any format :meth:`astropy.table.Table.read` guesses.

    Required columns are ``MJD``, ``F0`` and ``F0_err``; ``F1``/``F1_err``,
    ``pulsed_rate`` (counts/s) and ``label`` are optional. Errors are 1-sigma
    and symmetric. ``label`` groups points in the plots (e.g. by instrument or
    paper); it defaults to ``"extra"``.

    Returns
    -------
    list of dict
        One dict per row, keyed by :data:`SPIN_COLUMNS`.
    """
    table = Table.read(fname)
    missing = [c for c in ("MJD", "F0", "F0_err") if c not in table.colnames]
    if missing:
        raise ValueError(f"{fname} is missing required column(s): {missing}")

    epochs = []
    for row in table:
        epoch = _empty_epoch()
        epoch.update(
            label=str(row["label"]) if "label" in table.colnames else "extra",
            file=os.path.basename(fname),
            mjd=float(row["MJD"]),
            f0=float(row["F0"]),
            f0_err_neg=float(row["F0_err"]),
            f0_err_pos=float(row["F0_err"]),
        )
        if "F1" in table.colnames and "F1_err" in table.colnames:
            epoch.update(
                f1=float(row["F1"]),
                f1_err_neg=float(row["F1_err"]),
                f1_err_pos=float(row["F1_err"]),
            )
        if "pulsed_rate" in table.colnames:
            epoch["pulsed_rate"] = float(row["pulsed_rate"])
        epochs.append(epoch)
    return epochs


def load_spin_table(files, extra_files=()):
    """All epochs from ``ell1fit`` result files and extra tables, sorted by MJD.

    Returns
    -------
    astropy.table.Table
        Columns :data:`SPIN_COLUMNS`; missing measurements are NaN.
    """
    epochs = [read_spin_epoch(f) for f in files]
    for fname in extra_files:
        epochs.extend(read_extra_table(fname))
    if not epochs:
        raise ValueError("No input epochs")
    table = Table(rows=[[e[c] for c in SPIN_COLUMNS] for e in epochs], names=SPIN_COLUMNS)
    table.sort("mjd")
    return table
