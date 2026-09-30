"""Tests for ``ell1fold``: folding, the phase columns it saves, and event weights."""

import os
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from hendrics.io import load_events

from ..events import _load_events_for_all_files
from ..fold import PHASE_COLUMNS, fold_events, main, save_phase_columns
from ..plotting import figure_path
from ..profile_plotting import _plot_phaseogram

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


@pytest.fixture
def nc_event_file(tmp_path):
    for ext in ("nc", "par"):
        shutil.copy(os.path.join(datadir, f"events0.{ext}"), tmp_path)
    return str(tmp_path / "events0.nc"), str(tmp_path / "events0.par")


def _write_ogip_copy(nc_file, outfile):
    """Write the events of ``nc_file`` as a minimal NuSTAR-like OGIP event file.

    A ``WGT`` column equal to ``TIME`` makes any misalignment between weights
    and events visible, and an extra header card checks the copy keeps headers.
    The checked-in events all have ``PI = 0``, so ``PI`` is redrawn over the
    NuSTAR band to give an energy selection something to remove.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events = load_events(nc_file)
    mjdrefi = np.floor(events.mjdref)
    columns = [
        fits.Column("TIME", "D", array=events.time),
        fits.Column("PI", "J", array=np.random.default_rng(0).integers(35, 1910, events.time.size)),
        fits.Column("WGT", "D", array=events.time),
    ]
    hdu = fits.BinTableHDU.from_columns(columns, name="EVENTS")
    header = dict(
        MJDREFI=mjdrefi,
        MJDREFF=events.mjdref - mjdrefi,
        TIMESYS="TDB",
        TIMEREF="SOLARSYSTEM",
        TELESCOP="NuSTAR",
        INSTRUME="FPMA",
        TIMEZERO=0.0,
        TSTART=events.gti[0, 0],
        TSTOP=events.gti[-1, 1],
        ELL1TEST="kept",
    )
    for key, value in header.items():
        hdu.header[key] = value
    gti = fits.BinTableHDU.from_columns(
        [
            fits.Column("START", "D", array=events.gti[:, 0]),
            fits.Column("STOP", "D", array=events.gti[:, 1]),
        ],
        name="GTI",
    )
    fits.HDUList([fits.PrimaryHDU(), hdu, gti]).writeto(outfile)
    return events.time.size


def test_ell1fold_cli_writes_phaseogram_and_phase_file(nc_event_file):
    """The CLI's two products appear, and the saved phase columns are consistent."""
    event_file, parfile = nc_event_file
    main([event_file, "-p", parfile, "--save-phases"])

    root = event_file.replace(".nc", "")
    assert os.path.exists(figure_path(root + "_phaseogram"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original = load_events(event_file)
        phased = load_events(root + "_phase.nc")
    assert phased.time.size == original.time.size
    assert np.allclose(phased.PULSE_PHASE, phased.ABS_PHASE - np.floor(phased.ABS_PHASE))
    assert np.all((phased.ORBIT_PHASE >= 0) & (phased.ORBIT_PHASE < 1))


def test_ogip_copy_keeps_the_file_and_matches_the_nc_phases(nc_event_file, tmp_path):
    """A FITS input is copied intact, and phased event by event as its .nc twin."""
    event_file, parfile = nc_event_file
    fits_file = str(tmp_path / "copy.evt")
    nrows = _write_ogip_copy(event_file, fits_file)

    (nc_out,) = save_phase_columns([event_file], [parfile])
    (fits_out,) = save_phase_columns([fits_file], [parfile])
    assert fits_out == str(tmp_path / "copy_phase.evt")

    with fits.open(fits_out) as hdul:
        assert "GTI" in hdul
        assert hdul["EVENTS"].header["ELL1TEST"] == "kept"
        assert len(hdul["EVENTS"].data) == nrows
        saved = {col: hdul["EVENTS"].data[col] for col in PHASE_COLUMNS + ("WGT",)}
    reference = load_events(nc_out)
    for col in PHASE_COLUMNS:
        assert np.allclose(saved[col], getattr(reference, col), atol=1e-6)


def test_weight_column_follows_the_event_selection(nc_event_file, tmp_path):
    """Weights are filtered in step with the times by the GTI and energy cuts."""
    event_file, parfile = nc_event_file
    fits_file = str(tmp_path / "copy.evt")
    nrows = _write_ogip_copy(event_file, fits_file)

    times, _, _, _, weights = _load_events_for_all_files(
        [fits_file], [5, 20], [57000.0], lambda i: "x", plotlc=False, extra_column="WGT"
    )
    offset = weights[0] - times[0]
    assert 0 < times[0].size < nrows
    assert np.allclose(offset, offset[0])

    fold_events([fits_file], [parfile], energy_range=[5, 20], weight_column="WGT")


def test_missing_weight_column_is_reported(nc_event_file):
    """Asking for a column the file does not have fails with a clear message."""
    event_file, parfile = nc_event_file
    with pytest.raises(ValueError, match="No per-event column 'NOPE'"):
        fold_events([event_file], [parfile], weight_column="NOPE")


def test_unit_weights_leave_the_profile_unchanged():
    """Weighting is a pure extension: all-ones weights reproduce the plain profile."""
    rng = np.random.default_rng(0)
    phases, times = rng.uniform(0, 1, 500), rng.uniform(0, 1e4, 500)
    profiles = []
    for weights in (None, np.ones(500)):
        fig, (ax0, ax1) = plt.subplots(2)
        _plot_phaseogram(phases, times, ax0, ax1, weights=weights)
        profiles.append(ax0.lines[0].get_ydata())
        plt.close(fig)
    assert np.array_equal(*profiles)
