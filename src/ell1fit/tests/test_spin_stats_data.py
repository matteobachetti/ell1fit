"""Tests for loading per-epoch spin measurements for ``ell1stat``."""

import numpy as np
import pytest
from astropy.table import Table

from ..orbital_decay_data import OrbitalModelCompatibilityError
from ..spin_stats_data import load_spin_table, read_extra_table, read_spin_epoch


def _write_result(path, pepoch=56000.0, f1_fitted=True, joint=False):
    """Write a minimal single-epoch ell1fit result file, as the pipeline would."""
    suffix = "_0" if joint else ""
    row = {
        f"PEPOCH{suffix}": [pepoch],
        f"F0{suffix}": [0.7],  # starting value, not the fit result
        f"F1{suffix}": [0.0],
        f"ctrate{suffix}": [2.0],
        f"pf{suffix}": [0.25],
        f"fname{suffix}": ["events.nc"],
    }
    pars = {"F0": (0.7, 1e-7), "F1": (0.0, 1e-12)} if f1_fitted else {"F0": (0.7, 1e-7)}
    for par, (initial, factor) in pars.items():
        name = f"d{par}{suffix}"
        row[f"{name}_initial"] = [initial]
        row[f"{name}_factor"] = [factor]
        row[f"{name}_16"] = [2.0]
        row[f"{name}_50"] = [3.0]
        row[f"{name}_84"] = [5.0]
        row[f"{name}_mean"] = [3.0]
    Table(row).write(path, format="ascii.ecsv", overwrite=True)
    return str(path)


def test_read_spin_epoch_uses_posterior_median(tmp_path):
    """F0/F1 are the posterior medians, not the bare F0/F1 columns, which hold the
    fit's starting values; the pulsed rate is pf * ctrate."""
    epoch = read_spin_epoch(_write_result(tmp_path / "a_results.ecsv"))
    assert np.isclose(epoch["f0"], 0.7 + 3e-7, rtol=0, atol=1e-15)
    assert np.allclose([epoch["f0_err_neg"], epoch["f0_err_pos"]], [1e-7, 2e-7])
    assert np.isclose(epoch["f1"], 3e-12)
    assert np.isclose(epoch["pulsed_rate"], 0.5)
    assert epoch["mjd"] == 56000.0


def test_unfitted_f1_is_nan(tmp_path):
    """A fixed F1 is an input, not a measurement, so it must not enter the statistics."""
    epoch = read_spin_epoch(_write_result(tmp_path / "a_results.ecsv", f1_fitted=False))
    assert np.isnan(epoch["f1"]) and np.isnan(epoch["f1_err_neg"])


def test_joint_fit_is_rejected(tmp_path):
    """A joint multi-epoch fit's output has no single PEPOCH, and is refused clearly."""
    with pytest.raises(OrbitalModelCompatibilityError, match="joint"):
        read_spin_epoch(_write_result(tmp_path / "a_results.ecsv", joint=True))


def test_extra_table_merges_and_sorts(tmp_path):
    """Extra points (e.g. literature F0 values) join the ell1fit ones in MJD order, with
    their optional columns NaN when absent and symmetric errors on both sides."""
    extra = tmp_path / "extra.csv"
    Table({"MJD": [55000.0], "F0": [0.71], "F0_err": [1e-6]}).write(extra, format="ascii.csv")
    files = [
        _write_result(tmp_path / f"{i}_results.ecsv", pepoch=p)
        for i, p in enumerate([57000.0, 56000.0])
    ]
    table = load_spin_table(files, extra_files=[str(extra)])
    assert list(table["mjd"]) == [55000.0, 56000.0, 57000.0]
    assert table["f0_err_neg"][0] == table["f0_err_pos"][0] == 1e-6
    assert np.isnan(table["f1"][0]) and np.isnan(table["pulsed_rate"][0])
    assert table["label"][0] == "extra"


def test_extra_table_needs_f0(tmp_path):
    """An extra table without the required columns fails with the column names."""
    extra = tmp_path / "extra.csv"
    Table({"MJD": [55000.0], "F1": [1e-11]}).write(extra, format="ascii.csv")
    with pytest.raises(ValueError, match="F0"):
        read_extra_table(str(extra))
