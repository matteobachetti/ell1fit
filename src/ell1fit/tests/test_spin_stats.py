"""Tests for ``ell1stat``'s secular spin trend and local spin-derivative statistics."""

import json

import numpy as np
from astropy.table import Table

from ..spin_stats import fit_polynomial_with_scatter, main


def test_scatter_fit_recovers_line_and_scatter():
    """With intrinsic scatter far above the formal errors, the slope is recovered
    within its (scatter-inflated) error and the fitted scatter matches the injected one."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-500, 500, 300))
    y = 2 + 0.5 * x + rng.normal(0, 1.0, x.size)
    fit = fit_polynomial_with_scatter(x, y, np.full(x.size, 0.01), degree=1)
    assert abs(fit.coeffs[1] - 0.5) < 3 * fit.errors[1]
    assert abs(fit.scatter - 1.0) < 0.15
    assert np.isclose(fit.chi2 / fit.dof, 1.0)


def test_scatter_fit_is_weighted_least_squares_when_consistent():
    """Data that scatter less than their errors get no extra scatter, and a degree-0
    fit is then exactly the inverse-variance weighted mean."""
    y = np.array([1.0, 1.1, 0.9, 1.05])
    err = np.array([0.1, 0.2, 0.1, 0.3])
    fit = fit_polynomial_with_scatter(np.arange(4.0), y, err, degree=0)
    w = 1 / err**2
    assert fit.scatter == 0
    assert np.isclose(fit.coeffs[0], np.sum(w * y) / w.sum())
    assert np.isclose(fit.errors[0], w.sum() ** -0.5)


def test_cli_secular_and_local_spin(tmp_path):
    """End to end on an extra table: a spin-down secular trend with local spin-up
    episodes is reported as such, and the summary, epoch table and figures are written."""
    rng = np.random.default_rng(1)
    mjd = np.sort(rng.uniform(55000, 60000, 12))
    f1_secular = -4e-11
    f0 = 0.7 + f1_secular * (mjd - 57500) * 86400 + rng.normal(0, 1e-4, mjd.size)
    f1 = rng.normal(3e-11, 3e-11, mjd.size)
    extra = tmp_path / "extra.csv"
    Table(
        {
            "MJD": mjd,
            "F0": f0,
            "F0_err": np.full(mjd.size, 1e-7),
            "F1": f1,
            "F1_err": np.full(mjd.size, 5e-12),
        }
    ).write(extra, format="ascii.csv")

    outroot = str(tmp_path / "out")
    main(["--extra", str(extra), "-o", outroot, "--figure-format", "png"])

    summary = json.load(open(outroot + "_results.json"))
    secular, local = summary["secular"], summary["local_f1"]
    assert abs(secular["F1"][0] - f1_secular) < 3 * secular["F1"][1]
    assert np.isclose(local["mean"][0], np.mean(f1), rtol=0.05)
    assert local["fraction_spin_up"] == np.mean(f1 > 0)
    assert summary["secular_minus_local_sigma"] < -3
    for suffix in ("_epochs.ecsv", "_trend.png", "_f1.png"):
        assert (tmp_path / f"out{suffix}").exists()
