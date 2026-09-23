"""Tests for ``ell1stat``'s secular spin trend and local spin-derivative statistics."""

import json

import numpy as np
from astropy.table import Table

from ..spin_periodicity import (
    floating_mean_periodogram,
    rate_values,
    search_joint_periodicity,
    search_periodicity,
)
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
    episodes is reported as such, the periodicity search runs on F1 and the pulsed
    rate, and the summary, epoch table and figures are written."""
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
            "pulsed_rate": rng.uniform(0.1, 0.2, mjd.size),
        }
    ).write(extra, format="ascii.csv")

    outroot = str(tmp_path / "out")
    main(
        ["--extra", str(extra), "-o", outroot, "--figure-format", "png"]
        + ["--min-period", "50", "--max-period", "70", "--n-shuffle", "50"]
    )

    summary = json.load(open(outroot + "_results.json"))
    secular, local = summary["secular"], summary["local_f1"]
    assert abs(secular["F1"][0] - f1_secular) < 3 * secular["F1"][1]
    assert np.isclose(local["mean"][0], np.mean(f1), rtol=0.05)
    assert local["fraction_spin_up"] == np.mean(f1 > 0)
    assert summary["secular_minus_local_sigma"] < -3
    assert set(summary["periodicity"]) == {"f1", "pulsed_rate", "joint"}
    assert 50 <= summary["periodicity"]["f1"]["best_period_days"] <= 70
    for suffix in (
        "_epochs.ecsv",
        "_trend.png",
        "_f1.png",
        "_periodogram.png",
        "_folded.png",
        "_folded_joint.png",
    ):
        assert (tmp_path / f"out{suffix}").exists()


def test_periodogram_matches_astropy():
    """The vectorised floating-mean periodogram is the astropy LombScargle "standard"
    power, so the shuffling and amplitude machinery sits on a checked foundation."""
    from astropy.timeseries import LombScargle

    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 3000, 20))
    y = rng.normal(0, 1, t.size)
    dy = rng.uniform(0.5, 2, t.size)
    freq = np.linspace(1 / 70, 1 / 50, 50)
    power, _ = floating_mean_periodogram(t, y, dy**2, freq)
    expected = LombScargle(t, y, dy, fit_mean=True, center_data=True).power(freq, method="slow")
    assert np.allclose(power, expected)


def _sparse_times(rng, n=30):
    """Clustered, gappy sampling over ~20 years, like a real multi-campaign dataset."""
    starts = rng.uniform(0, 7000, 6)
    return np.sort(np.concatenate([s + rng.uniform(0, 40, n // 6) for s in starts]))


def test_search_finds_injected_period_and_not_noise():
    """A sinusoid well above the noise is found at its period with a small false-alarm
    probability; pure noise on the same sampling is not flagged."""
    rng = np.random.default_rng(3)
    t = _sparse_times(rng)
    noise = rng.normal(0, 1, t.size)
    signal = search_periodicity(
        t, noise + 3 * np.sin(2 * np.pi * t / 61.0), None, 50, 70, n_shuffle=200, rng=rng
    )
    assert abs(signal["best_period_days"] - 61.0) < 0.5
    assert signal["fap"] < 0.01
    null = search_periodicity(t, noise, None, 50, 70, n_shuffle=200, rng=rng)
    assert null["fap"] > 0.05


def test_rate_exclude_drops_matching_files_only_from_pulsed_rate():
    """Excluded files (e.g. another instrument, whose count rates are not comparable)
    lose their pulsed rate, but keep F0 and F1 for the other analyses."""
    table = Table(
        {
            "label": ["ell1fit"] * 3,
            "file": ["nu1_results.ecsv", "chandra_results.ecsv", "nu2_results.ecsv"],
            "f1": [1.0, 2.0, 3.0],
            "pulsed_rate": [0.1, 0.2, 0.3],
        }
    )
    rates = rate_values(table, exclude=["chandra*"])
    assert np.isnan(rates[1]) and rates[0] == 0.1 and rates[2] == 0.3
    assert table["f1"][1] == 2.0


def test_joint_search_finds_shared_period_and_lag():
    """Two quantities sharing a period, the second lagging by a quarter cycle, give
    a joint peak at that period with the lag measured."""
    rng = np.random.default_rng(4)
    t = _sparse_times(rng)
    phase = 2 * np.pi * t / 61.0
    y1 = 2 * np.cos(phase) + rng.normal(0, 1, t.size)
    y2 = 2 * np.cos(phase - np.pi / 2) + rng.normal(0, 1, t.size)
    result = search_joint_periodicity(t, [y1, y2], [None, None], 50, 70, n_shuffle=200, rng=rng)
    assert abs(result["best_period_days"] - 61.0) < 0.5
    assert result["fap"] < 0.01
    assert abs(result["phase_lags"][1] - 0.25) < 0.05


def test_joint_search_null_keeps_epochs_paired():
    """A near-copy of a weakly periodic quantity adds no evidence: with whole epochs
    shuffled, the joint false-alarm probability stays unremarkable (about 0.14 here),
    where shuffling each quantity independently would report about 0.006."""
    rng = np.random.default_rng(5)
    t = _sparse_times(rng)
    y1 = 1.2 * np.sin(2 * np.pi * t / 61) + rng.normal(0, 1, t.size)
    y2 = y1 + rng.normal(0, 0.1, t.size)
    result = search_joint_periodicity(t, [y1, y2], [None, None], 50, 70, n_shuffle=500, rng=1)
    assert result["fap"] > 0.05
