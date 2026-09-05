"""Tests for ``ell1decay``: the PBDOT/PBDDOT fit across per-epoch ell1fit results.

Two levels are tested. The nested-sampling fit itself
(:mod:`ell1fit.orbital_decay_model`/:mod:`ell1fit.orbital_decay_sampling`) is
tested directly against synthetic ``delta_tasc`` data with known injected
coefficients -- real end-to-end sampler runs, no mocking, following this
package's convention elsewhere (e.g. ``test_eccentricity.py``). The file-
handling layer (:mod:`ell1fit.orbital_decay_data`) is tested against small
hand-built ``.ecsv`` fixtures, since its job is entirely about what that file
format can and cannot represent.
"""

import os

import numpy as np
import pytest
from astropy.table import Table

pytest.importorskip("dynesty")

from ..mcmc_utils import plot_mcmc_comparison
from ..orbital_decay import fit_orbital_decay
from ..orbital_decay_data import (
    OrbitalModelCompatibilityError,
    _build_models,
    _float128_to_float64_header,
    build_reference_model,
    check_compatibility,
    load_epochs,
    read_result_table,
)
from ..orbital_decay_model import (
    delta_tasc_model,
    derivative_scale,
    log_likelihood_asymmetric_errors,
    physical_from_beta,
)
from ..orbital_decay_sampling import (
    bayes_factor,
    default_bounds,
    laplace_cross_check,
    run_seed_scatter,
)


#: Small enough for test runtime, per this package's convention for sampler
#: tests; large enough that the peak-shortfall gate reports convergence.
NLIVE = 250
N_SEEDS = 2
SEED = 20260824


# ---------------------------------------------------------------------------
# orbital_decay_model / orbital_decay_sampling: fit against synthetic data
# ---------------------------------------------------------------------------


def _synthetic_dataset(beta_true, baseline_days, n_points=25, noise_sec=5.0, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-baseline_days / 2, baseline_days / 2, n_points))
    y = delta_tasc_model(beta_true, x, baseline_days) + rng.normal(0, noise_sec, size=x.size)
    yerrn = yerrp = np.full_like(x, noise_sec)
    return x, y, yerrn, yerrp


def _fit(order, x, y, yerrn, yerrp, baseline_days, labels):
    def loglikelihood(beta):
        return log_likelihood_asymmetric_errors(beta, x, y, yerrn, yerrp, baseline_days)

    bounds = default_bounds(y, order)
    return run_seed_scatter(loglikelihood, bounds, labels, n_seeds=N_SEEDS, nlive=NLIVE, dlogz=0.3)


@pytest.fixture(scope="module")
def null_case():
    """Data generated from M0 (no PBDDOT): fit both M0 and M1 against it."""
    baseline_days = 3000.0
    beta_true = np.array([50.0, 30.0, -800.0])
    x, y, yerrn, yerrp = _synthetic_dataset(beta_true, baseline_days)

    mlin = _fit(1, x, y, yerrn, yerrp, baseline_days, ["b0", "b1"])
    m0 = _fit(2, x, y, yerrn, yerrp, baseline_days, ["b0", "b1", "b2"])
    m1 = _fit(3, x, y, yerrn, yerrp, baseline_days, ["b0", "b1", "b2", "b3"])
    return {
        "beta_true": beta_true,
        "baseline_days": baseline_days,
        "x": x,
        "y": y,
        "mlin": mlin,
        "m0": m0,
        "m1": m1,
    }


@pytest.fixture(scope="module")
def no_pbdot_case():
    """Data generated with no quadratic term at all: the null case for PBDOT.

    Deliberately built from an order-1 ``beta_true``, so the quadratic
    coefficient is exactly zero rather than merely small -- a "PBDOT is not
    detected" test has to be run against data that genuinely has none.
    """
    baseline_days = 3000.0
    beta_true = np.array([50.0, 30.0])
    x, y, yerrn, yerrp = _synthetic_dataset(beta_true, baseline_days, seed=SEED + 2)

    mlin = _fit(1, x, y, yerrn, yerrp, baseline_days, ["b0", "b1"])
    m0 = _fit(2, x, y, yerrn, yerrp, baseline_days, ["b0", "b1", "b2"])
    return {
        "beta_true": beta_true,
        "baseline_days": baseline_days,
        "x": x,
        "y": y,
        "mlin": mlin,
        "m0": m0,
    }


@pytest.fixture(scope="module")
def pbddot_case():
    """Data generated from M1, with a clearly-detectable cubic term."""
    baseline_days = 3000.0
    beta_true = np.array([50.0, 30.0, -800.0, 600.0])
    x, y, yerrn, yerrp = _synthetic_dataset(beta_true, baseline_days, noise_sec=3.0, seed=SEED + 1)

    m0 = _fit(2, x, y, yerrn, yerrp, baseline_days, ["b0", "b1", "b2"])
    m1 = _fit(3, x, y, yerrn, yerrp, baseline_days, ["b0", "b1", "b2", "b3"])
    return {
        "beta_true": beta_true,
        "baseline_days": baseline_days,
        "x": x,
        "y": y,
        "m0": m0,
        "m1": m1,
    }


def test_null_case_recovers_beta_within_3sigma(null_case):
    beta_16, beta_84 = np.percentile(null_case["m0"]["flat_samples"], [16, 84], axis=0)
    sigma = (beta_84 - beta_16) / 2
    pull = (np.median(null_case["m0"]["flat_samples"], axis=0) - null_case["beta_true"]) / sigma
    assert np.all(np.abs(pull) < 3), f"pulls too large: {pull}"


def test_null_case_bayes_factor_favors_m0(null_case):
    bf = bayes_factor(null_case["m0"], null_case["m1"])
    assert bf["ln_bf"] < 0
    assert "M0" in bf["interpretation"]


def test_bayes_factor_labels_are_configurable():
    """The PBDOT comparison is between a different pair of models than the
    PBDDOT one, so the interpretation text cannot name M0 and M1."""
    lower = {"log_evidence": 0.0, "log_evidence_err": 0.1}
    higher = {"log_evidence": 10.0, "log_evidence_err": 0.1}
    bf = bayes_factor(lower, higher, lower_label="MLIN", higher_label="M0")
    assert "M0" in bf["interpretation"]
    assert "MLIN" not in bf["interpretation"]
    assert bayes_factor(higher, lower, lower_label="MLIN", higher_label="M0")["interpretation"] == (
        "very strong evidence for MLIN"
    )


def test_pbdot_bayes_factor_favors_the_quadratic_when_pbdot_is_present(null_case):
    """null_case's data carries a large quadratic term, so dropping PBDOT
    entirely must be strongly disfavoured."""
    bf = bayes_factor(null_case["mlin"], null_case["m0"], lower_label="MLIN", higher_label="M0")
    assert bf["ln_bf"] > 0
    assert "M0" in bf["interpretation"]


def test_pbdot_bayes_factor_favors_the_linear_model_when_pbdot_is_absent(no_pbdot_case):
    bf = bayes_factor(
        no_pbdot_case["mlin"], no_pbdot_case["m0"], lower_label="MLIN", higher_label="M0"
    )
    assert bf["ln_bf"] < 0
    assert "MLIN" in bf["interpretation"]


def test_linear_fit_recovers_its_own_beta_within_3sigma(no_pbdot_case):
    beta_16, beta_84 = np.percentile(no_pbdot_case["mlin"]["flat_samples"], [16, 84], axis=0)
    sigma = (beta_84 - beta_16) / 2
    pull = (
        np.median(no_pbdot_case["mlin"]["flat_samples"], axis=0) - no_pbdot_case["beta_true"]
    ) / sigma
    assert np.all(np.abs(pull) < 3), f"pulls too large: {pull}"


def test_pbddot_case_recovers_beta3_within_3sigma(pbddot_case):
    beta_16, beta_84 = np.percentile(pbddot_case["m1"]["flat_samples"], [16, 84], axis=0)
    sigma = (beta_84 - beta_16) / 2
    pull = (np.median(pbddot_case["m1"]["flat_samples"], axis=0) - pbddot_case["beta_true"]) / sigma
    assert np.all(np.abs(pull) < 3), f"pulls too large: {pull}"


def test_pbddot_case_bayes_factor_favors_m1(pbddot_case):
    bf = bayes_factor(pbddot_case["m0"], pbddot_case["m1"])
    assert bf["ln_bf"] > 0
    assert "M1" in bf["interpretation"]


def test_physical_from_beta_recovers_injected_pbdot_and_pbddot():
    """Cross-check against an exact phase-integral simulation, independent of
    the fit above -- see the derivation and this same check in
    ``orbital_decay_model``'s module docstring."""
    from scipy.integrate import quad
    from scipy.optimize import brentq

    pb0, pbdot, pbddot, baseline = 1.7, 3e-8, 5e-12, 3000.0

    def period(t):
        return pb0 + pbdot * t + 0.5 * pbddot * t**2

    def n_cycles(t):
        return quad(lambda tp: 1.0 / period(tp), 0, t, limit=200)[0]

    xs = np.linspace(-baseline, baseline, 11)
    delta_tasc_exact = []
    for x in xs:
        target = x / pb0
        if abs(target) < 1e-8:
            delta_tasc_exact.append(0.0)
            continue
        lo, hi = sorted([x - 5, x + 5])

        def f(t):  # noqa: B023
            return n_cycles(t) - target

        while f(lo) * f(hi) > 0:
            lo, hi = lo - 5, hi + 5
        delta_tasc_exact.append(brentq(f, lo, hi, xtol=1e-10) - x)
    delta_tasc_exact_sec = np.array(delta_tasc_exact) * 86400.0

    tau = xs / baseline
    design = np.vstack([tau**2, tau**3]).T
    beta2, beta3 = np.linalg.lstsq(design, delta_tasc_exact_sec, rcond=None)[0]

    physical = physical_from_beta([0.0, 0.0, beta2, beta3], baseline, pb0)
    assert physical["PBDOT"] == pytest.approx(pbdot, rel=1e-4)
    assert physical["PBDDOT"] == pytest.approx(pbdot * 0 + pbddot * 365.25, rel=2e-3)


def test_laplace_cross_check_agrees_with_seed_scatter(null_case):
    m0 = null_case["m0"]
    baseline_days = null_case["baseline_days"]
    x, y = null_case["x"], null_case["y"]
    yerrn = yerrp = np.full_like(x, 5.0)

    def loglikelihood(beta):
        return log_likelihood_asymmetric_errors(beta, x, y, yerrn, yerrp, baseline_days)

    laplace = laplace_cross_check(loglikelihood, m0["bounds"], m0["map_position"])
    assert abs(laplace - m0["log_evidence"]) < 5.0


def test_plot_mcmc_comparison_smoke(null_case, tmp_path):
    fname = str(tmp_path / "comparison.jpg")
    plot_mcmc_comparison(
        [null_case["m0"]["flat_samples"], null_case["m1"]["flat_samples"]],
        [["b0", "b1", "b2"], ["b0", "b1", "b2", "b3"]],
        ["M0", "M1"],
        fname,
    )
    assert os.path.exists(fname)
    assert os.path.getsize(fname) > 0


# ---------------------------------------------------------------------------
# orbital_decay_data: file handling, compatibility check
# ---------------------------------------------------------------------------

_TASC_PERCENTILE_COLUMNS = ("1", "10", "16", "50", "84", "90", "99")


def _write_ecsv(
    path,
    pepoch,
    pb,
    tasc,
    a1=22.0,
    eps1=0.0,
    eps2=0.0,
    pbdot=0.0,
    tasc_spread_days=0.001,
    missing=(),
):
    """A minimal single-epoch ell1fit result file, with just enough columns
    for orbital_decay_data to read (see its ``_REQUIRED_COLUMNS``).

    ``tasc_spread_days`` sets the fitted TASC's 1-sigma half-width (equal on
    both sides here). ``missing`` drops named columns, to build the
    reader-edge-case fixtures.
    """
    columns = {
        "PEPOCH": [pepoch],
        "PB": [pb],
        "TASC": [tasc],
        "A1": [a1],
        "EPS1": [eps1],
        "EPS2": [eps2],
        "PBDOT": [pbdot],
    }
    # dTASC_mean/_initial/_factor + percentiles reconstruct `tasc` exactly
    # via retrieve_value_and_error: mean*factor + initial == tasc.
    columns["dTASC_initial"] = [tasc]
    columns["dTASC_mean"] = [0.0]
    columns["dTASC_factor"] = [1.0]
    for p in _TASC_PERCENTILE_COLUMNS:
        # 16/50/84 spread of tasc_spread_days either side of the mean, in
        # "mean units" (factor=1 here, so this is directly in days).
        spread = {"16": -tasc_spread_days, "50": 0.0, "84": tasc_spread_days}.get(p, 0.0)
        columns[f"dTASC_{p}"] = [spread]

    for name in missing:
        columns.pop(name, None)

    Table(columns).write(str(path), format="ascii.ecsv", overwrite=True)
    return str(path)


def test_load_epochs_reads_real_column_layout(tmp_path):
    fname = _write_ecsv(tmp_path / "epoch0.ecsv", pepoch=57000.0, pb=1.7, tasc=57000.1)
    (epoch,) = load_epochs([fname])
    assert epoch.pepoch == pytest.approx(57000.0)
    assert epoch.tasc == pytest.approx(57000.1)


def test_float128_header_is_rewritten_to_float64():
    text = "# - {name: PEPOCH, datatype: float128}\n# - {name: PB, datatype: float64}\n"
    rewritten = _float128_to_float64_header(text)
    assert "float128" not in rewritten
    assert "PEPOCH, datatype: float64" in rewritten


def test_joint_fit_output_raises_clear_error(tmp_path):
    fname = str(tmp_path / "joint.ecsv")
    Table({"PEPOCH_0": [57000.0], "PB": [1.7]}).write(fname, format="ascii.ecsv")
    with pytest.raises(OrbitalModelCompatibilityError, match="joint multi-epoch"):
        read_result_table(fname)


def test_rayleigh_only_output_raises_clear_error(tmp_path):
    fname = _write_ecsv(
        tmp_path / "rayleigh.ecsv", pepoch=57000.0, pb=1.7, tasc=57000.1, missing=("dTASC_mean",)
    )
    with pytest.raises(OrbitalModelCompatibilityError, match="F0/F1-only"):
        read_result_table(fname)


def test_check_compatibility_passes_for_consistent_pbdot_zero_files(tmp_path):
    files = [
        _write_ecsv(tmp_path / "e0.ecsv", pepoch=57000.0, pb=1.7, tasc=57000.1),
        _write_ecsv(tmp_path / "e1.ecsv", pepoch=57100.0, pb=1.7, tasc=57100.05),
        _write_ecsv(tmp_path / "e2.ecsv", pepoch=57200.0, pb=1.7, tasc=57200.02),
    ]
    epochs = load_epochs(files)
    check_compatibility(epochs, tolerance=1e-9)  # must not raise

    ref = build_reference_model(epochs)
    assert ref.PBDOT.value == 0.0
    assert ref.PEPOCH.value == pytest.approx(np.mean([57000.0, 57100.0, 57200.0]))


def test_check_compatibility_fires_on_inconsistent_a1(tmp_path):
    files = [
        _write_ecsv(tmp_path / "e0.ecsv", pepoch=57000.0, pb=1.7, tasc=57000.1, a1=22.0),
        _write_ecsv(tmp_path / "e1.ecsv", pepoch=57100.0, pb=1.7, tasc=57100.05, a1=22.5),
    ]
    epochs = load_epochs(files)
    with pytest.raises(OrbitalModelCompatibilityError, match="e1.ecsv"):
        check_compatibility(epochs, tolerance=1e-9)


def test_check_compatibility_warns_but_does_not_abort_on_small_pbdot_mismatch(tmp_path, caplog):
    """A file-to-file PBDOT difference whose spurious-delta_tasc impact stays
    well under a file's own TASC uncertainty should warn, not abort -- this
    mirrors a real disagreement measured between two M82 X-2 processing
    batches.

    The ``PB`` column is in *seconds* (see ``models.py``'s
    ``_OFFSET_PARAMETERS``), and a real per-file PB is already the shared
    orbit propagated to that file's own epoch using *that file's own*
    reported PBDOT -- so e1's PB here is built the same way, not just copied
    from e0, or the fixture would itself look like a PB disagreement
    unrelated to the PBDOT difference under test.
    """
    pb0_days = 2.53
    pepoch0, pepoch1 = 57000.0, 60748.0
    pbdot0 = -5.70e-8
    pbdot1 = -5.7e-8 + 1.11e-10
    pb0_sec = pb0_days * 86400.0
    pb1_sec = pb0_sec + pbdot1 * (pepoch1 - pepoch0) * 86400.0

    files = [
        _write_ecsv(
            tmp_path / "e0.ecsv",
            pepoch=pepoch0,
            pb=pb0_sec,
            tasc=57000.1,
            pbdot=pbdot0,
            tasc_spread_days=0.00233,
        ),
        _write_ecsv(
            tmp_path / "e1.ecsv",
            pepoch=pepoch1,
            pb=pb1_sec,
            tasc=60748.05,
            pbdot=pbdot1,
            tasc_spread_days=0.00233,
        ),
    ]
    epochs = load_epochs(files)
    with caplog.at_level("WARNING"):
        check_compatibility(epochs, tolerance=1e-9)  # must not raise
    assert any("spurious delta_tasc" in message for message in caplog.messages)


def test_check_compatibility_fires_on_large_pbdot_mismatch(tmp_path):
    """A PBDOT difference large enough that its spurious-delta_tasc impact
    would rival the epoch's own (tight) TASC uncertainty is unsound and
    still aborts.

    ``pb1_sec`` is built as ``pb0_sec + 86400*dt*mean_pbdot`` (mean_pbdot =
    the average of the two files' own PBDOT) -- algebraically exactly what
    "predicted_pb_seconds + explained_sec" reduces to for a 2-file case in
    ``check_compatibility`` (explained_sec uses dt relative to the *mean*
    epoch, which for 2 files is always the midpoint). That makes the
    leftover residual exactly zero regardless of how large the PBDOT
    mismatch is, isolating this test to the PBDOT-unsound check rather than
    also tripping the unrelated PB-residual check. (Both PBDOT values here
    happen to stay under 1e-7 in magnitude, but that no longer matters --
    see test_epoch_pbdot_above_1e7_threshold_is_not_corrupted below for the
    PINT parfile-parsing quirk this used to be sensitive to.)
    """
    pb0_days = 1.7
    pepoch0, pepoch1 = 57000.0, 58000.0
    pbdot0 = -5.7e-8
    pbdot1 = -9.9e-8
    dt_days = pepoch1 - pepoch0
    mean_pbdot = (pbdot0 + pbdot1) / 2.0
    pb0_sec = pb0_days * 86400.0
    pb1_sec = pb0_sec + 86400.0 * dt_days * mean_pbdot

    files = [
        _write_ecsv(
            tmp_path / "e0.ecsv",
            pepoch=pepoch0,
            pb=pb0_sec,
            tasc=57000.1,
            pbdot=pbdot0,
            tasc_spread_days=0.0001,
        ),
        _write_ecsv(
            tmp_path / "e1.ecsv",
            pepoch=pepoch1,
            pb=pb1_sec,
            tasc=58000.05,
            pbdot=pbdot1,
            tasc_spread_days=0.0001,
        ),
    ]
    epochs = load_epochs(files)
    with pytest.raises(OrbitalModelCompatibilityError, match="unsound"):
        check_compatibility(epochs, tolerance=1e-9)


def test_check_compatibility_fires_on_pb_unexplained_by_pbdot(tmp_path):
    """A PB disagreement with no matching PBDOT difference to explain it (a
    genuinely different orbital model, not just a differing PBDOT
    assumption) has no benign explanation and must still hard-abort."""
    files = [
        _write_ecsv(
            tmp_path / "e0.ecsv", pepoch=57000.0, pb=1.7 * 86400.0, tasc=57000.1, pbdot=0.0
        ),
        _write_ecsv(
            tmp_path / "e1.ecsv", pepoch=57100.0, pb=2.5 * 86400.0, tasc=57100.05, pbdot=0.0
        ),
    ]
    epochs = load_epochs(files)
    with pytest.raises(OrbitalModelCompatibilityError, match="unexplained"):
        check_compatibility(epochs, tolerance=1e-9)


def test_epoch_pbdot_above_1e7_threshold_is_not_corrupted(tmp_path):
    """PINT's parfile parser (``floatParameter._set_quantity``) assumes any
    PBDOT magnitude above 1e-7 was written in the "x1e-12" pulsar-timing
    convention and silently multiplies it by 1e-12 -- a 12-order-of-magnitude
    corruption with no error raised. Real M82 X-2 PBDOT (~5.7e-8) never
    crosses that threshold, so this was previously latent; a large PBDOT like
    the one used here (5.7e-6) would previously have been read back as
    ~5.7e-18, which check_compatibility would have seen as PBDOT=0 --
    silently failing to explain the PB difference between the two files
    below and firing a spurious "unexplained" error, or (had PB been built
    from the corrupted value too) simply losing the decay signal outright.
    """
    pb0_days = 1.7
    pepoch0, pepoch1 = 57000.0, 58000.0
    pbdot = 5.7e-6
    pb0_sec = pb0_days * 86400.0
    pb1_sec = pb0_sec + pbdot * (pepoch1 - pepoch0) * 86400.0

    files = [
        _write_ecsv(tmp_path / "e0.ecsv", pepoch=pepoch0, pb=pb0_sec, tasc=57000.1, pbdot=pbdot),
        _write_ecsv(tmp_path / "e1.ecsv", pepoch=pepoch1, pb=pb1_sec, tasc=58000.05, pbdot=pbdot),
    ]
    epochs = load_epochs(files)

    model_list, _, ref_model = _build_models(epochs)
    for model in model_list:
        assert model.PBDOT.value == pytest.approx(pbdot)
    assert ref_model.PBDOT.value == pytest.approx(pbdot)

    check_compatibility(epochs, tolerance=1e-9)  # must not raise


# ---------------------------------------------------------------------------
# orbital_decay: the reported measurement-or-limit for each PB derivative
# ---------------------------------------------------------------------------

#: A 1.7-day orbit, the scale of the systems ell1fit is used on. The ``PB``
#: column of an ell1fit result file is in *seconds*, hence the conversion at
#: the point of writing.
_PB_DAYS = 1.7
_REF_MJD = 57000.0


def _write_decay_epochs(directory, pbdot=0.0, pbddot_per_yr=0.0, tasc_err_sec=5.0, seed=SEED):
    """A set of per-epoch ``.ecsv`` files carrying an injected PBDOT/PBDDOT.

    The TASC of each epoch is the ascending node a fixed-period ephemeris
    would predict, displaced by the ``delta_tasc(t)`` the injected
    derivatives produce (the same closed form
    :mod:`ell1fit.orbital_decay_model` fits), plus Gaussian noise of
    ``tasc_err_sec``.
    """
    rng = np.random.default_rng(seed)
    files = []
    for i, dt in enumerate(np.linspace(-1500.0, 1500.0, 9)):
        delta_sec = (
            pbdot * dt**2 / (2 * _PB_DAYS) + (pbddot_per_yr / 365.25) * dt**3 / (6 * _PB_DAYS)
        ) * 86400.0
        n_orbits = round(dt / _PB_DAYS)
        tasc = (
            _REF_MJD + n_orbits * _PB_DAYS + (delta_sec + rng.normal(0.0, tasc_err_sec)) / 86400.0
        )
        files.append(
            _write_ecsv(
                os.path.join(str(directory), f"epoch{i}.ecsv"),
                pepoch=_REF_MJD + dt,
                pb=_PB_DAYS * 86400.0,
                tasc=tasc,
                tasc_spread_days=tasc_err_sec / 86400.0,
            )
        )
    return files


def _run_decay(directory, **kwargs):
    return fit_orbital_decay(
        _write_decay_epochs(directory, **kwargs),
        outroot=os.path.join(str(directory), "decay"),
        nlive=200,
        dlogz=0.5,
        seeds=2,
        reference_epoch=_REF_MJD,
        write_parfile=False,
    )


@pytest.fixture(scope="module")
def strong_pbdot_run(tmp_path_factory):
    """A large injected PBDOT and no PBDDOT: one run exercising both branches,
    a measurement for PBDOT and an upper limit for PBDDOT."""
    return _run_decay(tmp_path_factory.mktemp("strong_pbdot"), pbdot=3e-8)


@pytest.fixture(scope="module")
def flat_run(tmp_path_factory):
    """No period derivative of any order: both must come back as limits."""
    return _run_decay(tmp_path_factory.mktemp("flat"), seed=SEED + 3)


def test_injected_pbdot_is_recovered(strong_pbdot_run):
    """Guards the whole chain of unit conversions between the .ecsv files and
    the reported PBDOT, not just the summary formatting."""
    assert strong_pbdot_run["M0"]["PBDOT"] == pytest.approx(3e-8, rel=1e-2)


def test_detected_pbdot_is_quoted_as_a_measurement(strong_pbdot_run):
    m0 = strong_pbdot_run["M0"]
    assert m0["PBDOT_detected"] is True
    assert np.isnan(m0["PBDOT_upper_limit"])
    assert "upper limit" not in m0["PBDOT_summary"]
    assert m0["PBDOT_significance_sigma"] > 10


def test_absent_pbddot_is_quoted_as_a_limit(strong_pbdot_run):
    m1 = strong_pbdot_run["M1"]
    assert m1["PBDDOT_detected"] is False
    assert np.isfinite(m1["PBDDOT_upper_limit"])
    assert m1["PBDDOT_upper_limit"] > 0
    assert "|PBDDOT| <" in m1["PBDDOT_summary"]
    assert "1/yr" in m1["PBDDOT_summary"]


def test_both_derivatives_are_limits_when_the_data_is_flat(flat_run):
    assert flat_run["M0"]["PBDOT_detected"] is False
    assert flat_run["M1"]["PBDDOT_detected"] is False
    assert np.isfinite(flat_run["M0"]["PBDOT_upper_limit"])
    assert np.isfinite(flat_run["M1"]["PBDDOT_upper_limit"])


def test_limit_brackets_the_injected_zero(flat_run):
    """A limit that did not contain the truth would be a wrong limit."""
    assert flat_run["M0"]["PBDOT_2sigma_lo"] < 0 < flat_run["M0"]["PBDOT_2sigma_hi"]
    assert flat_run["M1"]["PBDDOT_2sigma_lo"] < 0 < flat_run["M1"]["PBDDOT_2sigma_hi"]


def test_intervals_are_properly_nested(flat_run):
    for model, name in (("M0", "PBDOT"), ("M1", "PBDDOT")):
        block = flat_run[model]
        assert (
            block[f"{name}_3sigma_lo"]
            < block[f"{name}_2sigma_lo"]
            < block[f"{name}_1sigma_lo"]
            < block[f"{name}_1sigma_hi"]
            < block[f"{name}_2sigma_hi"]
            < block[f"{name}_3sigma_hi"]
        )


def test_three_sigma_limit_is_reported_alongside_the_headline_one(flat_run):
    """A three-sigma bound is what a non-detection is often quoted as."""
    for model, name in (("M0", "PBDOT"), ("M1", "PBDDOT")):
        block = flat_run[model]
        assert block[f"{name}_upper_limit_3sigma"] > block[f"{name}_upper_limit"]
        assert block[f"{name}_upper_limit_3sigma_level"] == pytest.approx(0.9973, abs=1e-4)
    assert "3 sigma" in flat_run["M0"]["PBDOT_summary"]


def test_upper_limit_level_is_honoured(tmp_path):
    """A more demanding credible level has to give a larger limit."""
    files = _write_decay_epochs(tmp_path, seed=SEED + 4)
    common = dict(nlive=200, dlogz=0.5, seeds=1, reference_epoch=_REF_MJD, write_parfile=False)
    loose = fit_orbital_decay(
        files, outroot=os.path.join(str(tmp_path), "loose"), upper_limit_level=0.68, **common
    )
    tight = fit_orbital_decay(
        files, outroot=os.path.join(str(tmp_path), "tight"), upper_limit_level=0.99, **common
    )
    assert tight["M1"]["PBDDOT_upper_limit"] > loose["M1"]["PBDDOT_upper_limit"]
    assert loose["M1"]["PBDDOT_upper_limit_level"] == 0.68


def test_detection_threshold_is_honoured(strong_pbdot_run, tmp_path):
    """An absurdly high threshold must turn even the strong PBDOT into a limit,
    proving the Bayes factor is what drives the switch."""
    files = _write_decay_epochs(tmp_path, pbdot=3e-8)
    strict = fit_orbital_decay(
        files,
        outroot=os.path.join(str(tmp_path), "strict"),
        nlive=200,
        dlogz=0.5,
        seeds=1,
        reference_epoch=_REF_MJD,
        write_parfile=False,
        detection_ln_bf=1e9,
    )
    assert strong_pbdot_run["M0"]["PBDOT_detected"] is True
    assert strict["M0"]["PBDOT_detected"] is False
    assert np.isfinite(strict["M0"]["PBDOT_upper_limit"])


def test_results_json_carries_the_limits(tmp_path):
    """The summaries have to survive the trip through json.dump."""
    import json

    files = _write_decay_epochs(tmp_path, pbdot=3e-8)
    outroot = os.path.join(str(tmp_path), "json")
    fit_orbital_decay(
        files,
        outroot=outroot,
        nlive=200,
        dlogz=0.5,
        seeds=1,
        reference_epoch=_REF_MJD,
        write_parfile=False,
    )
    with open(outroot + "_results.json") as fobj:
        stored = json.load(fobj)
    assert "PBDOT_summary" in stored["M0"]
    assert "PBDDOT_upper_limit" in stored["M1"]
    assert "PBDDOT_upper_limit_3sigma" in stored["M1"]
    assert "bayes_factor_pbdot" in stored


def test_derivative_scale_matches_physical_from_beta():
    """The bulk chain conversion and the per-point one must agree exactly."""
    beta = np.array([1.0, 2.0, 3.0, 4.0])
    physical = physical_from_beta(beta, 3000.0, 1.7)
    assert beta[2] * derivative_scale(2, 3000.0, 1.7) == pytest.approx(physical["PBDOT"])
    assert beta[3] * derivative_scale(3, 3000.0, 1.7) == pytest.approx(physical["PBDDOT"])
