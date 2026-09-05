r"""Combining the ``EPS1``/``EPS2`` posteriors into one posterior on ``e``.

Every test here builds its samples analytically rather than by fitting, so that
the right answer is known in closed form. Two limits do most of the work:

* **Pure noise.** If ``EPS1`` and ``EPS2`` are independent Gaussians of width
  :math:`\sigma` centred on the origin -- no eccentricity at all -- then
  :math:`e = \sqrt{\epsilon_1^2+\epsilon_2^2}` is Rayleigh distributed, whose
  95th percentile is :math:`\sigma\sqrt{-2\ln 0.05} = 2.4477\,\sigma`. That is
  the number the upper limit has to reproduce, and the fact that it is not zero
  is the whole reason an upper limit is needed.
* **Pure noise, reweighted.** Undoing the :math:`p(e)\propto e` prior turns
  that Rayleigh into a half-normal, whose 95th percentile is
  :math:`1.9600\,\sigma`.

The Monte Carlo error on a percentile from ``NSAMPLES`` draws is well under a
percent, so the tolerances below are set by the estimator and not fitted to the
answer.
"""

import numpy as np
import pytest
from astropy.table import Table

from ..eccentricity import (
    RESULTS_SUFFIX,
    default_chain_file,
    draw_eccentricity_posterior,
    eccentricity_and_omega,
    eccentricity_summary,
    eccentricity_summary_from_run,
    eps_samples_from_chain,
    load_eps_samples,
    plot_eccentricity_posterior,
    zero_eccentricity_exclusion,
)
from ..mcmc_utils import SAMPLES_SUFFIX, load_flat_samples, save_flat_samples
from ..pipeline import _enrich_results_with_eccentricity


SEED = 20260903
NSAMPLES = 200_000

#: Per-component posterior width used by every synthetic case below.
SIGMA = 1e-4

#: 95th percentiles of the Rayleigh and half-normal distributions, in units of
#: SIGMA: sqrt(-2 ln 0.05) and the 0.975 quantile of a standard normal.
RAYLEIGH_95 = 2.447747
HALF_NORMAL_95 = 1.959964


def _eps_samples(ecc=0.0, omega_deg=0.0, sigma=SIGMA, correlation=0.0, size=NSAMPLES):
    """Gaussian posterior samples of (EPS1, EPS2) around a given orbit."""
    omega = np.radians(omega_deg)
    mean = [ecc * np.sin(omega), ecc * np.cos(omega)]
    covariance = np.array([[sigma**2, correlation * sigma**2], [correlation * sigma**2, sigma**2]])
    rng = np.random.default_rng(SEED)
    samples = rng.multivariate_normal(mean, covariance, size=size)
    return samples[:, 0], samples[:, 1]


def test_transform_follows_the_ell1_definition():
    """EPS1 = e sin(omega), EPS2 = e cos(omega), inverted sample by sample."""
    ecc_in = np.array([2e-3, 5e-4, 1e-2])
    omega_in = np.array([71.0, 190.0, 355.0])
    eps1 = ecc_in * np.sin(np.radians(omega_in))
    eps2 = ecc_in * np.cos(np.radians(omega_in))

    ecc, omega = eccentricity_and_omega(eps1, eps2)

    assert np.allclose(ecc, ecc_in)
    assert np.allclose(omega, omega_in)


def test_strong_detection_is_quoted_as_a_measurement():
    """Twenty sigma out: a median, an interval, an angle, and no upper limit."""
    ecc_in, omega_in = 20 * SIGMA, 71.0
    summary = eccentricity_summary(*_eps_samples(ecc_in, omega_in))

    assert summary["ECC_detected"]
    assert summary["ECC_50"] == pytest.approx(ecc_in, rel=0.01)
    # A 20-sigma length is unbiased to O(1/40), and its width is the component
    # width: the transformation is locally a rotation.
    assert summary["ECC_84"] - summary["ECC_16"] == pytest.approx(2 * SIGMA, rel=0.02)
    assert summary["ECC_significance_sigma"] > 15
    assert summary["ECC_zero_credibility"] > 1 - 1e-10
    assert np.isnan(summary["ECC_upper_limit"])

    assert summary["OM_deg_mean"] == pytest.approx(omega_in, abs=0.5)
    assert summary["OM_concentration"] > 0.99


def test_pure_noise_is_not_a_detection_and_gets_a_rayleigh_upper_limit():
    """No eccentricity at all still gives a positive e: quote a limit, not a value."""
    summary = eccentricity_summary(*_eps_samples(ecc=0.0))

    assert not summary["ECC_detected"]
    assert summary["ECC_significance_sigma"] < 1.0
    assert summary["ECC_upper_limit"] == pytest.approx(RAYLEIGH_95 * SIGMA, rel=0.02)
    assert summary["ECC_upper_limit_level"] == 0.95
    assert "upper limit" in summary["ECC_summary"]

    # The trap this whole module exists to avoid: the median of the noise is
    # 1.18 sigma, a number that looks like a detection and is not.
    assert summary["ECC_50"] == pytest.approx(1.17741 * SIGMA, rel=0.02)

    # With no eccentricity there is no periastron: the angle is uniform.
    assert summary["OM_concentration"] < 0.02


def test_flat_in_e_prior_tightens_the_limit_to_the_half_normal():
    """Reweighting by 1/e undoes p(e) ~ e, turning the Rayleigh into a half-normal."""
    eps1, eps2 = _eps_samples(ecc=0.0)
    default = eccentricity_summary(eps1, eps2)
    flat = eccentricity_summary(eps1, eps2, flat_in_e_prior=True)

    assert flat["ECC_upper_limit"] == pytest.approx(HALF_NORMAL_95 * SIGMA, rel=0.03)
    assert flat["ECC_upper_limit"] < default["ECC_upper_limit"]
    assert flat["ECC_prior"] == "flat in e"
    # The detection test is deliberately unmoved by the radial prior.
    assert flat["ECC_significance_sigma"] == pytest.approx(default["ECC_significance_sigma"])


def test_marginals_can_hide_a_detection_that_the_joint_posterior_makes():
    """Why quadrature on the published error bars is the wrong recipe.

    Strongly correlated components, with the orbit displaced along the *short*
    axis of the posterior ellipse: each parameter on its own sits only 1.5
    sigma from zero, and anyone combining the two marginals would report no
    eccentricity. The joint posterior excludes the origin overwhelmingly.
    """
    eps1, eps2 = _eps_samples(ecc=1.5 * np.sqrt(2) * SIGMA, omega_deg=135.0, correlation=0.98)

    assert abs(np.mean(eps1)) / np.std(eps1) == pytest.approx(1.5, rel=0.05)
    assert abs(np.mean(eps2)) / np.std(eps2) == pytest.approx(1.5, rel=0.05)

    summary = eccentricity_summary(eps1, eps2)
    assert summary["ECC_detected"]
    assert summary["ECC_significance_sigma"] > 10


def test_exclusion_matches_the_analytic_two_degree_of_freedom_result():
    """A mean three sigma from the origin in an isotropic posterior.

    d^2 = 9, so the origin sits on the 1 - exp(-4.5) = 98.89% contour, which in
    one-dimensional Gaussian language is 2.54 sigma -- less than three, because
    two free directions offer more ways to land far from the centre.
    """
    credibility, sigma = zero_eccentricity_exclusion(*_eps_samples(ecc=3 * SIGMA))

    assert credibility == pytest.approx(1 - np.exp(-4.5), abs=0.005)
    assert sigma == pytest.approx(2.5415, abs=0.02)


def test_a_borderline_case_reports_a_limit_unless_the_threshold_is_lowered():
    """The detection threshold is a knob, and it is the one that picks the branch."""
    eps1, eps2 = _eps_samples(ecc=3 * SIGMA)

    assert not eccentricity_summary(eps1, eps2)["ECC_detected"]
    relaxed = eccentricity_summary(eps1, eps2, detection_sigma=2.0)
    assert relaxed["ECC_detected"]
    assert np.isnan(relaxed["ECC_upper_limit"])


def test_summary_rejects_unpaired_samples():
    with pytest.raises(ValueError, match="paired"):
        eccentricity_summary(np.zeros(10), np.zeros(11))


def _results_row(eps1_local, eps2_local, initial, factor, extra_column=None):
    """A minimal result table carrying what the unscaling needs.

    Columns are written in an order that does *not* match the chain, since
    ``split_output_results`` reorders them in real runs.
    """
    row = {}
    if extra_column is not None:
        for perc in (16, 50, 84):
            row[f"dF0_{perc}"] = np.percentile(extra_column, perc)
    for name, local in (("EPS2", eps2_local), ("EPS1", eps1_local)):
        for perc in (16, 50, 84):
            row[f"d{name}_{perc}"] = np.percentile(local, perc)
        row[f"d{name}_initial"] = initial[name]
        row[f"d{name}_factor"] = factor[name]
    return Table(rows=[row])[0]


def test_chain_columns_are_identified_by_their_recorded_percentiles():
    """Undo the local coordinates without relying on the table's column order."""
    rng = np.random.default_rng(SEED)
    eps1_local = rng.normal(0.3, 1.0, 5000)
    eps2_local = rng.normal(-0.8, 1.0, 5000)
    freq_local = rng.normal(4.0, 1.0, 5000)
    # Chain order: F0, EPS1, EPS2 -- the table above lists EPS2 first.
    chain = np.column_stack([freq_local, eps1_local, eps2_local])

    initial = {"EPS1": 1e-3, "EPS2": -2e-3}
    factor = {"EPS1": 1e-5, "EPS2": 2e-5}
    row = _results_row(eps1_local, eps2_local, initial, factor, extra_column=freq_local)

    eps1, eps2 = eps_samples_from_chain(row, chain)

    assert np.allclose(eps1, initial["EPS1"] + eps1_local * factor["EPS1"])
    assert np.allclose(eps2, initial["EPS2"] + eps2_local * factor["EPS2"])


def test_unfitted_eccentricity_raises_a_clear_error():
    rng = np.random.default_rng(SEED)
    chain = rng.normal(size=(1000, 1))
    row = Table(rows=[{"dF0_16": -1.0, "dF0_50": 0.0, "dF0_84": 1.0}])[0]

    with pytest.raises(ValueError, match="EPS1 was not a fitted parameter"):
        eps_samples_from_chain(row, chain)


def test_mismatched_chain_and_table_are_caught():
    """Percentiles that match nothing mean the two files are from different fits."""
    rng = np.random.default_rng(SEED)
    chain = rng.normal(size=(5000, 2))
    row = _results_row(
        rng.normal(50.0, 1.0, 5000),
        rng.normal(-70.0, 1.0, 5000),
        {"EPS1": 0.0, "EPS2": 0.0},
        {"EPS1": 1.0, "EPS2": 1.0},
    )

    with pytest.raises(ValueError, match="No column of the chain matches"):
        eps_samples_from_chain(row, chain)


def test_default_chain_file_falls_back_to_the_emcee_backend():
    """With no sample file beside the table, the HDF5 chain is the fallback."""
    assert default_chain_file("out_A1_EPS1_EPS2_results.ecsv") == "out_A1_EPS1_EPS2.h5"
    with pytest.raises(ValueError, match="does not end in"):
        default_chain_file("out.ecsv")


def test_default_chain_file_prefers_the_saved_samples(tmp_path):
    root = str(tmp_path / "out_EPS1_EPS2")
    save_flat_samples(root, np.zeros((10, 2)), ["dEPS1", "dEPS2"])

    assert default_chain_file(root + RESULTS_SUFFIX) == root + SAMPLES_SUFFIX


def test_saved_samples_round_trip(tmp_path):
    """Every sampler writes this file; it is the one route to the samples."""
    rng = np.random.default_rng(SEED)
    samples = rng.normal(size=(500, 3))
    labels = ["dF0", "dEPS1", "dEPS2"]

    fname = save_flat_samples(str(tmp_path / "run"), samples, labels)
    read_samples, read_labels = load_flat_samples(fname)

    assert fname.endswith(SAMPLES_SUFFIX)
    assert np.allclose(read_samples, samples)
    assert read_labels == labels


def test_saved_labels_are_used_instead_of_the_percentile_fingerprint(tmp_path):
    """A table with no recorded percentiles is enough when the names are saved.

    Dropping ``dEPS1_16/50/84`` from the table breaks the fingerprint route
    outright, so a run that still succeeds can only have used the labels.
    """
    rng = np.random.default_rng(SEED)
    chain = rng.normal(size=(2000, 3))
    initial = {"EPS1": 1e-3, "EPS2": -2e-3}
    factor = {"EPS1": 1e-5, "EPS2": 2e-5}

    root = str(tmp_path / "out_EPS1_EPS2")
    save_flat_samples(root, chain, ["dF0", "dEPS1", "dEPS2"])
    row = {
        f"d{par}_{field}": value[par]
        for par in ("EPS1", "EPS2")
        for field, value in (("initial", initial), ("factor", factor))
    }
    Table(rows=[row]).write(root + RESULTS_SUFFIX)

    eps1, eps2 = load_eps_samples(root + RESULTS_SUFFIX)

    assert np.allclose(eps1, initial["EPS1"] + chain[:, 1] * factor["EPS1"])
    assert np.allclose(eps2, initial["EPS2"] + chain[:, 2] * factor["EPS2"])


def test_unsampled_eccentricity_raises_even_with_labels(tmp_path):
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="EPS1 is not among the sampled parameters"):
        eps_samples_from_chain({}, rng.normal(size=(100, 2)), labels=["dF0", "dA1"])


def test_labels_win_over_a_disagreeing_fingerprint_but_warn(caplog):
    """A chain extended after the table was written must still load."""
    rng = np.random.default_rng(SEED)
    chain = rng.normal(size=(2000, 2))
    row = {
        "dEPS1_16": 100.0,
        "dEPS1_50": 101.0,
        "dEPS1_84": 102.0,
        "dEPS1_initial": 0.0,
        "dEPS1_factor": 1.0,
        "dEPS2_16": -1.0,
        "dEPS2_50": 0.0,
        "dEPS2_84": 1.0,
        "dEPS2_initial": 0.0,
        "dEPS2_factor": 1.0,
    }

    with caplog.at_level("WARNING"):
        eps1, _ = eps_samples_from_chain(row, chain, labels=["dEPS1", "dEPS2"])

    assert np.allclose(eps1, chain[:, 0])
    assert "disagree with the percentiles" in caplog.text


def _write_run(tmp_path):
    """Write an emcee chain and its result table the way a real fit does."""
    import emcee

    root = str(tmp_path / "out_EPS1_EPS2")
    initial = {"EPS1": 1e-3, "EPS2": -2e-3}
    factor = {"EPS1": 1e-5, "EPS2": 2e-5}

    # Two well-separated Gaussians, so the two chain columns cannot be confused.
    centres, widths = np.array([0.3, -6.0]), np.array([1.0, 2.0])

    def log_prob(pars):
        return -0.5 * np.sum(((pars - centres) / widths) ** 2)

    backend = emcee.backends.HDFBackend(root + ".h5")
    backend.reset(16, 2)
    rng = np.random.default_rng(SEED)
    start = centres + rng.normal(size=(16, 2))
    sampler = emcee.EnsembleSampler(16, 2, log_prob, backend=backend)
    sampler.run_mcmc(start, 2000, progress=False)

    from ..mcmc_utils import get_flat_samples

    flat_chain, _ = get_flat_samples(backend)
    row = _results_row(flat_chain[:, 0], flat_chain[:, 1], initial, factor)
    Table(rows=[dict(zip(row.colnames, row))]).write(root + "_results.ecsv")

    return root, flat_chain, initial, factor


def test_load_eps_samples_round_trip(tmp_path):
    """The loader finds the chain beside the table and undoes the scaling."""
    root, flat_chain, initial, factor = _write_run(tmp_path)

    eps1, eps2 = load_eps_samples(root + "_results.ecsv")

    assert np.allclose(eps1, initial["EPS1"] + flat_chain[:, 0] * factor["EPS1"])
    assert np.allclose(eps2, initial["EPS2"] + flat_chain[:, 1] * factor["EPS2"])

    summary = eccentricity_summary_from_run(root + "_results.ecsv")
    assert summary["ECC_nsamples"] == flat_chain.shape[0]
    assert summary["ECC_50"] > 0


def test_plot_marks_the_limit_when_undetected(tmp_path):
    eps1, eps2 = _eps_samples(ecc=0.0, size=20_000)
    fname = str(tmp_path / "ecc.jpg")

    plot_eccentricity_posterior(eps1, eps2, fname=fname)

    assert (tmp_path / "ecc.jpg").exists()


def test_plot_marks_the_interval_when_detected(tmp_path):
    eps1, eps2 = _eps_samples(ecc=20 * SIGMA, omega_deg=71.0, size=20_000)
    fname = str(tmp_path / "ecc_detected.jpg")

    plot_eccentricity_posterior(eps1, eps2, fname=fname)

    assert (tmp_path / "ecc_detected.jpg").exists()


def test_the_panel_can_be_drawn_into_an_axis_the_caller_owns(tmp_path):
    """Same panel, someone else's figure: nothing is created and nothing saved."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps1, eps2 = _eps_samples(ecc=20 * SIGMA, omega_deg=71.0, size=20_000)
    fig, ax = plt.subplots()

    summary = draw_eccentricity_posterior(ax, eps1, eps2)

    assert summary["ECC_detected"]
    assert ax.get_xlabel() == "Eccentricity"
    # The median line and the 68% span, on top of the histogram.
    assert len(ax.lines) == 1
    assert fig.axes == [ax]
    assert not list(tmp_path.iterdir())
    plt.close(fig)


def test_the_panel_reuses_a_summary_it_is_handed():
    """No recomputation, so the panel and the table can never disagree."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps1, eps2 = _eps_samples(ecc=0.0, size=20_000)
    summary = eccentricity_summary(eps1, eps2, upper_limit_level=0.99)
    fig, ax = plt.subplots()

    returned = draw_eccentricity_posterior(ax, eps1, eps2, summary=summary)

    assert returned is summary
    assert "99% upper limit" in ax.get_legend().get_texts()[-1].get_text()
    plt.close(fig)


# --- Pipeline integration: _enrich_results_with_eccentricity ----------------


def _mock_results_and_samples(tmp_path, ecc=0.0, omega_deg=0.0, size=20_000):
    """Build a results dict and samples file for testing pipeline enrichment."""
    rng = np.random.default_rng(SEED)
    initial = {"EPS1": 1e-3, "EPS2": -2e-3}
    factor = {"EPS1": 1e-5, "EPS2": 2e-5}
    omega = np.radians(omega_deg)
    centre_eps1 = (ecc * np.sin(omega) - initial["EPS1"]) / factor["EPS1"]
    centre_eps2 = (ecc * np.cos(omega) - initial["EPS2"]) / factor["EPS2"]

    eps1_local = rng.normal(centre_eps1, SIGMA / factor["EPS1"], size)
    eps2_local = rng.normal(centre_eps2, SIGMA / factor["EPS2"], size)
    flat_chain = np.column_stack([eps1_local, eps2_local])
    labels = ["dEPS1", "dEPS2"]

    outroot = str(tmp_path / "run")
    save_flat_samples(outroot, flat_chain, labels)

    results = {}
    for name, local in (("EPS1", eps1_local), ("EPS2", eps2_local)):
        for perc in (16, 50, 84):
            results[f"d{name}_{perc}"] = float(np.percentile(local, perc))
        results[f"d{name}_initial"] = initial[name]
        results[f"d{name}_factor"] = factor[name]

    return results, outroot


def test_pipeline_enrichment_adds_eccentricity_columns(tmp_path):
    """ECC_* columns are added when EPS1 and EPS2 are both fitted."""
    results, outroot = _mock_results_and_samples(tmp_path, ecc=20 * SIGMA, omega_deg=71.0)
    enriched = _enrich_results_with_eccentricity(results, outroot, ["A1", "EPS1", "EPS2", "F0"])

    assert "ECC_50" in enriched
    assert "ECC_detected" in enriched
    assert "ECC_summary" in enriched
    assert enriched["ECC_detected"]
    assert enriched["ECC_50"] > 0
    assert (tmp_path / "run_eccentricity.jpg").exists()


def test_pipeline_enrichment_skips_without_both_eps(tmp_path):
    """No ECC_* columns when only EPS1 (or neither) is fitted."""
    results, outroot = _mock_results_and_samples(tmp_path)
    enriched = _enrich_results_with_eccentricity(results, outroot, ["A1", "EPS1", "F0"])
    assert "ECC_50" not in enriched

    enriched = _enrich_results_with_eccentricity(results, outroot, ["A1", "F0"])
    assert "ECC_50" not in enriched


def test_pipeline_enrichment_upper_limit_for_null(tmp_path):
    """With no injected eccentricity, the enrichment reports an upper limit."""
    results, outroot = _mock_results_and_samples(tmp_path, ecc=0.0)
    enriched = _enrich_results_with_eccentricity(results, outroot, ["EPS1", "EPS2"])

    assert not enriched["ECC_detected"]
    assert np.isfinite(enriched["ECC_upper_limit"])


def _mock_orbital_run(tmp_path, ecc=20 * SIGMA, omega_deg=71.0, size=20_000):
    """A run that varied A1 and PB as well as the eccentricity pair."""
    results, outroot = _mock_results_and_samples(tmp_path, ecc=ecc, omega_deg=omega_deg, size=size)
    eps_chain, labels = load_flat_samples(outroot + SAMPLES_SUFFIX)

    rng = np.random.default_rng(SEED + 1)
    extra = {"A1": (22.225, 1e-6), "PB": (218668.4, 1e-3)}
    columns, names = [], []
    for name, (initial, factor) in extra.items():
        local = rng.normal(0.0, 300.0, eps_chain.shape[0])
        columns.append(local)
        names.append("d" + name)
        for perc in (16, 50, 84):
            results[f"d{name}_{perc}"] = float(np.percentile(local, perc))
        results[f"d{name}_initial"] = initial
        results[f"d{name}_factor"] = factor

    save_flat_samples(outroot, np.column_stack(columns + [eps_chain]), names + list(labels))
    return results, outroot


def test_pipeline_enrichment_draws_the_orbit_summary(tmp_path):
    """The eccentricity hook is where the orbit summary is written too."""
    results, outroot = _mock_results_and_samples(tmp_path, ecc=20 * SIGMA, omega_deg=71.0)

    _enrich_results_with_eccentricity(results, outroot, ["EPS1", "EPS2"])

    assert (tmp_path / "run_orbit.jpg").exists()


def test_the_orbit_summary_gets_every_orbital_parameter_that_was_fitted(tmp_path, monkeypatch):
    """Not just the EPS pair: A1 and PB were sampled, so they get panels too."""
    from matplotlib.figure import Figure

    # The hook saves the standalone eccentricity plot first and the summary
    # second, so it is the last figure that is the one to look at.
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: saved.append(self))

    results, outroot = _mock_orbital_run(tmp_path)
    _enrich_results_with_eccentricity(results, outroot, ["A1", "PB", "EPS1", "EPS2"])

    # A1, PB, EPS1, EPS2 -> a 4x4 corner block, plus the eccentricity panel.
    assert len(saved) == 2
    assert len(saved[-1].axes) == 4 * 4 + 1


# --- CLI entry point --------------------------------------------------------


def test_ell1ecc_cli_prints_summary(tmp_path, capsys):
    """The CLI prints the eccentricity summary and writes a plot."""
    from ..eccentricity import main as ell1ecc_main

    results, outroot = _mock_results_and_samples(tmp_path, ecc=20 * SIGMA, omega_deg=71.0)
    results_file = outroot + "_results.ecsv"
    Table(rows=[results]).write(results_file)

    plot_path = str(tmp_path / "ecc_cli.jpg")
    orbit_path = str(tmp_path / "orbit_cli.jpg")
    ell1ecc_main([results_file, "--plot", plot_path, "--orbit-plot", orbit_path])

    captured = capsys.readouterr()
    assert "e =" in captured.out or "e <" in captured.out
    assert (tmp_path / "ecc_cli.jpg").exists()
    assert (tmp_path / "orbit_cli.jpg").exists()


def test_ell1ecc_cli_names_both_plots_after_the_output_root(tmp_path, capsys):
    """With no explicit paths, both figures land beside the result table."""
    from ..eccentricity import main as ell1ecc_main

    results, outroot = _mock_orbital_run(tmp_path)
    results_file = outroot + "_results.ecsv"
    Table(rows=[results]).write(results_file)

    ell1ecc_main([results_file])

    assert (tmp_path / "run_eccentricity.jpg").exists()
    assert (tmp_path / "run_orbit.jpg").exists()
