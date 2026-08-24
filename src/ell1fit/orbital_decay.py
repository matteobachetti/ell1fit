"""``ell1decay``: measure PBDOT (and test for PBDDOT) from a set of per-epoch
``ell1fit`` TASC results, replacing a standalone research script that did the
same thing with ``emcee`` and an externally-maintained reference ``.par``.

Two models are always fit and compared, never one in isolation: M0 is a
quadratic ``delta_tasc(t)`` (offset, linear drift, PBDOT), M1 adds a cubic
term (PBDDOT). Their evidences (via nested sampling) give a Bayes factor for
whether the data need the cubic term at all -- that comparison, not just the
PBDOT point estimate, is the reason this command exists.
"""

import argparse
import copy
import json
import logging
import os

import astropy.units as u
import numpy as np

from .logging import configure_logging
from .mcmc_utils import plot_mcmc_comparison, plot_mcmc_results
from .orbital_decay_data import (
    OrbitalModelCompatibilityError,
    build_reference_model,
    check_compatibility,
    load_epochs,
)
from .orbital_decay_model import delta_tasc_model, log_likelihood_asymmetric_errors, physical_from_beta
from .orbital_decay_sampling import bayes_factor, default_bounds, laplace_cross_check, run_seed_scatter


__all__ = ["fit_orbital_decay", "main"]


def _assemble_data(epochs, ref_model):
    """Build ``(x, y, yerrn, yerrp)`` from the epochs and a ``PBDOT = 0`` reference.

    ``y`` is each epoch's fitted TASC minus the *nearest* ascending-node time
    the reference model predicts (``ref_TASC + n*ref_PB`` for the integer
    ``n`` closest to that epoch's own TASC) -- TASC is only ever defined
    modulo PB, so without this wrap ``y`` would jump by whole multiples of PB
    between epochs instead of tracking the genuine deviation. ``x`` is time
    since the reference model's own ``PEPOCH`` (which :func:`build_reference_model`
    sets to the reference epoch used), measured at each epoch's own fitted
    TASC rather than its (looser, mid-observation) ``PEPOCH``.
    """
    ref_tasc = float(ref_model.TASC.value)
    ref_pb = float(ref_model.PB.value)
    reference_epoch = float(ref_model.PEPOCH.value)

    x, y, yerrn, yerrp = [], [], [], []
    for epoch in epochs:
        n_orbits = round((epoch.tasc - ref_tasc) / ref_pb)
        predicted_tasc = ref_tasc + n_orbits * ref_pb
        x.append(epoch.tasc - reference_epoch)
        y.append((epoch.tasc - predicted_tasc) * 86400.0)
        yerrn.append(epoch.tasc_err[0] * 86400.0)
        yerrp.append(epoch.tasc_err[1] * 86400.0)

    return np.array(x), np.array(y), np.array(yerrn), np.array(yerrp)


def _fit_model(order, x, y, yerrn, yerrp, baseline_days, labels, nlive, dlogz, seeds, outroot):
    def loglikelihood(beta):
        return log_likelihood_asymmetric_errors(beta, x, y, yerrn, yerrp, baseline_days)

    bounds = default_bounds(y, order)
    result = run_seed_scatter(
        loglikelihood, bounds, labels, n_seeds=seeds, nlive=nlive, dlogz=dlogz, outroot=outroot
    )
    result["laplace_log_evidence"] = laplace_cross_check(loglikelihood, bounds, result["map_position"])
    result["loglikelihood"] = loglikelihood
    return result


def _write_diagnostic_plot(x, y, yerrn, yerrp, baseline_days, m0_result, m1_result, fname):
    """delta_tasc(t) data with both models' median curves and posterior-draw
    fans overlaid, distinguishably colored, with a residual panel.

    Uses ``constrained_layout`` rather than this package's other plots' hand-
    tuned ``figure.subplot.*`` margins (see :mod:`ell1fit.plotting`): those
    were tuned for one paper's fixed-size single-model corner plots, and
    break under this plot's now-required legend and two-model overlay.
    """
    import matplotlib.pyplot as plt

    x_smooth = np.linspace(x.min(), x.max(), 400)

    fig, (ax_data, ax_resid) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 5.5), height_ratios=[3, 1], constrained_layout=True
    )

    ax_data.errorbar(
        x, y, yerr=[yerrn, yerrp], fmt="o", color="black", ms=4, capsize=2, label="data", zorder=5
    )

    models = [("M0 (PBDOT)", m0_result, "C0", "-"), ("M1 (PBDOT+PBDDOT)", m1_result, "C1", "--")]
    for name, result, color, linestyle in models:
        draws = result["flat_samples"]
        n_draws = min(200, draws.shape[0])
        draw_idx = np.random.default_rng(0).choice(draws.shape[0], n_draws, replace=False)
        for i in draw_idx:
            ax_data.plot(
                x_smooth,
                delta_tasc_model(draws[i], x_smooth, baseline_days),
                color=color,
                alpha=0.02,
                zorder=1,
            )
        beta_median = np.median(draws, axis=0)
        ax_data.plot(
            x_smooth,
            delta_tasc_model(beta_median, x_smooth, baseline_days),
            color=color,
            linestyle=linestyle,
            label=name,
            zorder=4,
        )
        residual = y - delta_tasc_model(beta_median, x, baseline_days)
        ax_resid.errorbar(
            x, residual, yerr=[yerrn, yerrp], fmt="o", color=color, ms=3, capsize=2, alpha=0.7
        )

    ax_resid.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax_data.set_ylabel(r"$\Delta$TASC (s)")
    ax_resid.set_ylabel("residual (s)")
    ax_resid.set_xlabel("days since reference epoch")
    ax_data.legend(loc="best", frameon=False)
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def _write_parfile(ref_model, m0_result, baseline_days, pb0_days, outroot):
    """Write ``{outroot}.par`` with M0's median PBDOT/PB/TASC and an uncertainty
    equal to the larger of its two one-sigma half-widths, matching
    :func:`ell1fit.create_parfile.update_model`'s parfile convention (a
    ``.par`` file can only hold one symmetric uncertainty).

    M1 is never adopted here, regardless of the Bayes factor -- it is
    reported in the JSON output, not written into the ephemeris.
    """
    beta_16, beta_50, beta_84 = np.percentile(m0_result["flat_samples"], [16, 50, 84], axis=0)
    phys_16 = physical_from_beta(beta_16, baseline_days, pb0_days)
    phys_50 = physical_from_beta(beta_50, baseline_days, pb0_days)
    phys_84 = physical_from_beta(beta_84, baseline_days, pb0_days)

    def sym_err(key):
        return max(abs(phys_50[key] - phys_16[key]), abs(phys_84[key] - phys_50[key]))

    model = copy.deepcopy(ref_model)
    model.TASC.value = model.TASC.value + phys_50["tasc_offset_sec"] / 86400.0
    model.TASC.uncertainty_value = sym_err("tasc_offset_sec") / 86400.0
    model.PB.value = model.PB.value + phys_50["pb_offset_sec"] / 86400.0
    model.PB.uncertainty_value = sym_err("pb_offset_sec") / 86400.0
    # PBDOT.value silently multiplies by 1e-12 whenever the magnitude passed
    # exceeds 1e-7 (PINT's "PBDOT 7.2 means 7.2e-12" parfile convention,
    # pint.models.parameter.floatParameter._set_quantity) -- setting .quantity
    # with an explicit dimensionless unit bypasses that magnitude-guessing
    # heuristic instead of relying on our fitted value happening to stay
    # under the threshold.
    model.PBDOT.quantity = phys_50["PBDOT"] * u.dimensionless_unscaled
    model.PBDOT.uncertainty = sym_err("PBDOT") * u.dimensionless_unscaled

    fname = outroot + ".par"
    with open(fname, "w") as fobj:
        fobj.write(model.as_parfile())
    return fname


def fit_orbital_decay(
    files,
    outroot="orbital_decay",
    nlive=500,
    dlogz=0.1,
    seeds=3,
    compat_tolerance=1e-9,
    pbdot_impact_fraction=1.0,
    reference_epoch=None,
    write_parfile=True,
):
    """Load, validate, fit M0 and M1, and write every output artifact.

    Returns
    -------
    dict
        Also written to ``{outroot}_results.json``.
    """
    epochs = load_epochs(files)
    check_compatibility(epochs, tolerance=compat_tolerance, pbdot_impact_fraction=pbdot_impact_fraction)
    ref_model = build_reference_model(epochs, reference_epoch=reference_epoch)

    x, y, yerrn, yerrp = _assemble_data(epochs, ref_model)
    baseline_days = float(x.max() - x.min())
    pb0_days = float(ref_model.PB.value)

    m0_result = _fit_model(
        2, x, y, yerrn, yerrp, baseline_days, ["b0", "b1", "b2"], nlive, dlogz, seeds, outroot + "_m0"
    )
    m1_result = _fit_model(
        3,
        x,
        y,
        yerrn,
        yerrp,
        baseline_days,
        ["b0", "b1", "b2", "b3"],
        nlive,
        dlogz,
        seeds,
        outroot + "_m1",
    )

    bf = bayes_factor(m0_result, m1_result)

    plot_mcmc_comparison(
        [m0_result["flat_samples"], m1_result["flat_samples"]],
        [["b0", "b1", "b2"], ["b0", "b1", "b2", "b3"]],
        ["M0", "M1"],
        outroot + "_comparison.jpg",
    )
    _write_diagnostic_plot(x, y, yerrn, yerrp, baseline_days, m0_result, m1_result, outroot + "_data.jpg")

    beta_16_0, beta_50_0, beta_84_0 = np.percentile(m0_result["flat_samples"], [16, 50, 84], axis=0)
    beta_16_1, beta_50_1, beta_84_1 = np.percentile(m1_result["flat_samples"], [16, 50, 84], axis=0)
    phys_m0 = physical_from_beta(beta_50_0, baseline_days, pb0_days)
    phys_m1 = physical_from_beta(beta_50_1, baseline_days, pb0_days)

    def phys_err(order, beta_16, beta_50, beta_84, key):
        lo = physical_from_beta(beta_16, baseline_days, pb0_days)[key]
        mid = physical_from_beta(beta_50, baseline_days, pb0_days)[key]
        hi = physical_from_beta(beta_84, baseline_days, pb0_days)[key]
        return {"neg": mid - lo, "pos": hi - mid}

    results = {
        "n_epochs": len(epochs),
        "baseline_days": baseline_days,
        "reference_epoch": float(ref_model.PEPOCH.value),
        "PB0_days": pb0_days,
        "M0": {
            "PBDOT": phys_m0["PBDOT"],
            "PBDOT_err": phys_err(2, beta_16_0, beta_50_0, beta_84_0, "PBDOT"),
            "log_evidence": m0_result["log_evidence"],
            "log_evidence_err": m0_result["log_evidence_err"],
            "laplace_log_evidence": m0_result["laplace_log_evidence"],
            "peak_shortfall": m0_result["peak_shortfall"],
            "converged": m0_result["converged"],
        },
        "M1": {
            "PBDOT": phys_m1["PBDOT"],
            "PBDOT_err": phys_err(3, beta_16_1, beta_50_1, beta_84_1, "PBDOT"),
            "PBDDOT_per_yr": phys_m1["PBDDOT"],
            "PBDDOT_per_yr_err": phys_err(3, beta_16_1, beta_50_1, beta_84_1, "PBDDOT"),
            "log_evidence": m1_result["log_evidence"],
            "log_evidence_err": m1_result["log_evidence_err"],
            "laplace_log_evidence": m1_result["laplace_log_evidence"],
            "peak_shortfall": m1_result["peak_shortfall"],
            "converged": m1_result["converged"],
        },
        "bayes_factor": bf,
    }

    with open(outroot + "_results.json", "w") as fobj:
        json.dump(results, fobj, indent=2)

    if write_parfile:
        results["parfile"] = _write_parfile(ref_model, m0_result, baseline_days, pb0_days, outroot)

    logging.info(f"M0 PBDOT = {phys_m0['PBDOT']:.4e}, ln BF (M1/M0) = {bf['ln_bf']:.2f} +- {bf['ln_bf_err']:.2f} ({bf['interpretation']})")

    return results


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Fit PBDOT (and test for PBDDOT) from a set of ell1fit per-epoch TASC results."
    )
    parser.add_argument("files", nargs="+", help="ell1fit .ecsv result files, one per epoch")
    parser.add_argument("-o", "--outroot", default="orbital_decay", help="Output file root")
    parser.add_argument("--nlive", type=int, default=500, help="Nested sampling live points")
    parser.add_argument("--dlogz", type=float, default=0.1, help="Nested sampling stopping criterion")
    parser.add_argument("--seeds", type=int, default=3, help="Nested sampling repeats per model")
    parser.add_argument(
        "--compat-tolerance",
        type=float,
        default=1e-9,
        dest="compat_tolerance",
        help="Relative tolerance for the cross-file orbital-model compatibility check",
    )
    parser.add_argument(
        "--pbdot-impact-fraction",
        type=float,
        default=1.0,
        dest="pbdot_impact_fraction",
        help=(
            "Abort threshold for a file-to-file PBDOT disagreement, as a fraction of that "
            "epoch's own TASC uncertainty its spurious-delta_tasc impact would have to reach "
            "(see spurious_tasc_from_pbdot_mismatch). Below this fraction it is only a warning."
        ),
    )
    parser.add_argument(
        "--reference-epoch",
        type=float,
        default=None,
        dest="reference_epoch",
        help="MJD to reference the model at (default: mean PEPOCH across input files)",
    )
    parser.add_argument(
        "--no-parfile", action="store_false", dest="write_parfile", help="Do not write {outroot}.par"
    )
    parsed = parser.parse_args(args)

    configure_logging()

    try:
        fit_orbital_decay(
            parsed.files,
            outroot=parsed.outroot,
            nlive=parsed.nlive,
            dlogz=parsed.dlogz,
            seeds=parsed.seeds,
            compat_tolerance=parsed.compat_tolerance,
            pbdot_impact_fraction=parsed.pbdot_impact_fraction,
            reference_epoch=parsed.reference_epoch,
            write_parfile=parsed.write_parfile,
        )
    except OrbitalModelCompatibilityError as exc:
        logging.error(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
