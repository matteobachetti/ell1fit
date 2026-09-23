"""``ell1stat``: rough statistics of a pulsar's spin history across epochs.

Takes the same per-epoch ``ell1fit`` results as ``ell1decay`` (plus optional
extra tables, see :mod:`ell1fit.spin_stats_data`) and reports

* the **secular** spin trend: a polynomial fit of ``F0`` against time, whose
  slope is the long-term ``F1``;
* the **local** ``F1`` values measured within each epoch: their mean, their
  scatter, and the fraction of epochs spinning up;
* optionally, a search for a coherent periodicity in local ``F1`` and in the
  pulsed count rate within a given period range (see
  :mod:`ell1fit.spin_periodicity`).

In an accreting pulsar the two need not agree: local ``F1`` follows the
instantaneous accretion torque, while the secular trend integrates every
torque episode, including the ones no observation caught.

Rough by design: fitting with extra scatter
-------------------------------------------
Accretion-torque noise makes ``F0`` wander far beyond its measurement errors,
so a plain weighted fit would return an absurd chi-squared and error bars far
too small. Every fit here instead adds one "intrinsic scatter" term in
quadrature to the errors, chosen so that chi-squared per degree of freedom is
exactly 1 (:func:`fit_polynomial_with_scatter`). When the scatter dominates,
the fit becomes effectively unweighted, so one very precise epoch cannot pin
the result; when the data agree with their errors, nothing is added and it is
an ordinary weighted fit. It does not model the *correlation* of torque noise
between nearby epochs, so its error bars are indicative, not rigorous.
"""

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .logging import configure_logging
from .plotting import add_figure_format_argument, set_figure_format
from .spin_periodicity import rate_values, search_periodicity
from .spin_stats_data import load_spin_table

__all__ = [
    "ScatterFit",
    "fit_polynomial_with_scatter",
    "local_f1_statistics",
    "main",
    "run_periodicity",
    "secular_spin",
    "spin_statistics",
]

DAY = 86400.0


@dataclass
class ScatterFit:
    """A polynomial fit with intrinsic scatter; see :func:`fit_polynomial_with_scatter`."""

    coeffs: np.ndarray  #: ascending order: y = c0 + c1 x + c2 x^2 ...
    cov: np.ndarray  #: covariance of ``coeffs``, with the scatter included
    scatter: float  #: intrinsic scatter added in quadrature, in units of y
    chi2: float  #: chi-squared with the scatter included (= dof unless scatter is 0)
    chi2_formal: float  #: chi-squared with the measurement errors alone
    dof: int

    @property
    def errors(self):
        return np.sqrt(np.diag(self.cov))

    def __call__(self, x):
        return np.polynomial.polynomial.polyval(x, self.coeffs)


def _weighted_polyfit(x, y, var, degree):
    design = np.vander(x, degree + 1, increasing=True)
    weights = 1 / var
    cov = np.linalg.inv(design.T @ (design * weights[:, None]))
    coeffs = cov @ design.T @ (weights * y)
    chi2 = np.sum((y - design @ coeffs) ** 2 * weights)
    return coeffs, cov, chi2


def fit_polynomial_with_scatter(x, y, err, degree=1):
    """Weighted polynomial fit with an intrinsic scatter ``s`` added to ``err`` in quadrature.

    ``s`` is the smallest value that makes chi-squared equal the number of
    degrees of freedom, or 0 if the data already scatter less than their
    errors. A degree-0 fit is a weighted mean.

    Parameters
    ----------
    x, y, err : array-like
        Abscissae, values and 1-sigma errors. ``x`` should be centred near 0
        for a well-conditioned fit at higher degree.
    degree : int

    Returns
    -------
    ScatterFit
    """
    x, y, err = (np.asarray(a, dtype=float) for a in (x, y, err))
    dof = x.size - (degree + 1)
    if dof < 0:
        raise ValueError(f"A degree-{degree} fit needs at least {degree + 1} points, got {x.size}")

    _, _, chi2_formal = _weighted_polyfit(x, y, err**2, degree)
    scatter = 0.0
    if dof > 0 and chi2_formal > dof:

        def excess(s):
            return _weighted_polyfit(x, y, err**2 + s**2, degree)[2] - dof

        upper = np.std(y) + np.max(err)
        while excess(upper) > 0:
            upper *= 2
        scatter = brentq(excess, 0.0, upper, rtol=1e-10)
    elif dof == 0:
        logging.warning(f"Degree-{degree} fit to {x.size} points: no scatter can be estimated")

    coeffs, cov, chi2 = _weighted_polyfit(x, y, err**2 + scatter**2, degree)
    return ScatterFit(coeffs, cov, scatter, chi2, chi2_formal, dof)


def _symmetric_error(table, par):
    return 0.5 * (table[f"{par}_err_neg"] + table[f"{par}_err_pos"])


def secular_spin(table, reference_mjd, degree=1):
    """Fit F0(t) with a polynomial plus intrinsic scatter.

    Returns
    -------
    fit : ScatterFit
        With ``x`` in seconds from ``reference_mjd``.
    summary : dict
        ``F0``, ``F1`` (and ``F2`` for degree 2...) at ``reference_mjd`` as
        ``[value, error]`` in Hz, Hz/s, Hz/s^2; the scatter in Hz; the formal
        chi-squared and the degrees of freedom.
    """
    x = (np.asarray(table["mjd"]) - reference_mjd) * DAY
    fit = fit_polynomial_with_scatter(x, table["f0"], _symmetric_error(table, "f0"), degree)
    summary = {"degree": degree, "reference_mjd": reference_mjd}
    factorial = 1.0
    for order, (value, error) in enumerate(zip(fit.coeffs, fit.errors)):
        factorial *= max(order, 1)
        summary[f"F{order}"] = [float(value * factorial), float(error * factorial)]
    summary.update(scatter_hz=fit.scatter, chi2_formal=fit.chi2_formal, dof=fit.dof)
    return fit, summary


def local_f1_statistics(table):
    """Mean and scatter of the F1 values measured within each epoch.

    Epochs with no measured F1 (NaN) are skipped. ``mean`` includes the
    intrinsic scatter in its weights (see the module docstring);
    ``formal_weighted_mean`` is the plain inverse-variance mean, whose
    ``formal_chi2`` against ``dof`` says how far the local values exceed their errors.
    """
    good = np.isfinite(np.asarray(table["f1"], dtype=float))
    f1 = np.asarray(table["f1"][good], dtype=float)
    err = np.asarray(_symmetric_error(table[good], "f1"), dtype=float)
    if f1.size == 0:
        return {"n": 0}
    fit = fit_polynomial_with_scatter(np.zeros_like(f1), f1, err, degree=0)
    weights = 1 / err**2
    formal_mean = np.sum(weights * f1) / weights.sum()
    return {
        "n": int(f1.size),
        "mean": [float(fit.coeffs[0]), float(fit.errors[0])],
        "intrinsic_scatter": float(fit.scatter),
        "formal_weighted_mean": [float(formal_mean), float(weights.sum() ** -0.5)],
        "formal_chi2": float(fit.chi2_formal),
        "dof": int(fit.dof),
        "unweighted_mean": float(np.mean(f1)),
        "std": float(np.std(f1, ddof=1)) if f1.size > 1 else float("nan"),
        "fraction_spin_up": float(np.mean(f1 > 0)),
    }


def _power_of_ten(values):
    """Exponent to divide ``values`` by for readable axis labels (e.g. -11 for ~3e-11)."""
    values = np.abs(np.asarray(values, dtype=float))
    values = values[np.isfinite(values) & (values > 0)]
    return int(np.floor(np.log10(values.max()))) if values.size else 0


def _errorbars_by_label(ax, table, x, y, yerr_neg, yerr_pos, scale=1.0):
    """One errorbar series per ``label``, so points from different sources are told apart."""
    from .plotting import DATA_COLOR

    labels = list(dict.fromkeys(table["label"]))
    markers = "osD^v<>"
    for i, label in enumerate(labels):
        sel = np.asarray(table["label"] == label)
        ax.errorbar(
            x[sel],
            y[sel] / scale,
            yerr=[yerr_neg[sel] / scale, yerr_pos[sel] / scale],
            fmt=markers[i % len(markers)],
            color=DATA_COLOR if len(labels) == 1 else f"C{i}",
            ms=3,
            label=label,
            zorder=5,
        )
    if len(labels) > 1:
        ax.legend(loc="best")


def _plot_trend(table, fit, reference_mjd, fname, tangent_fraction=0.02):
    """F0(t) with the secular fit, over the residuals with each epoch's local F1
    drawn as a short segment: its slope against the secular one at a glance."""
    import matplotlib.pyplot as plt

    from .plotting import GUIDE_COLOR, figure_size, plot_style_context, save_figure

    mjd = np.asarray(table["mjd"], dtype=float)
    x = (mjd - reference_mjd) * DAY
    f0 = np.asarray(table["f0"], dtype=float)
    err_neg = np.asarray(table["f0_err_neg"], dtype=float)
    err_pos = np.asarray(table["f0_err_pos"], dtype=float)
    residual = f0 - fit(x)
    exponent = _power_of_ten(residual)
    scale = 10.0**exponent

    with plot_style_context():
        fig, (ax_data, ax_resid) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=figure_size("wide-tall"),
            height_ratios=[1, 1],
            layout="constrained",
        )
        mjd_smooth = np.linspace(mjd.min(), mjd.max(), 400)
        ax_data.plot(mjd_smooth, fit((mjd_smooth - reference_mjd) * DAY), color=GUIDE_COLOR)
        _errorbars_by_label(ax_data, table, mjd, f0, err_neg, err_pos)
        ax_data.set_ylabel(r"$\nu$ (Hz)")

        ax_resid.axhline(0, color=GUIDE_COLOR, linewidth=0.8, linestyle=":")
        _errorbars_by_label(ax_resid, table, mjd, residual, err_neg, err_pos, scale=scale)
        if ax_resid.get_legend() is not None:
            ax_resid.get_legend().remove()

        half_length = tangent_fraction * max(np.ptp(mjd), 1.0)
        secular_slope = np.polynomial.polynomial.polyval(
            x, np.polynomial.polynomial.polyder(fit.coeffs)
        )
        for m, r, f1, slope in zip(mjd, residual, table["f1"], secular_slope):
            if not np.isfinite(f1):
                continue
            dt = np.array([-half_length, half_length])
            ax_resid.plot(m + dt, (r + (f1 - slope) * dt * DAY) / scale, color="C3", lw=1)
        ax_resid.set_ylabel(rf"$\nu - \nu_{{\rm sec}}$ ($10^{{{exponent}}}$ Hz)")
        ax_resid.set_xlabel("MJD")
        return save_figure(fig, fname)


def _plot_f1(table, local, secular_fit, reference_mjd, fname):
    """Local F1 against time, with their mean, the intrinsic-scatter band, and the secular F1."""
    import matplotlib.pyplot as plt

    from .plotting import BAND_ALPHA, GUIDE_COLOR, figure_size, plot_style_context, save_figure

    good = np.isfinite(np.asarray(table["f1"], dtype=float))
    sub = table[good]
    mjd = np.asarray(sub["mjd"], dtype=float)
    f1 = np.asarray(sub["f1"], dtype=float)
    exponent = _power_of_ten(f1)
    scale = 10.0**exponent

    with plot_style_context():
        fig, ax = plt.subplots(figsize=figure_size("column"), layout="constrained")
        mean, scatter = local["mean"][0], local["intrinsic_scatter"]
        ax.axhspan(
            (mean - scatter) / scale, (mean + scatter) / scale, color="C0", alpha=BAND_ALPHA, lw=0
        )
        ax.axhline(mean / scale, color="C0", label="local mean")
        mjd_smooth = np.linspace(table["mjd"].min(), table["mjd"].max(), 400)
        secular_f1 = np.polynomial.polynomial.polyval(
            (mjd_smooth - reference_mjd) * DAY, np.polynomial.polynomial.polyder(secular_fit.coeffs)
        )
        ax.plot(mjd_smooth, secular_f1 / scale, color="C3", ls="--", label="secular")
        ax.axhline(0, color=GUIDE_COLOR, linewidth=0.8, linestyle=":")
        _errorbars_by_label(
            ax,
            sub,
            mjd,
            f1,
            np.asarray(sub["f1_err_neg"], dtype=float),
            np.asarray(sub["f1_err_pos"], dtype=float),
            scale=scale,
        )
        ax.legend(loc="best")
        ax.set_ylabel(rf"$\dot\nu$ ($10^{{{exponent}}}$ Hz s$^{{-1}}$)")
        ax.set_xlabel("MJD")
        return save_figure(fig, fname)


#: Fewest points a periodicity search is run on: three sinusoid parameters, plus two.
MIN_POINTS_FOR_PERIODICITY = 5

#: Quantities searched for periodicity: key, axis label, unit.
PERIODICITY_QUANTITIES = (
    ("f1", r"$\dot\nu$", r"Hz s$^{-1}$"),
    ("pulsed_rate", "pulsed rate", r"ct s$^{-1}$"),
)


def _periodicity_inputs(table, local, rate_exclude=()):
    """``{key: (t, y, var, yerr)}`` for each searchable quantity.

    Local F1 is weighted by its error and the intrinsic scatter in quadrature,
    as in :func:`local_f1_statistics`; the pulsed rate has no error and is
    unweighted.
    """
    mjd = np.asarray(table["mjd"], dtype=float)
    inputs = {}
    f1 = np.asarray(table["f1"], dtype=float)
    good = np.isfinite(f1)
    if local["n"] > 0:
        err = np.asarray(_symmetric_error(table, "f1"), dtype=float)[good]
        var = err**2 + local["intrinsic_scatter"] ** 2
        inputs["f1"] = (mjd[good], f1[good], var, err)
    rates = rate_values(table, rate_exclude)
    good = np.isfinite(rates)
    inputs["pulsed_rate"] = (mjd[good], rates[good], None, None)
    return inputs


def run_periodicity(table, local, min_period, max_period, rate_exclude=(), **search_kwargs):
    """:func:`~ell1fit.spin_periodicity.search_periodicity` on local F1 and the pulsed rate.

    Returns
    -------
    dict
        ``{key: result}``, where each result also carries its input ``t``,
        ``y`` and ``yerr`` for plotting. Quantities with fewer than
        :data:`MIN_POINTS_FOR_PERIODICITY` points are skipped with a warning.
    """
    results = {}
    for key, (t, y, var, yerr) in _periodicity_inputs(table, local, rate_exclude).items():
        if t.size < MIN_POINTS_FOR_PERIODICITY:
            logging.warning(f"Periodicity search skipped for {key}: only {t.size} points")
            continue
        result = search_periodicity(t, y, var, min_period, max_period, **search_kwargs)
        result.update(t=t, y=y, yerr=yerr)
        results[key] = result
        logging.info(
            f"{key}: best period {result['best_period_days']:.2f} d, "
            f"semi-amplitude {result['best_amplitude']:.3g}, "
            f"false-alarm probability {result['fap']:.3g}; "
            f"{result['detection_fraction']:.0%} of sinusoids of semi-amplitude "
            f">= {result['detectable_amplitude']:.3g} would be detected at "
            f"false-alarm probability {result['fap_level']:g}"
        )
    return results


def _json_ready(result):
    return {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}


def _plot_periodograms(results, fname):
    """Power against period for each searched quantity, with the shuffled-data
    detection threshold, over the spectral window of each one's sampling."""
    import matplotlib.pyplot as plt

    from .plotting import GUIDE_COLOR, figure_size, plot_style_context, save_figure

    labels = dict((key, label) for key, label, _ in PERIODICITY_QUANTITIES)
    with plot_style_context():
        fig, axes = plt.subplots(
            len(results) + 1, 1, sharex=True, figsize=figure_size("wide-tall"), layout="constrained"
        )
        ax_window = axes[-1]
        for i, (ax, (key, result)) in enumerate(zip(axes, results.items())):
            ax.plot(result["periods_days"], result["power"], color=f"C{i}", lw=1)
            ax.axhline(
                result["power_threshold"],
                color=GUIDE_COLOR,
                ls="--",
                lw=0.8,
                label=f"false-alarm prob. {result['fap_level']:g}",
            )
            ax.set_ylabel(f"power, {labels[key]}")
            ax.set_ylim(0, 1)
            ax.legend(loc="upper right")
            ax_window.plot(
                result["periods_days"], result["window"], color=f"C{i}", lw=1, label=labels[key]
            )
        ax_window.set_ylabel("spectral window")
        ax_window.set_ylim(0, 1)
        ax_window.legend(loc="upper right")
        ax_window.set_xlabel("period (d)")
        return save_figure(fig, fname)


def _plot_folded(results, fname):
    """Each searched quantity folded at its own best period, with the best-fit sinusoid."""
    import matplotlib.pyplot as plt

    from .plotting import DATA_COLOR, figure_size, plot_style_context, save_figure

    labels = {key: (label, unit) for key, label, unit in PERIODICITY_QUANTITIES}
    with plot_style_context():
        fig, axes = plt.subplots(
            1, len(results), figsize=figure_size("wide"), layout="constrained", squeeze=False
        )
        for i, (ax, (key, result)) in enumerate(zip(axes[0], results.items())):
            period = result["best_period_days"]
            t0 = result["t"].min()
            exponent = _power_of_ten(result["y"])
            scale = 10.0**exponent
            phase = ((result["t"] - t0) / period) % 1
            yerr = None if result["yerr"] is None else result["yerr"] / scale
            for shift in (0, 1):
                ax.errorbar(
                    phase + shift, result["y"] / scale, yerr=yerr, fmt="o", color=DATA_COLOR, ms=3
                )
            model_phase = np.linspace(0, 2, 200)
            c0, a, b = result["best_coeffs"]
            arg = 2 * np.pi * (t0 / period + model_phase)
            ax.plot(model_phase, (c0 + a * np.cos(arg) + b * np.sin(arg)) / scale, color=f"C{i}")
            label, unit = labels[key]
            scale_label = "" if exponent == 0 else rf"$10^{{{exponent}}}$ "
            ax.set_ylabel(f"{label} ({scale_label}{unit})")
            ax.set_xlabel(f"phase (P = {period:.2f} d, MJD$_0$ = {t0:.1f})")
            ax.set_title(f"false-alarm probability {result['fap']:.2g}")
        return save_figure(fig, fname)


def spin_statistics(
    files, extra_files=(), outroot="ell1stat", degree=1, reference_mjd=None, periodicity=None
):
    """Run the analysis, write ``{outroot}_results.json``, ``{outroot}_epochs.ecsv``
    and the figures, and return the summary dict.

    ``periodicity``, if given, is a dict of keyword arguments for
    :func:`run_periodicity` (at least ``min_period`` and ``max_period``).
    """
    table = load_spin_table(files, extra_files)
    if reference_mjd is None:
        reference_mjd = float(np.mean(table["mjd"]))

    fit, secular = secular_spin(table, reference_mjd, degree=degree)
    x = (np.asarray(table["mjd"]) - reference_mjd) * DAY
    table["f0_secular_residual"] = np.asarray(table["f0"]) - fit(x)
    local = local_f1_statistics(table)

    summary = {"n_epochs": len(table), "secular": secular, "local_f1": local}
    if local["n"] > 0:
        difference = secular["F1"][0] - local["mean"][0]
        summary["secular_minus_local_sigma"] = float(
            difference / np.hypot(secular["F1"][1], local["mean"][1])
        )

    logging.info(
        f"Secular F1 = {secular['F1'][0]:.3e} +- {secular['F1'][1]:.1e} Hz/s "
        f"(F0 scatter {secular['scatter_hz']:.2e} Hz around the degree-{degree} trend)"
    )
    if local["n"] > 0:
        logging.info(
            f"Local F1 mean = {local['mean'][0]:.3e} +- {local['mean'][1]:.1e} Hz/s, "
            f"intrinsic scatter {local['intrinsic_scatter']:.2e} Hz/s, "
            f"{local['fraction_spin_up']:.0%} of {local['n']} epochs spinning up"
        )
        logging.info(f"Secular minus local mean: {summary['secular_minus_local_sigma']:.1f} sigma")

    periodicity_results = {}
    if periodicity is not None:
        periodicity_results = run_periodicity(table, local, **periodicity)
        summary["periodicity"] = {k: _json_ready(r) for k, r in periodicity_results.items()}

    table.write(outroot + "_epochs.ecsv", overwrite=True)
    with open(outroot + "_results.json", "w") as fobj:
        json.dump(summary, fobj, indent=2)
    _plot_trend(table, fit, reference_mjd, outroot + "_trend")
    if local["n"] > 0:
        _plot_f1(table, local, fit, reference_mjd, outroot + "_f1")
    if periodicity_results:
        _plot_periodograms(periodicity_results, outroot + "_periodogram")
        _plot_folded(periodicity_results, outroot + "_folded")
    return summary


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Secular and local spin statistics from a set of per-epoch ell1fit results."
    )
    parser.add_argument("files", nargs="*", help="ell1fit .ecsv result files, one per epoch")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help=(
            "Table of extra epochs (columns MJD, F0, F0_err; optional F1, F1_err, "
            "pulsed_rate, label). Can be repeated."
        ),
    )
    parser.add_argument("-o", "--outroot", default="ell1stat", help="Output file root")
    parser.add_argument(
        "--degree", type=int, default=1, help="Degree of the secular F0(t) polynomial (default 1)"
    )
    parser.add_argument(
        "--reference-epoch",
        type=float,
        default=None,
        dest="reference_epoch",
        help="MJD the secular fit is referenced to (default: mean epoch)",
    )
    search = parser.add_argument_group(
        "periodicity search", "Run when both --min-period and --max-period are given"
    )
    search.add_argument("--min-period", type=float, default=None, help="Shortest period (days)")
    search.add_argument("--max-period", type=float, default=None, help="Longest period (days)")
    search.add_argument(
        "--oversample", type=int, default=5, help="Frequency steps per 1/baseline (default 5)"
    )
    search.add_argument(
        "--n-shuffle",
        type=int,
        default=1000,
        help="Shuffles calibrating the false-alarm probability (default 1000)",
    )
    search.add_argument(
        "--fap-level",
        type=float,
        default=0.01,
        help="False-alarm probability defining a detection, for the detectable amplitude "
        "(default 0.01)",
    )
    search.add_argument(
        "--rate-exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Leave epochs whose file name or label matches this shell pattern out of the "
            "pulsed-rate search (e.g. 'chandra*', whose count rates are not comparable). "
            "Can be repeated."
        ),
    )
    search.add_argument("--seed", type=int, default=None, help="Random seed for the shuffles")
    add_figure_format_argument(parser)
    parsed = parser.parse_args(args)
    if not parsed.files and not parsed.extra:
        parser.error("give at least one ell1fit result file or --extra table")
    periodicity = None
    if (parsed.min_period is None) != (parsed.max_period is None):
        parser.error("give both --min-period and --max-period, or neither")
    if parsed.min_period is not None:
        periodicity = {
            "min_period": parsed.min_period,
            "max_period": parsed.max_period,
            "rate_exclude": parsed.rate_exclude,
            "oversample": parsed.oversample,
            "n_shuffle": parsed.n_shuffle,
            "fap_level": parsed.fap_level,
            "rng": parsed.seed,
        }

    configure_logging()
    set_figure_format(parsed.figure_format)
    spin_statistics(
        parsed.files,
        extra_files=parsed.extra,
        outroot=parsed.outroot,
        degree=parsed.degree,
        reference_mjd=parsed.reference_epoch,
        periodicity=periodicity,
    )


if __name__ == "__main__":
    main()
