"""Energy-dependent event weighting for ell1fit.

Pulsed fraction usually varies with photon energy, so events in energy bands
where the pulse is strong carry more timing information than events where it is
weak. Weighting by that trend recovers signal that an unweighted fit throws
away.

The weights returned here feed the weighted branch of
:func:`ell1fit.likelihoods.pletsch_clarke_likelihood`, which requires them to
lie in ``[0, 1]``: a weight of 1 means "trust this event's phase fully", 0 means
"treat it as unmodulated background". The normalization at the end of
:func:`pf_weight_versus_energy` enforces that by scaling the peak amplitude to
1, and falls back to uniform weights if the amplitude trend is degenerate.

Only the *shape* of the weight curve matters. Both the weighted profile and its
noise level scale linearly with the weights, so rescaling them all by a constant
leaves the weighted statistic — and the fit — unchanged. That is why peak
normalization is free to be chosen for the ``[0, 1]`` constraint alone.

How the trend is measured
-------------------------

The pulsed amplitude at energy ``E`` is estimated by projecting each event's
phase onto the harmonic model of the whole observation's pulse profile
(:func:`_pulse_modulation`). That projection is a *linear*, unbiased estimator:
it is centered on the true amplitude and, crucially, is free to come out
negative where the pulse is undetected. Estimating amplitude from
:math:`Z_n^2` instead — as this module used to — is rectified, so pure noise
biases every low-significance band upward by an amount that depends on its
count rate.

The trend is then a penalized cubic spline in ``log E``, fit to the individual
events with no energy binning at all and with the smoothing strength chosen by
generalized cross-validation. Binning was the other source of error: energy
bands hold wildly different numbers of counts, so any fixed binning either
smears real structure where counts are plentiful or reports noise as signal
where they are scarce. A penalized fit borrows strength across neighboring
energies instead, which is exactly what a hand-tuned detection-limit floor was
previously trying to approximate.
"""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.interpolate import BSpline

from .phase_utils import _calculate_phases
from .plotting import plot_style_context as _plot_style_context


__all__ = [
    "pf_weight_versus_energy",
]


# Enough interior knots that the penalty, not the knot spacing, sets the
# resolution: doubling this leaves the fitted curve unchanged.
_N_KNOTS = 40
_SPLINE_DEGREE = 3
# Smoothing strengths scanned by generalized cross-validation. The useful range
# is wide because it absorbs the number of events, which spans orders of
# magnitude between a short snapshot and a deep observation.
_LOG_LAMBDA_GRID = np.arange(-4.0, 10.01, 0.25)
# Below this, an energy-resolved pulsed fraction is not a measurement.
_MIN_EVENTS = 100


def _pulse_modulation(phases, nharm):
    """Harmonic model of the pulse shape, normalized to unit mean square.

    Returns a callable ``M(phase)`` with zero mean and
    :math:`\\langle M^2 \\rangle = 1` over a cycle, or None if the profile
    carries no modulation at all.

    The normalization makes the per-event projection ``M(phase)`` have unit
    variance regardless of how strongly pulsed the source is, which is what lets
    :func:`_fit_penalized_spline` treat every event as carrying equal weight.
    """
    harmonics = np.arange(1, nharm + 1)
    coefficients = np.array([np.mean(np.exp(-2j * np.pi * k * phases)) for k in harmonics])
    # <M^2> for M(phi) = 2 Re(sum_k c_k exp(2 pi i k phi)).
    mean_square = 2 * np.sum(np.abs(coefficients) ** 2)
    if not np.isfinite(mean_square) or mean_square <= 0:
        return None

    norm = np.sqrt(mean_square)

    def modulation(phase):
        phase = np.asarray(phase, dtype=float)
        waves = np.exp(2j * np.pi * np.outer(phase, harmonics))
        return 2 * np.real(waves @ coefficients) / norm

    return modulation


class _SplineFit:
    """A fitted penalized spline plus what the diagnostic plot needs from it."""

    def __init__(self, knots, coefficients, covariance, edf, lam, low, high):
        self.spline = BSpline(knots, coefficients, _SPLINE_DEGREE)
        self.knots = knots
        self.covariance = covariance
        self.edf = edf
        self.lam = lam
        self.low = low
        self.high = high

    def __call__(self, x):
        return self.spline(np.clip(x, self.low, self.high))

    def uncertainty(self, x):
        """One-sigma uncertainty of the fitted curve at ``x``."""
        design = BSpline.design_matrix(
            np.clip(np.asarray(x, dtype=float), self.low, self.high),
            self.knots,
            _SPLINE_DEGREE,
        ).toarray()
        return np.sqrt(np.einsum("ij,jk,ik->i", design, self.covariance, design))


def _fit_penalized_spline(x, y, n_knots=_N_KNOTS):
    """Fit ``y(x)`` with a cubic spline under a second-difference penalty.

    ``y`` is one value per event and is homoscedastic by construction (see
    :func:`_pulse_modulation`), so the fit needs no per-point weights: every
    event counts once, and dense energies automatically constrain the curve more
    tightly than sparse ones. Nothing is binned or discarded.

    The smoothing strength is chosen by generalized cross-validation over
    :data:`_LOG_LAMBDA_GRID`. Returns a :class:`_SplineFit`, or None if the
    normal equations are singular at every trial value.
    """
    low, high = float(np.min(x)), float(np.max(x))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None

    knots = np.concatenate(
        [
            np.repeat(low, _SPLINE_DEGREE),
            np.linspace(low, high, n_knots),
            np.repeat(high, _SPLINE_DEGREE),
        ]
    )
    design = BSpline.design_matrix(np.clip(x, low, high), knots, _SPLINE_DEGREE)
    # Accumulate the normal equations once: they are (n_coeff x n_coeff), so the
    # smoothing scan below never touches the per-event arrays again.
    gram = (design.T @ design).toarray()
    projection = design.T @ y
    n_coefficients = gram.shape[0]
    differences = np.diff(np.eye(n_coefficients), n=2, axis=0)
    penalty = differences.T @ differences

    n_events = y.size
    total_square = float(y @ y)

    best = None
    for log_lambda in _LOG_LAMBDA_GRID:
        lam = 10.0**log_lambda
        system = gram + lam * penalty
        try:
            coefficients = np.linalg.solve(system, projection)
            inverse = np.linalg.inv(system)
        except np.linalg.LinAlgError:
            continue
        edf = float(np.trace(inverse @ gram))
        residual_square = (
            total_square
            - 2 * float(coefficients @ projection)
            + float(coefficients @ gram @ coefficients)
        )
        denominator = max(n_events - edf, 1.0)
        gcv = n_events * residual_square / denominator**2
        if not np.isfinite(gcv):
            continue
        if best is None or gcv < best[0]:
            best = (gcv, coefficients, inverse, edf, lam, residual_square / denominator)

    if best is None:
        return None

    _, coefficients, inverse, edf, lam, variance = best
    return _SplineFit(knots, coefficients, variance * inverse, edf, lam, low, high)


def _energy_coordinate(energies):
    """Fitting coordinate for the energy axis.

    Pulsed-fraction structure — continuum curvature, lines, instrumental edges —
    is spread out evenly in log energy, not in energy, so that is where a
    uniform knot grid belongs. PI channels can be zero, so fall back to a linear
    axis when the values do not all lie above zero.
    """
    if np.all(energies > 0):
        return np.log(energies), True
    return np.array(energies, dtype=float), False


def _binned_amplitudes(coordinate, projection, n_bins=40):
    """Bin the per-event projection for display only; the fit never sees this."""
    edges = np.linspace(coordinate.min(), coordinate.max(), n_bins + 1)
    index = np.clip(np.digitize(coordinate, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    good = counts > 0
    centers = (0.5 * (edges[:-1] + edges[1:]))[good]
    means = (np.bincount(index, weights=projection, minlength=n_bins) / np.maximum(counts, 1))[good]
    # Unit variance per event, so the standard error is set by the count alone.
    errors = 1.0 / np.sqrt(counts[good])
    return centers, means, errors, counts[good]


def _plot_weight_diagnostic(filename, energies, coordinate, is_log, fit, projection, peak, nharm):
    """Save the amplitude-versus-energy diagnostic for one observation.

    Everything is drawn in units of the applied weight — the fitted curve
    divided by its peak — so the measured points, the fit, its uncertainty and
    the number that actually multiplies each event in the likelihood all share
    one axis. Values below zero are real: the estimator is unbiased, so bands
    with no pulse scatter either side of zero, and the shaded strip marks where
    they get clipped to a weight of zero.
    """
    centers, means, errors, _ = _binned_amplitudes(coordinate, projection)
    grid = np.linspace(fit.low, fit.high, 400)
    curve = fit(grid)
    band = fit.uncertainty(grid)

    curve, band, means, errors = curve / peak, band / peak, means / peak, errors / peak

    if is_log:
        plot_energy, plot_centers = np.exp(grid), np.exp(centers)
        edges = np.logspace(np.log10(energies.min()), np.log10(energies.max()), 60)
        xlabel = "Energy (keV)"
    else:
        plot_energy, plot_centers = grid, centers
        edges = np.linspace(energies.min(), energies.max(), 60)
        xlabel = "PI channel"

    # What the weighting buys, in units the reader can act on: an unweighted fit
    # would need this much more exposure to reach the same phase precision. Both
    # terms use the fitted amplitude, so nothing about the truth is assumed.
    amplitude = np.clip(fit(coordinate), 0, None)
    total = float(np.sum(amplitude))
    gain = coordinate.size * float(np.sum(amplitude**2)) / total**2 if total > 0 else 1.0

    with _plot_style_context():
        fig, (ax_spectrum, ax_weight) = plt.subplots(
            2,
            1,
            figsize=(7, 5.5),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 2.4]},
        )

        ax_spectrum.hist(energies, bins=edges, histtype="step", color="0.3", lw=0.9)
        ax_spectrum.set_yscale("log")
        ax_spectrum.set_ylabel("counts / bin")
        ax_spectrum.tick_params(labelbottom=False)

        low, high = -0.6, 1.6
        low = min(low, float(np.min(curve - 2 * band)) - 0.1)
        high = max(high, float(np.max(curve + 2 * band)) + 0.1)
        ax_weight.axhspan(low, 0, color="0.85", alpha=0.5, lw=0, zorder=0)
        ax_weight.axhline(0, color="k", lw=0.5, zorder=1)
        ax_weight.annotate(
            "clipped to zero weight",
            xy=(0.015, 0.02),
            xycoords="axes fraction",
            va="bottom",
            color="0.4",
        )
        ax_weight.errorbar(
            plot_centers,
            means,
            yerr=errors,
            fmt="o",
            ms=2.5,
            lw=0.7,
            color="0.45",
            alpha=0.85,
            zorder=2,
            label="measured (binned for display only)",
        )
        ax_weight.fill_between(
            plot_energy, curve - 2 * band, curve + 2 * band, color="tab:blue", alpha=0.15, lw=0
        )
        ax_weight.fill_between(
            plot_energy, curve - band, curve + band, color="tab:blue", alpha=0.3, lw=0
        )
        ax_weight.plot(
            plot_energy,
            curve,
            color="tab:blue",
            lw=1.6,
            zorder=3,
            label=(
                f"penalized spline, {fit.edf:.1f} effective d.o.f. ($\\pm$1$\\sigma$, 2$\\sigma$)"
            ),
        )
        ax_weight.set_ylim(low, high)
        ax_weight.set_xlabel(xlabel)
        ax_weight.set_ylabel(f"applied weight ($Z^2_{nharm}$ amplitude / peak)")
        ax_weight.legend(loc="upper left", framealpha=0.9)
        ax_weight.annotate(
            f"equivalent to {gain:.2f}$\\times$ the exposure of an unweighted fit",
            xy=(0.98, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.7", "boxstyle": "round,pad=0.3"},
        )

        if is_log:
            ax_weight.set_xscale("log")
            # The default log formatter writes 3 x 10^0 and runs the labels into
            # each other on the narrow decade this axis usually spans.
            for axis in (ax_spectrum.xaxis, ax_weight.xaxis):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(NullFormatter())
            # Leave a margin so a tick sitting exactly on the first or last
            # event energy is not swallowed by the spine.
            ax_weight.set_xlim(energies.min() * 0.93, energies.max() * 1.07)
            ticks = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 70, 100])
            inside = (ticks >= energies.min() * 0.99) & (ticks <= energies.max() * 1.01)
            ax_weight.set_xticks(ticks[inside])

        fig.suptitle(os.path.basename(filename))
        fig.tight_layout()
        fig.savefig(f"{filename}.jpg")
        plt.close(fig)


def pf_weight_versus_energy(
    times, energies, parameters, nharm=1, tolerance=1e-8, plot_root_file_name=None
):
    """Estimate per-event weights from pulsed amplitude versus energy.

    For each input observation, this function computes phases with the current
    timing model, projects them onto an ``nharm``-harmonic model of that
    observation's own pulse profile, and fits the resulting per-event amplitude
    estimates against energy with a penalized cubic spline. The fitted curve is
    evaluated at each event's energy and rescaled to peak at 1 to obtain
    per-event weights in ``[0, 1]``.

    Parameters
    ----------
    times : list of np.ndarray
        Event times (seconds from each file PEPOCH), one array per file.
    energies : list of np.ndarray
        Event energies (or PI channels if provided upstream), one array per file.
    parameters : dict
        Timing/orbital parameter dictionary consumed by
        :func:`_calculate_phases`.
    nharm : int, optional
        Number of harmonics in the pulse model the amplitudes are measured
        against.
    tolerance : float, optional
        Convergence tolerance (seconds) for deorbiting iterations.
    plot_root_file_name : list of str or None, optional
        If provided, save one diagnostic amplitude-versus-energy plot per file
        using these roots.

    Returns
    -------
    list of np.ndarray
        Event weights for each file, aligned with ``times`` and ``energies``.
    """
    n_files = len(times)
    phases = _calculate_phases(times, parameters, tolerance=tolerance)

    weights = []
    for i in range(n_files):
        local_phases = np.asarray(phases[i], dtype=float)
        local_energies = np.asarray(energies[i], dtype=float)
        uniform = np.ones_like(local_energies)

        if local_phases.size < _MIN_EVENTS:
            warnings.warn(
                f"Only {local_phases.size} events: too few to measure pulsed fraction "
                "versus energy. Falling back to uniform weights."
            )
            weights.append(uniform)
            continue

        modulation = _pulse_modulation(local_phases, nharm)
        if modulation is None:
            warnings.warn("Pulse profile carries no modulation; falling back to uniform weights.")
            weights.append(uniform)
            continue

        projection = modulation(local_phases)
        coordinate, is_log = _energy_coordinate(local_energies)
        fit = _fit_penalized_spline(coordinate, projection)
        if fit is None:
            warnings.warn(
                "Could not fit pulsed amplitude versus energy; falling back to uniform weights."
            )
            weights.append(uniform)
            continue

        logging.info(
            f"Pulsed amplitude versus energy: penalized spline with "
            f"{fit.edf:.1f} effective degrees of freedom (lambda = {fit.lam:.3g})"
        )

        amplitude = fit(coordinate)
        peak = np.nanmax(amplitude)
        if not np.isfinite(peak) or peak <= 0:
            warnings.warn(
                "Could not normalize pulsed-fraction weights; falling back to uniform weights."
            )
            weights.append(uniform)
            continue

        local_weights = np.clip(amplitude / peak, 0.0, 1.0)
        local_weights = np.nan_to_num(local_weights, nan=0.0, posinf=1.0, neginf=0.0)

        if plot_root_file_name is not None:
            _plot_weight_diagnostic(
                plot_root_file_name[i],
                local_energies,
                coordinate,
                is_log,
                fit,
                projection,
                peak,
                nharm,
            )

        weights.append(local_weights)

    return weights
