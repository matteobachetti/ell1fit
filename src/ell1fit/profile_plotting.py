"""Profile and phaseogram plotting for ell1fit.

Presentation only: nothing here feeds the fit. Template *construction*, which
does, lives in :mod:`ell1fit.templates`.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .phase_utils import phases_from_zero_to_one
from .plotting import plot_style_context as _plot_style_context


__all__ = [
    "_compare_phaseograms",
    "normalize_dyn_profile",
]


def normalize_dyn_profile(dynprof, norm):
    """Normalize a dynamical profile (e.g. a phaseogram)."""
    dynprof = np.array(dynprof, dtype=float)

    if norm is None:
        norm = ""

    if norm.endswith("_smooth"):
        dynprof = gaussian_filter(dynprof, 1, mode=("constant", "wrap"))
        norm = norm.replace("_smooth", "")

    if norm.startswith("median"):
        y_mean = np.median(dynprof, axis=1)
        prof_mean = np.median(dynprof, axis=0)
        norm = norm.replace("median", "")
    else:
        y_mean = np.mean(dynprof, axis=1)
        prof_mean = np.mean(dynprof, axis=0)
        norm = norm.replace("mean", "")

    if "ratios" in norm:
        dynprof /= prof_mean[np.newaxis, :]
        norm = norm.replace("ratios", "")
        y_mean = np.mean(dynprof, axis=1)

    y_min = np.min(dynprof, axis=1)
    y_max = np.max(dynprof, axis=1)
    y_std = np.std(np.diff(dynprof, axis=0)) / np.sqrt(2)

    if norm in ("", "none"):
        pass
    elif norm == "to1":
        dynprof -= y_min[:, np.newaxis]
        dynprof /= (y_max - y_min)[:, np.newaxis]
    elif norm == "std":
        dynprof -= y_mean[:, np.newaxis]
        dynprof /= y_std
    elif norm == "sub":
        dynprof -= y_mean[:, np.newaxis]
    elif norm == "norm":
        dynprof -= y_mean[:, np.newaxis]
        dynprof /= y_mean[:, np.newaxis]
    else:
        warnings.warn(f"Profile normalization {norm} not known. Using default")
    return dynprof


def _plot_phaseogram(phases, times, ax0, ax1, norm="meansub_smooth"):
    """Plot folded profile and phaseogram for one event list."""
    ph = np.concatenate((phases, phases + 1)).astype(float)
    tm = np.concatenate((times, times)).astype(float) / 86400

    nbin = 32
    bins = np.linspace(0, 2, nbin + 1)
    prof, _ = np.histogram(ph, bins=bins)

    ax0.plot(bins[:-1] + 0.5 / nbin, prof, color="k", alpha=0.5)
    for num in (0.5, 1, 1.5):
        ax1.axvline(num, color="grey", lw=2, ls="--")
    H, xedges, yedges = np.histogram2d(ph, tm, bins=(bins, nbin))
    X, Y = np.meshgrid(xedges, yedges)
    H = normalize_dyn_profile(H.T, norm)
    ax1.pcolormesh(X, Y, H, cmap="cubehelix")
    for num in (0.5, 1, 1.5):
        ax1.axvline(num, color="grey", lw=2, ls="--")

    ax1.set_xlabel("Phase")
    ax1.set_ylabel("Time from pepoch (d)")
    ax1.set_xlim([0, 2])


def _compare_phaseograms(phase1, phase2, times, fname):
    """Compare two phase solutions by plotting side-by-side phaseograms."""
    with _plot_style_context():
        fig = plt.figure(figsize=(7, 7))
        gs = plt.GridSpec(2, 2, height_ratios=(1, 3))
        ax00 = plt.subplot(gs[0, 0])
        ax10 = plt.subplot(gs[1, 0], sharex=ax00)
        ax01 = plt.subplot(gs[0, 1], sharey=ax00)
        ax11 = plt.subplot(gs[1, 1], sharex=ax01, sharey=ax10)

        _plot_phaseogram(phases_from_zero_to_one(phase1), times, ax00, ax10)
        _plot_phaseogram(phases_from_zero_to_one(phase2), times, ax01, ax11)

        plt.savefig(fname)
        plt.close(fig)
