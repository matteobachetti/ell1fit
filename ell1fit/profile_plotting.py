"""Template, profile, and phaseogram plotting helpers for ell1fit."""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from .phase_utils import phases_around_zero, phases_from_zero_to_one
from .plotting import plot_style_context as _plot_style_context


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


def create_template_from_profile_harm(
    profile,
    imagefile="template.png",
    nharm=None,
    final_nbin=None,
):
    """Create a smooth pulse template from a folded profile."""
    nbin = profile.size
    prof = np.concatenate((profile, profile, profile))
    dph = 1 / profile.size
    ft = fft(prof)
    freq = fftfreq(prof.size, dph)

    if nharm is None:
        nharm = max(1, int(prof.size / 16))

    if final_nbin is None:
        final_nbin = nbin

    if nharm == 1:
        additional_phase = -np.angle(ft[3]) / 2 / np.pi
        B = np.mean(profile)
        A = np.abs(ft[3]) / prof.size * 2 / B

        def template_func(x):
            return B * (1 + A * np.cos(2 * np.pi * x))

    else:
        oversample_factor = 10
        dph_fine = 1 / final_nbin / oversample_factor
        new_ft_fine = np.zeros(final_nbin * 3 * oversample_factor, dtype=complex)
        new_ft_freq = fftfreq(final_nbin * 3 * oversample_factor, dph_fine)

        new_ft_fine[np.abs(new_ft_freq) <= nharm] = ft[np.abs(freq) <= nharm]

        template_fine = ifft(new_ft_fine).real * oversample_factor * final_nbin / nbin

        phases_fine = np.arange(0.5 * dph_fine, 3, dph_fine)

        templ_func_fine = interp1d(phases_fine, template_fine, kind="cubic", assume_sorted=True)

        additional_phase = (
            np.argmax(template_fine[: final_nbin * oversample_factor])
            / final_nbin
            / oversample_factor
            + dph_fine / 2
        )

        def template_func(x):
            return templ_func_fine(1 + x + additional_phase)

        logging.debug(f"Additional phase: {additional_phase}")

    dph = 1 / final_nbin
    phas = np.arange(dph / 2, 1, dph)

    template = template_func(phas)

    additional_phase = phases_around_zero(additional_phase)
    template = template[:final_nbin].real

    with _plot_style_context():
        fig = plt.figure(figsize=(3.5, 2.65))
        plt.plot(np.arange(0.5 / nbin, 1, 1 / nbin), profile, drawstyle="steps-mid", label="data")
        plt.plot(phas[:final_nbin], template, label="template values", ls="--", lw=2)
        plt.plot(
            phas[:final_nbin],
            template_func(phas[:final_nbin]),
            label="template func",
            ls=":",
            lw=2,
        )
        plt.plot(
            phas[:final_nbin],
            template_func(phas[:final_nbin] - additional_phase),
            label="template aligned",
            lw=3,
        )
        plt.axvline(phases_from_zero_to_one(additional_phase))
        plt.legend
        plt.savefig(imagefile)
        plt.close(fig)
    return template * final_nbin / nbin, additional_phase


def get_template_func(template):
    """Get a cubic interpolation function of a pulse template."""
    dph = 1 / template.size
    phases = np.linspace(0, 1, template.size + 1) + dph / 2

    allph = np.concatenate(([-dph / 2], phases))
    allt = np.concatenate((template[-1:], template, template[:1]))
    allt /= np.sum(template) * dph

    template_interp = interp1d(allph, allt, kind="cubic")

    def template_fun(x):
        ph = x - np.floor(x)
        return template_interp(ph)

    return template_fun


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


def estimate_weighted_profile_std(weights, nbin=16, ntrials=100):
    """Estimate expected weighted-profile scatter under pure noise."""
    logging.info(f"Estimating weighted profile std (ntrials={ntrials}, nbin={nbin})")

    std = np.mean(
        [
            np.std(
                np.histogram(
                    np.random.uniform(0, 1, len(weights)),
                    bins=np.linspace(0, 1, nbin + 1),
                    weights=weights,
                )[0]
            )
            for j in range(ntrials)
        ]
    )

    return std
