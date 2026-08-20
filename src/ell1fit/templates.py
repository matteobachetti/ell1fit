"""Pulse-template construction for ell1fit.

The template is the model of the pulse shape that the likelihood compares event
phases against. It is built by truncating the Fourier series of a folded profile
at ``nharm`` harmonics, which both smooths away counting noise and gives a
continuous function that can be evaluated at any phase.

Templates versus phaseograms
----------------------------
This module holds template *construction*, which feeds the likelihood and so is
part of the science path. Phaseogram and profile *plotting* lives in
:mod:`ell1fit.profile_plotting` and affects nothing but the figures. The two
used to share a module, which made it hard to see which code the fit depended
on.

Phase conventions
-----------------
``create_template_from_profile_harm`` returns the template together with the
phase offset of its peak. That offset is what the pipeline stores as each file's
``Phase_i`` starting value, and what
:func:`ell1fit.priors.assign_logpriors` centres that parameter's prior on.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d

from .phase_utils import phases_around_zero, phases_from_zero_to_one
from .plotting import plot_style_context as _plot_style_context


def create_template_from_profile_harm(
    profile,
    imagefile="template.png",
    nharm=None,
    final_nbin=None,
    plot=True,
):
    """Create a smooth pulse template from a folded profile.

    Parameters
    ----------
    profile : np.ndarray
        Folded pulse profile.
    imagefile : str, optional
        Where to save the diagnostic figure. Ignored when ``plot`` is False.
    nharm : int or None, optional
        Number of harmonics retained. Defaults to ``profile.size * 3 / 16``.
    final_nbin : int or None, optional
        Number of bins in the returned template.
    plot : bool, optional
        Whether to write the diagnostic figure. Iterative refinement rebuilds
        the template once per pass, and writing a figure each time is both slow
        and useless -- only the final template is worth looking at.

    Returns
    -------
    template : np.ndarray
        The smoothed template.
    additional_phase : float
        Phase offset of the template peak, wrapped to ``[-0.5, 0.5)``.
    """
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

    if not plot:
        return template * final_nbin / nbin, additional_phase

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
