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
from ._numba_compat import njit, prange
from scipy.interpolate import interp1d, make_interp_spline

from .phase_utils import phases_around_zero, phases_from_zero_to_one
from .plotting import plot_style_context as _plot_style_context


__all__ = [
    "UniformCubicTemplate",
    "_evaluate_uniform_cubic_floored",
    "_evaluate_uniform_cubic_floored_parallel",
    "create_template_from_profile_harm",
    "estimate_weighted_profile_std",
    "get_template_func",
]


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


@njit(fastmath=False)
def _evaluate_uniform_cubic(phases, coefficients, x0, dx, n_intervals):
    """Evaluate a cubic spline that sits on a uniformly spaced grid.

    Because the grid is uniform, locating the right polynomial piece is a
    division rather than a search -- which is what makes this so much faster
    than the general-purpose interpolator it replaces. Each piece is stored as
    its Taylor coefficients about the interval's left edge and evaluated by
    Horner's rule.
    """
    out = np.empty(phases.size)
    for i in range(phases.size):
        phase = phases[i] - np.floor(phases[i])
        j = int((phase - x0) / dx)
        if j < 0:
            j = 0
        elif j >= n_intervals:
            j = n_intervals - 1
        u = phase - (x0 + j * dx)
        out[i] = coefficients[j, 0] + u * (
            coefficients[j, 1] + u * (coefficients[j, 2] + u * coefficients[j, 3])
        )
    return out


@njit(fastmath=False)
def _evaluate_uniform_cubic_floored(phases, coefficients, x0, dx, n_intervals, floor):
    """As :func:`_evaluate_uniform_cubic`, but clamping the result as it goes.

    The likelihood needs strictly positive densities before taking logs. Doing
    the clamp inside the interpolation loop removes two extra full-length passes
    (a ``nan_to_num`` and a ``clip``) and the temporaries they allocate, which at
    a million events is real memory traffic on every posterior evaluation.
    """
    out = np.empty(phases.size)
    for i in range(phases.size):
        phase = phases[i] - np.floor(phases[i])
        j = int((phase - x0) / dx)
        if j < 0:
            j = 0
        elif j >= n_intervals:
            j = n_intervals - 1
        u = phase - (x0 + j * dx)
        value = coefficients[j, 0] + u * (
            coefficients[j, 1] + u * (coefficients[j, 2] + u * coefficients[j, 3])
        )
        if np.isnan(value) or value < floor:
            value = floor
        out[i] = value
    return out


@njit(fastmath=False, parallel=True)
def _evaluate_uniform_cubic_floored_parallel(phases, coefficients, x0, dx, n_intervals, floor):
    """Multi-threaded twin of :func:`_evaluate_uniform_cubic_floored`.

    Each iteration writes one independent output element, so there is no
    reduction whose order could change: the result is bitwise identical to the
    serial kernel, verified in the tests. Only the wall-clock time differs.
    """
    out = np.empty(phases.size)
    for i in prange(phases.size):
        phase = phases[i] - np.floor(phases[i])
        j = int((phase - x0) / dx)
        if j < 0:
            j = 0
        elif j >= n_intervals:
            j = n_intervals - 1
        u = phase - (x0 + j * dx)
        value = coefficients[j, 0] + u * (
            coefficients[j, 1] + u * (coefficients[j, 2] + u * coefficients[j, 3])
        )
        if np.isnan(value) or value < floor:
            value = floor
        out[i] = value
    return out


@njit(fastmath=False)
def _evaluate_uniform_cubic_mixture(phases, coefficients, x0, dx, n_intervals, weights, floor):
    """Interpolate, form the weighted mixture, and clamp it, in one pass.

    The weighted likelihood's per-event term is ``1 + w_i (T - 1)``, so that --
    not the template -- is what has to stay positive before the log. Clamping
    the template instead would be too strict: a correctly undiluted template can
    dip below zero at the trough of a strong pulse while every mixture value is
    still safely positive, because ``w_i <= 1``.

    Weights are sanitized here too, so the caller needs no separate passes for
    them; the whole weighted path stays a single traversal, which is the reason
    this fused kernel exists at all.
    """
    out = np.empty(phases.size)
    for i in range(phases.size):
        phase = phases[i] - np.floor(phases[i])
        j = int((phase - x0) / dx)
        if j < 0:
            j = 0
        elif j >= n_intervals:
            j = n_intervals - 1
        u = phase - (x0 + j * dx)
        value = coefficients[j, 0] + u * (
            coefficients[j, 1] + u * (coefficients[j, 2] + u * coefficients[j, 3])
        )
        weight = weights[i]
        if np.isnan(weight) or weight < 0.0:
            weight = 0.0
        elif weight > 1.0:
            weight = 1.0
        mixture = weight * value + (1.0 - weight)
        if np.isnan(mixture) or mixture < floor:
            mixture = floor
        out[i] = mixture
    return out


@njit(fastmath=False, parallel=True)
def _evaluate_uniform_cubic_mixture_parallel(
    phases, coefficients, x0, dx, n_intervals, weights, floor
):
    """Multi-threaded twin of :func:`_evaluate_uniform_cubic_mixture`.

    As with the other parallel kernel, each iteration writes one independent
    output element, so the result is bitwise identical to the serial version.
    """
    out = np.empty(phases.size)
    for i in prange(phases.size):
        phase = phases[i] - np.floor(phases[i])
        j = int((phase - x0) / dx)
        if j < 0:
            j = 0
        elif j >= n_intervals:
            j = n_intervals - 1
        u = phase - (x0 + j * dx)
        value = coefficients[j, 0] + u * (
            coefficients[j, 1] + u * (coefficients[j, 2] + u * coefficients[j, 3])
        )
        weight = weights[i]
        if np.isnan(weight) or weight < 0.0:
            weight = 0.0
        elif weight > 1.0:
            weight = 1.0
        mixture = weight * value + (1.0 - weight)
        if np.isnan(mixture) or mixture < floor:
            mixture = floor
        out[i] = mixture
    return out


#: Event count above which threading the template evaluation pays for itself.
#: Below it the thread launch overhead dominates and the parallel kernel is
#: markedly *slower* -- measured at 0.35x for ten thousand events, against 2.6x
#: at two hundred thousand and 4.7x at two million.
PARALLEL_TEMPLATE_THRESHOLD = 50_000


class UniformCubicTemplate:
    """A pulse template that is callable and can score events in one pass.

    Behaves as the plain ``phase -> density`` function every caller expects, so
    plotting and diagnostics are unaffected. The likelihood additionally uses
    :meth:`loglike`, which fuses interpolation, flooring and the log-sum into a
    single traversal rather than three.
    """

    def __init__(self, coefficients, x0, dx, n_intervals):
        self.coefficients = coefficients
        self.x0 = x0
        self.dx = dx
        self.n_intervals = n_intervals

    def __call__(self, phases):
        return _evaluate_uniform_cubic(
            np.ascontiguousarray(phases, dtype=float),
            self.coefficients,
            self.x0,
            self.dx,
            self.n_intervals,
        )

    def loglike(self, phases, weights=None, floor=1e-12):
        """Total log-likelihood of ``phases`` under this template.

        The interpolation and clamp happen in one compiled pass; the logarithm
        and the sum are then left to numpy, which is both faster and more
        accurate than doing them element-by-element. Faster because numpy's
        ``log`` is SIMD-vectorised, where a scalar loop calling ``log`` once per
        event is not -- measured at roughly 0.6x, so fusing everything is a
        pessimisation. More accurate because ``np.sum`` accumulates pairwise,
        with error growing like log(N) rather than the N of a running total: at
        two million events that is a 30x smaller error than the loop this
        replaces.
        """
        phases = np.ascontiguousarray(phases, dtype=float)
        threaded = phases.size >= PARALLEL_TEMPLATE_THRESHOLD

        if weights is None:
            kernel = (
                _evaluate_uniform_cubic_floored_parallel
                if threaded
                else _evaluate_uniform_cubic_floored
            )
            terms = kernel(phases, self.coefficients, self.x0, self.dx, self.n_intervals, floor)
        else:
            kernel = (
                _evaluate_uniform_cubic_mixture_parallel
                if threaded
                else _evaluate_uniform_cubic_mixture
            )
            terms = kernel(
                phases,
                self.coefficients,
                self.x0,
                self.dx,
                self.n_intervals,
                np.ascontiguousarray(weights, dtype=float),
                floor,
            )

        return float(np.sum(np.log(terms)))


def _template_spline_grid(template):
    """Build the wrapped, normalized sample grid a template is interpolated on."""
    dph = 1 / template.size
    phases = np.linspace(0, 1, template.size + 1) + dph / 2

    allph = np.concatenate(([-dph / 2], phases))
    allt = np.concatenate((template[-1:], template, template[:1]))
    allt = allt / (np.sum(template) * dph)
    return allph, allt, dph


def get_template_func(template, backend="numba"):
    """Get a cubic interpolation function of a pulse template.

    The returned callable maps pulse phase to probability density and is
    evaluated once per event on every posterior call, which makes it the single
    hottest piece of the whole fit -- around 80% of the cost of one likelihood
    evaluation before this was specialized.

    Parameters
    ----------
    template : np.ndarray
        Pulse template, uniformly sampled in phase.
    backend : {"numba", "scipy"}, optional
        ``"numba"`` evaluates the spline directly, exploiting the fact that the
        sample grid is uniform. ``"scipy"`` uses ``interp1d`` and is retained as
        the reference implementation: it is what the fast path is validated
        against, and what to fall back to if a result is ever in doubt.

    Returns
    -------
    callable
        Phase to density. Input phase is wrapped, so any real value is valid.

    Notes
    -----
    The two backends agree to about 1e-14 relative -- roughly 80 machine
    epsilon. That residual is not one of them being wrong: it is the difference
    between evaluating the *same* cubic in Taylor form about each interval's
    left edge and in scipy's B-spline basis. For scale, the pipeline's phase
    accuracy in float64 is around 1e-10 cycles, so this sits four orders of
    magnitude below the existing numerical floor.
    """
    allph, allt, dph = _template_spline_grid(template)

    if backend == "scipy":
        template_interp = interp1d(allph, allt, kind="cubic")

        def template_fun_scipy(x):
            ph = x - np.floor(x)
            return template_interp(ph)

        return template_fun_scipy

    if backend != "numba":
        raise ValueError(f"Unknown template backend {backend!r}; use 'numba' or 'scipy'")

    # interp1d(kind="cubic") is make_interp_spline(k=3) underneath, so this
    # reproduces the identical spline. Each uniform interval lies inside a
    # single polynomial piece, so its cubic is recovered exactly from the
    # spline's derivatives at the interval's left edge.
    spline = make_interp_spline(allph, allt, k=3)
    starts = allph[:-1]
    coefficients = np.ascontiguousarray(
        np.stack(
            [
                spline(starts),
                spline.derivative(1)(starts),
                spline.derivative(2)(starts) / 2.0,
                spline.derivative(3)(starts) / 6.0,
            ],
            axis=1,
        )
    )
    return UniformCubicTemplate(coefficients, float(allph[0]), dph, starts.size)


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
