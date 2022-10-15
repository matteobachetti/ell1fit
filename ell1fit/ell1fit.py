import warnings

import os
import copy
import matplotlib.pyplot as plt
import numpy as np
from hendrics.io import load_events
from pint.models import get_model

import matplotlib as mpl

from scipy.interpolate import interp1d

import emcee
import corner
from numba import njit, vectorize, int64, float32, float64, prange
from astropy.table import Table, vstack
from astropy.time import Time
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

from numpy.fft import ifft, fft, fftfreq
from . import version

params = {
    "font.size": 7,
    "xtick.major.size": 0,
    "xtick.minor.size": 0,
    "xtick.major.width": 0,
    "xtick.minor.width": 0,
    "ytick.major.size": 0,
    "ytick.minor.size": 0,
    "ytick.major.width": 0,
    "ytick.minor.width": 0,
    "figure.figsize": (3.5, 3.5),
    "axes.grid": True,
    "grid.color": "grey",
    "grid.linewidth": 0.3,
    "grid.linestyle": ":",
    "axes.grid.axis": "y",
    "axes.grid.which": "both",
    "axes.axisbelow": False,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "figure.dpi": 300,  # the left side of the subplots of the figure
    "figure.subplot.left": 0.195,  # the left side of the subplots of the figure
    "figure.subplot.right": 0.97,  # the right side of the subplots of the figure
    "figure.subplot.bottom": 0.145,  # the bottom of the subplots of the figure
    "figure.subplot.top": 0.97,  # the top of the subplots of the figure
    "figure.subplot.wspace": 0.2,  # the amount of width reserved for space between subplots,
    # expressed as a fraction of the average axis width
    "figure.subplot.hspace": 0.2,  # the amount of height reserved for space between subplots,
    # expressed as a fraction of the average axis height
}
mpl.rcParams.update(params)


def splitext_improved(path):
    """
    Examples
    --------
    >>> np.all(splitext_improved("a.tar.gz") ==  ('a', '.tar.gz'))
    True
    >>> np.all(splitext_improved("a.tar") ==  ('a', '.tar'))
    True
    >>> path_with_dirs = os.path.join("a.f", "a.tar")
    >>> path_without_ext = os.path.join("a.f", "a")
    >>> np.all(splitext_improved(path_with_dirs) ==  (path_without_ext, '.tar'))
    True
    >>> path_with_dirs = os.path.join("a.a.a.f", "a.tar.gz")
    >>> path_without_ext = os.path.join("a.a.a.f", "a")
    >>> np.all(splitext_improved(path_with_dirs) ==  (path_without_ext, '.tar.gz'))
    True
    """
    import os

    dir, file = os.path.split(path)

    if len(file.split(".")) > 2:
        froot, ext = file.split(".")[0], "." + ".".join(file.split(".")[-2:])
    else:
        froot, ext = os.path.splitext(file)

    return os.path.join(dir, froot), ext


@njit
def interp_nb(x_vals, x, y):
    return np.interp(x_vals, x, y)


def normalize_dyn_profile(dynprof, norm):
    """Normalize a dynamical profile (e.g. a phaseogram).

    Parameters
    ----------
    dynprof : np.ndarray
        The dynamical profile has to be a 2d array structured as:
        `dynprof = [profile0, profile1, profile2, ...]`
        where each `profileX` is a pulse profile.
    norm : str
        The chosen normalization. If it ends with `_smooth`, a
        simple Gaussian smoothing is applied to the image.
        Besides the smoothing string, the options are:
        1. to1: make each profile normalized between 0 and 1
        2. std: subtract the mean and divide by standard deviation
            in each row
        3. ratios: divide by the average profile (particularly
            useful in energy vs phase plots)
        4. mediansub, meansub: just subtract the median or the mean
            from each profile
        5. mediannorm, meannorm: subtract the median or the norm
            and divide by it to get fractional amplitude

    Examples
    --------
    >>> hist = [[1, 2], [2, 3], [3, 4]]
    >>> hnorm = normalize_dyn_profile(hist, "meansub")
    >>> np.allclose(hnorm[0], [-0.5, 0.5])
    True
    >>> hnorm = normalize_dyn_profile(hist, "meannorm")
    >>> np.allclose(hnorm[0], [-1/3, 1/3])
    True
    >>> hnorm = normalize_dyn_profile(hist, "ratios")
    >>> np.allclose(hnorm[1], [1, 1])
    True
    """
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


@vectorize([(int64,), (float32,), (float64,)])
def phases_from_zero_to_one(phase):
    """Normalize pulse phases from 0 to 1

    Examples
    --------
    >>> phases_from_zero_to_one(0.1)
    0.1
    >>> phases_from_zero_to_one(-0.9)
    0.1
    >>> phases_from_zero_to_one(0.9)
    0.9
    >>> phases_from_zero_to_one(3.1)
    0.1
    >>> assert np.allclose(phases_from_zero_to_one([0.1, 3.1, -0.9]), 0.1)
    True
    """

    return phase - np.floor(phase)


@vectorize([(int64,), (float32,), (float64,)])
def phases_around_zero(phase):
    """Normalize pulse phases from -0.5 to 0.5

    Examples
    --------
    >>> phases_around_zero(0.6)
    -0.4
    >>> phases_around_zero(-0.9)
    0.1
    >>> phases_around_zero(3.9)
    -0.1
    >>> assert np.allclose(phases_from_zero_to_one([0.6, -0.4]), -0.4)
    True
    """
    ph = phase - np.floor(phase)
    while ph >= 0.5:
        ph -= 1.0
    while ph < -0.5:
        ph += 1.0
    return ph


def create_template_from_profile_harm(
    profile,
    imagefile="template.png",
    nharm=None,
    final_nbin=None,
):
    """
    Parameters
    ----------
    phase: :class:`np.array`
    profile: :class:`np.array`
    imagefile: str
    final_nbin: int
    Returns
    -------
    template: :class:`np.array`
        The calculated template
    additional_phase: float
    Examples
    --------
    >>> phase = np.arange(0.005, 1, 0.01)
    >>> profile = np.cos(2 * np.pi * phase)
    >>> profile_err = profile * 0
    >>> template, additional_phase = create_template_from_profile_harm(
    ...     profile)
    ...
    >>> np.allclose(template, profile, atol=0.001)
    True
    """
    import matplotlib.pyplot as plt

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
        # np.abs(np.fft.fft(sine(x, a, b, ph + 0.7)))[1] / x.size * 2

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
        # print(template_fine.size, phases_fine.size)

        templ_func_fine = interp1d(phases_fine, template_fine, kind="cubic", assume_sorted=True)

        additional_phase = (
            np.argmax(template_fine[: final_nbin * oversample_factor])
            / final_nbin
            / oversample_factor
            + dph_fine / 2
        )
        # phases_fine += 0.5 * dph_fine

        def template_func(x):
            return templ_func_fine(1 + x + additional_phase)

        # print(additional_phase)

    dph = 1 / final_nbin
    phas = np.arange(dph / 2, 1, dph)

    template = template_func(phas)

    additional_phase = phases_around_zero(additional_phase)
    template = template[:final_nbin].real

    fig = plt.figure(figsize=(3.5, 2.65))
    plt.plot(np.arange(0.5 / nbin, 1, 1 / nbin), profile, drawstyle="steps-mid", label="data")
    # plt.plot(phases_fine, template_fine, label="template fine")
    plt.plot(phas[:final_nbin], template, label="template values", ls="--", lw=2)
    plt.plot(
        phas[:final_nbin], template_func(phas[:final_nbin]), label="template func", ls=":", lw=2
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


def likelihood(phases, template_func, weights=None):
    probs = template_func(phases)
    if weights is None:
        return np.log(probs).sum()
    else:
        return np.log(weights * probs + 1.0 - weights).sum()


def get_template_func(template):
    """Get a cubic interpolation function of a pulse template.

    Parameters
    ----------
    template : array-like
        The input template profile

    Returns
    -------
    template_fun : function
        This function accepts pulse phases (even not distributed
        between 0 and 1) and returns the corresponding interpolated
        value of the pulse profile)
    """
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


@njit(fastmath=True, parallel=True)
def simple_circular_deorbit_numba(times, PB, A1, TASC, tolerance=1e-8):
    twopi = 2 * np.pi
    omega = twopi / PB
    out_times = np.empty_like(times)
    for i in prange(times.size):
        old_out = 0
        t = times[i] - TASC
        out_times[i] = t - A1 * np.sin(omega * t)
        while np.abs(out_times[i] - old_out) > tolerance:
            old_out = out_times[i]
            out_times[i] = t - A1 * np.sin(omega * out_times[i])
        out_times[i] += TASC

        # out_times[i] = times[i] - A1 * np.sin(omega * (out_times[i] - TASC))
    return out_times


def add_circular_orbit_numba(times, PB, A1, TASC):
    twopi = 2 * np.pi
    omega = twopi / PB
    return times + A1 * np.sin(omega * (times - TASC))


@njit(fastmath=True, parallel=True)
def simple_ell1_deorbit_numba(times, PB, A1, TASC, EPS1, EPS2, tolerance=1e-8):
    twopi = 2 * np.pi
    omega = twopi / PB
    out_times = np.empty_like(times)
    k1 = EPS1 / 2
    k2 = EPS2 / 2
    for i in prange(times.size):
        old_out = 0
        t = times[i] - TASC
        out_times[i] = t - A1 * np.sin(omega * t)
        while np.abs(out_times[i] - old_out) > tolerance:
            old_out = out_times[i]
            phase = omega * out_times[i]
            twophase = 2 * phase
            out_times[i] = t - A1 * (np.sin(phase) + k1 * np.sin(twophase) + k2 * np.cos(twophase))
        out_times[i] += TASC

        # out_times[i] = times[i] - A1 * np.sin(omega * (out_times[i] - TASC))
    return out_times


def add_ell1_orbit_numba(times, PB, A1, TASC, EPS1, EPS2):
    twopi = 2 * np.pi
    omega = twopi / PB
    phase = omega * (times - TASC)
    twophase = 2 * phase
    k1 = EPS1 / 2
    k2 = EPS2 / 2
    return times + A1 * (np.sin(phase) + k1 * np.sin(twophase) + k2 * np.cos(twophase))


def get_flat_samples(sampler):
    tau = sampler.get_autocorr_time(quiet=True)
    maxtau = np.max(tau)
    burnin = int(2 * maxtau)
    thin = int(0.5 * maxtau)
    flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    # log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)
    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(flat_samples.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))
    return flat_samples, maxtau


def calculate_result_array_from_samples(sampler, labels):
    flat_samples, maxtau = get_flat_samples(sampler)
    result_dict = {}
    ndim = flat_samples.shape[1]
    percs = [1, 10, 16, 50, 84, 90, 99]
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], percs)
        for i_p, p in enumerate(percs):
            result_dict[labels[i] + f"_{p:g}"] = mcmc[i_p]

    result_dict["date"] = Time.now().mjd
    result_dict["nsamples"] = flat_samples.shape[0]
    result_dict["maxtau"] = maxtau
    result_dict["burnin"] = maxtau
    result_dict["thin"] = maxtau

    return result_dict, flat_samples


def plot_mcmc_results(
    sampler=None, backend=None, flat_samples=None, labels=None, fname="results.jpg", **plot_kwargs
):
    assert np.any([a is not None for a in [sampler, backend, flat_samples]]), (
        "At least one between backend, sampler, or flat_samples, should be specified, in",
        "increasing order of priority",
    )

    if flat_samples is None:
        if sampler is None:
            assert os.path.exists(backend), "Backend file does not exist"
            sampler = emcee.backends.HDFBackend(backend)
            assert sampler.iteration > 0, "Backend is empty"

        flat_samples, _ = get_flat_samples(sampler)

    fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], **plot_kwargs)
    fig.savefig(fname, dpi=300)


def safe_run_sampler(
    func_to_maximize,
    starting_pars,
    max_n=100_000,
    outroot="chain_results",
    labels=None,
    corner_labels=None,
    n_autocorr=200,
):

    # https://emcee.readthedocs.io/en/stable/tutorials/monitor/?highlight=run_mcmc#saving-monitoring-progress
    # We'll track how the average autocorrelation time estimate changes
    starting_pars = np.asarray(starting_pars)
    ndim = len(starting_pars)

    if labels is None:
        labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))
    if corner_labels is None:
        corner_labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))

    backend_filename = outroot + ".h5"
    backend = emcee.backends.HDFBackend(backend_filename)
    initial_size = 0
    if os.path.exists(backend_filename):
        initial_size = backend.iteration

    print("Initial size: {0}".format(initial_size))
    # backend.reset(nwalkers, ndim)
    if initial_size < 100:
        print("Starting from zero")
        pos = np.array(starting_pars) + np.random.normal(np.zeros((32, starting_pars.size)), 1e-5)
        nwalkers, ndim = pos.shape
        backend.reset(nwalkers, ndim)
    elif initial_size < max_n:
        print("Starting from where we left")
        reader = emcee.backends.HDFBackend(backend_filename)
        samples = reader.get_chain(discard=initial_size // 2, flat=True)

        pos = samples[-32:, :]

        nwalkers, ndim = pos.shape

        max_n = max_n - initial_size
    else:
        reader = emcee.backends.HDFBackend(backend_filename)

        result_dict, flat_samples = calculate_result_array_from_samples(reader, labels)
        print("Nothing to be done here")
        return result_dict

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, func_to_maximize, backend=backend)

    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(pos, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * n_autocorr < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau

    result_dict, flat_samples = calculate_result_array_from_samples(sampler, labels)
    plot_mcmc_results(
        flat_samples=flat_samples, labels=labels, fname=outroot + "_corner.jpg", backend=backend
    )

    return result_dict


def renormalize_results(results, name, result_name, mean, factor):
    """

    Examples
    --------
    >>> results = {"Bu0_mean": 0, "Bu0_ne": 0.1, "Bu0_pe": 0.2}
    >>> mean = 13
    >>> factor = 10
    >>> res = renormalize_results(results, "Bu1", "Bu0", mean, factor)
    >>> np.isclose(res["Bu1"], 13)
    True
    >>> np.isclose(res["Bu1_ne"], 1)
    True
    >>> np.isclose(res["Bu1_pe"], 2)
    True
    """
    value = results[result_name + "_mean"]
    error_n = results[result_name + "_ne"]
    error_p = results[result_name + "_pe"]

    results[name] = value * factor + mean
    results[name + "_ne"] = error_n * factor
    results[name + "_pe"] = error_p * factor

    return results


def _plot_phaseogram(phases, times, ax0, ax1, norm="meansub_smooth"):
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


def _list_zoom_factors(input_fit_par_labels, zoom):
    factors = []
    for par in input_fit_par_labels:
        if zoom is not None and par in zoom:
            factors.append(zoom[par])
        else:
            factors.append(1)
    return factors


def _mjd_to_sec(mjd, mjdref):
    return ((mjd - mjdref) * 86400).astype(float)


def _sec_to_mjd(met, mjdref):
    return met / 86400 + mjdref


@njit(parallel=True)
def _fast_phase_fdot(ts, mean_f, mean_fdot):
    phases = ts * mean_f + 0.5 * ts * ts * mean_fdot
    return phases


ONE_SIXTH = 1 / 6


@njit(parallel=True)
def _fast_phase_fddot(ts, mean_f, mean_fdot, mean_fddot):
    tssq = ts * ts
    phases = ts * mean_f + 0.5 * tssq * mean_fdot + ONE_SIXTH * tssq * ts * mean_fddot
    return phases


@njit(parallel=True)
def _fast_phase(ts, mean_f):
    phases = ts * mean_f
    return phases


@njit(parallel=True)
def _fast_phase_generic(times, frequency_derivatives):
    if len(frequency_derivatives) == 1:
        return times / frequency_derivatives[0]

    fact = 1.0
    n = 0.0
    ph = np.zeros_like(times)

    t_pow = np.ones_like(times)

    for f in frequency_derivatives:
        t_pow *= times
        n += 1
        fact *= n
        ph += (1 / fact * f) * t_pow

    return ph


def fast_phase(times, frequency_derivatives):
    """
    Calculate pulse phase from the frequency and its derivatives.

    Parameters
    ----------
    times : array of floats
        The times at which the phase is calculated
    *frequency_derivatives: floats
        List of derivatives in increasing order, starting from zero.

    Returns
    -------
    phases : array of floats
        The absolute pulse phase

    Examples
    --------
    >>> from stingray.pulse import pulse_phase
    >>> times = np.random.uniform(0, 100000, 100)
    >>> ph1 = fast_phase(times, [0.2123, 1e-5, 1e-9, 1e-15])
    >>> ph2 = pulse_phase(times, 0.2123, 1e-5, 1e-9, 1e-15, ph0=0, to_1=False)
    >>> np.allclose(ph1, ph2)
    True
    """
    if len(frequency_derivatives) == 1:
        return _fast_phase(times, frequency_derivatives[0])
    elif len(frequency_derivatives) == 2:
        return _fast_phase_fdot(times, frequency_derivatives[0], frequency_derivatives[1])
    elif len(frequency_derivatives) == 3:
        return _fast_phase_fddot(
            times, frequency_derivatives[0], frequency_derivatives[1], frequency_derivatives[2]
        )

    return _fast_phase_generic(times, np.array(frequency_derivatives))


def _calculate_phases(times_from_pepoch, pars_dict):

    n_files = len(times_from_pepoch)
    list_phases_from_zero_to_one = []

    for i in range(n_files):
        tasc = _mjd_to_sec(pars_dict["TASC"], pars_dict[f"PEPOCH_{i}"])

        deorbit_times_from_pepoch = simple_ell1_deorbit_numba(
            times_from_pepoch[i],
            pars_dict["PB"],
            pars_dict["A1"],
            tasc,
            pars_dict["EPS1"],
            pars_dict["EPS2"],
        )

        deorbited_pepoch = simple_ell1_deorbit_numba(
            np.array([0.0]),
            pars_dict["PB"],
            pars_dict["A1"],
            tasc,
            pars_dict["EPS1"],
            pars_dict["EPS2"],
        )

        count = 0
        freq_ders = []
        while f"F{count}_{i}" in pars_dict:
            freq_ders.append(pars_dict[f"F{count}_{i}"])
            count += 1

        phase_pepoch = fast_phase(deorbited_pepoch.astype(float), freq_ders)

        phases = (
            pars_dict[f"Phase_{i}"]
            - phase_pepoch
            + fast_phase(deorbit_times_from_pepoch.astype(float), freq_ders)
        )
        list_phases_from_zero_to_one.append(phases_from_zero_to_one(phases))
    return list_phases_from_zero_to_one


def folded_profile(times, parameters, nbin=16):
    n_files = len(times)
    phases = _calculate_phases(times, parameters)
    profile = []
    for i in range(n_files):
        profile.append(np.histogram(phases[i], bins=np.linspace(0, 1, nbin + 1))[0])
    return profile


def _get_par_dict(model):
    parameters = {
        "Phase": 0,
        "PB": model.PB.value.astype(float) * 86400,
        "TASC": _mjd_to_sec(model.TASC.value, model.PEPOCH.value),
        "A1": model.A1.value.astype(float),
        "EPS1": model.EPS1.value.astype(float),
        "EPS2": model.EPS2.value.astype(float),
        "PEPOCH": model.PEPOCH.value.astype(float),  # I added Pepoch
    }

    count = 0
    while hasattr(model, f"F{count}"):
        parameters[f"F{count}"] = getattr(model, f"F{count}").value.astype(float)
        count += 1
    return parameters


def _load_and_format_events(
    event_file, energy_range, pepoch, plotlc=True, plotfile="lightcurve.jpg"
):
    events = load_events(event_file)
    if plotlc:
        lc = events.to_lc(100)

        fig = plt.figure("LC", figsize=(3.5, 2.65))
        plt.plot(_sec_to_mjd(lc.time, events.mjdref), lc.counts / lc.dt)
        GTI = _sec_to_mjd(events.gti, events.mjdref)
        for g0, g1 in zip(GTI[:, 1], GTI[:, 0]):
            plt.axvspan(g0, g1, color="r", alpha=0.5)
        plt.xlabel("MJD")
        plt.ylabel("Count rate")
        plt.savefig(plotfile)
        plt.close(fig)

    if energy_range is not None:
        events.filter_energy_range(energy_range, inplace=True)
    mjdref = events.mjdref
    pepoch_met = _mjd_to_sec(pepoch, mjdref)
    times_from_pepoch = (events.time - pepoch_met).astype(float)
    gtis_from_pepoch = (events.gti - pepoch_met).astype(float)
    return times_from_pepoch, gtis_from_pepoch


def optimize_solution(
    times_from_pepoch,
    model_parameters,
    fit_parameters,
    values,
    bounds,
    factors,
    template_func,
    nsteps=1000,
    minimize_first=False,
    nharm=1,
    outroot="out",
):
    def logprior(pars):
        if np.any(np.isnan(pars)):
            return -np.inf

        for parname, bound, initial, local_value, f in zip(
            fit_parameters, bounds, values, pars, factors
        ):
            value = local_value * f + initial
            # Add correction for PB and TASC

            if not (bound[0] < value < bound[1]):
                print(f"Value out of bound: {parname}={value}")
                return -np.inf

        return 0

    def local_phases(pars):
        allpars = copy.deepcopy(model_parameters)

        for par, initial, value, f in zip(fit_parameters, values, pars, factors):
            allpars[par] = value * f + initial

        return _calculate_phases(times_from_pepoch, allpars)

    def func_to_maximize(pars):
        # print(pars)
        lp = logprior(pars)
        if np.isinf(lp):
            return lp
        phases = local_phases(pars)

        ll = 0
        for i in range(len(phases)):
            ll += likelihood(phases[i], template_func[i])

        return ll + lp

    def func_to_minimize(pars):
        return -func_to_maximize(pars)

    all_zeros = [0] * len(values)
    if minimize_first:
        res = minimize(func_to_minimize, all_zeros)
        fit_pars = res.x
    else:
        fit_pars = all_zeros

    pars_dict = copy.deepcopy(model_parameters)

    for par, initial, value, f in zip(fit_parameters, values, fit_pars, factors):
        pars_dict[par] = value * f + initial

    phases = local_phases(fit_pars)

    rough_results = {}
    for par, value, f in zip(fit_parameters, fit_pars, factors):
        rough_results["rough_d" + par] = value
    for i in range(len(times_from_pepoch)):
        _compare_phaseograms(
            local_phases(all_zeros)[i],
            phases_from_zero_to_one(phases[i]),
            times_from_pepoch[i],
            fname=outroot + f"_{i}.jpg",
        )

    corner_labels = [
        "d" + par + f"{np.log10(fac):+g}" for (par, fac) in zip(fit_parameters, factors)
    ]
    results = safe_run_sampler(
        func_to_maximize,
        fit_pars,
        max_n=nsteps,
        outroot=outroot,
        labels=["d" + par for par in fit_parameters],
        corner_labels=corner_labels,
        n_autocorr=0,
    )

    count = 0
    while f"Phase_{count}" in pars_dict:
        results[f"additional_phase_{count}"] = pars_dict[f"Phase_{count}"]
        count += 1

    results.update(model_parameters)
    results.update(rough_results)
    for par, initial, f in zip(fit_parameters, values, factors):
        results["d" + par + "_mean"] = results["d" + par + "_50"]
        results["d" + par + "_initial"] = initial
        results["d" + par + "_factor"] = f

    fit_pars = [results["d" + par + "_50"] for par in fit_parameters]
    phases = local_phases(fit_pars)

    for i in range(len(times_from_pepoch)):
        _compare_phaseograms(
            local_phases(all_zeros)[i],
            phases_from_zero_to_one(phases[i]),
            times_from_pepoch[i],
            fname=outroot + f"_{i}_final.jpg",
        )

    return results


def create_bounds(parnames):
    bounds = []
    for par in parnames:
        if par.startswith("EPS"):
            bounds.append((-1, 1))
        else:
            bounds.append((-np.inf, np.inf))
    return bounds


def order_of_magnitude(value):
    return 10 ** np.int(np.log10(value) - 1)


def get_factors(parnames, model, observation_length):
    import re

    n_files = len(observation_length)
    freq_re = re.compile(r"^F([0-9]+)_([0-9]+)$")
    zoom = []
    P = model[0].PB.value * 86400
    X = model[0].A1.value

    F = np.max([model[i].F0.value for i in range(n_files)])
    obs_length = np.max(observation_length)

    for par in parnames:
        matchobj = freq_re.match(par)
        if matchobj:
            order = int(matchobj.group(1))
            file_n = int(matchobj.group(2))
            zoom.append(order_of_magnitude(1 / observation_length[file_n] ** (order + 1)))
        elif par == "A1":
            zoom.append(min(1, order_of_magnitude(1 / np.pi / 2 / F)))
        elif par == "PB":
            dp = np.sqrt(3) / (2 * np.pi**2 * F) * P**2 / X / obs_length
            zoom.append(min(1.0, order_of_magnitude(dp)))
        elif par.startswith("EPS"):
            zoom.append(0.001)
        else:
            zoom.append(1.0)
    return zoom


def _format_energy_string(energy_range):
    if energy_range is None:
        return ""
    if energy_range[0] is None and energy_range[1] is None:
        return ""
    lower = "**" if energy_range[0] is None else f"{energy_range[0]:g}"
    upper = "**" if energy_range[1] is None else f"{energy_range[1]:g}"

    return f"_{lower}-{upper}keV"


def main(args=None):
    """Main function called by the `ell1fit` script"""
    import argparse

    description = "Fit an ELL1 model and frequency derivatives to an X-ray " "pulsar observation."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "-p",
        "--parfile",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Input parameter files, one per event file. Must contain a simple ELL1 binary model, "
            "with no orbital derivatives, and a number of spin derivatives (F0, F1, ...). "
            "All other models will be ignored."
        ),
    )
    parser.add_argument("-o", "--outroot", type=str, default=None, help="Root of output file names")
    parser.add_argument(
        "-N",
        "--nharm",
        type=int,
        help="Number of harmonics to describe the pulse profile",
        default=1,
    )
    parser.add_argument(
        "-E",
        "--erange",
        nargs=2,
        type=int,
        help="Energy range",
        default=None,
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        help="Maximum number of MCMC steps",
        default=100_000,
    )
    parser.add_argument(
        "-P",
        "--parameters",
        type=str,
        help="Comma-separated list of parameters to fit",
        default="F0,F1",
    )
    parser.add_argument("--minimize-first", action="store_true", default=False)

    args = parser.parse_args(args)
    files = args.files
    parfile = args.parfile
    n_files = len(files)
    assert len(parfile) == len(
        files
    ), "The number of parameter files must match that of event files."
    model = []
    pepoch = []

    for i in range(n_files):
        model.append(get_model(parfile[i]))
        pepoch.append(model[i].PEPOCH.value)

        if hasattr(model[i], "T0") or model[i].BINARY.value != "ELL1":
            raise ValueError("This script wants an ELL1 model, with TASC, not T0, defined")

        model[i].change_binary_epoch(pepoch[i])

    nsteps = args.nsteps
    nharm = args.nharm
    nbin = max(16, nharm * 8)

    energy_range = args.erange
    energy_str = _format_energy_string(energy_range)
    nharm_str = ""
    if args.nharm > 1:
        nharm_str = f"_N{args.nharm}"

    general_tasc = np.min([mod.TASC.value for mod in model])

    ch_ind = 0

    parameters = _get_par_dict(model[ch_ind])
    parameters["TASC"] = general_tasc

    count = 0
    while f"F{count}" in model[ch_ind]:
        del parameters[f"F{count}"]
        count += 1

    del parameters["PEPOCH"]

    for i in range(n_files):

        count = 0
        while f"F{count}" in _get_par_dict(model[i]):
            parameters[f"F{count}_{i}"] = _get_par_dict(model[i])[f"F{count}"]
            count += 1

        parameters[f"PEPOCH_{i}"] = _get_par_dict(model[i])["PEPOCH"]
        parameters[f"Phase_{i}"] = parameters["Phase"]
        #  I initialized the phases because _calculate_phases calls parameters[f"Phase_{i}"]
    del parameters["Phase"]

    parameter_names = []
    list_parameter_names = args.parameters.split(",")

    for f in parameters:
        if f.startswith("Phase"):
            parameter_names.append(f)
        for g in list_parameter_names:
            if f.startswith(g):
                parameter_names.append(f)

    minimize_first = args.minimize_first

    outroot = args.outroot
    if outroot is None and len(files) == 1:
        outroot = (
            splitext_improved(files[0])[0]
            + "_"
            + "_".join(args.parameters.split(","))
            + energy_str
            + nharm_str
        )
    elif outroot is None:
        outroot = "out" + "_" + "_".join(args.parameters.split(",")) + energy_str + nharm_str

    gtis = [[] for _ in range(n_files)]
    times_from_pepoch = [[] for _ in range(n_files)]
    observation_length = [[] for _ in range(n_files)]
    expo = np.zeros(n_files)
    for i in range(n_files):
        fname = files[i]
        times_from_pepoch[i], gtis[i] = _load_and_format_events(
            fname, energy_range, pepoch[i], plotfile=outroot + f"_lightcurve_{i}.jpg"
        )
        expo[i] += np.sum(np.diff(gtis[i], axis=1))

        observation_length[i] = times_from_pepoch[i][-1] - times_from_pepoch[i][0]

    bounds = create_bounds(parameter_names)
    factors = get_factors(parameter_names, model, observation_length)

    profile = folded_profile(times_from_pepoch, parameters, nbin=nbin)

    template_func = []
    mint = []
    maxt = []
    pulsed_frac = []
    ph0 = []
    try:
        input_mean_fit_pars = [parameters[par] for par in parameter_names]
    except KeyError:
        raise ValueError("One or more parameters are missing from the parameter file")

    for i in range(n_files):
        template, additional_phase = create_template_from_profile_harm(
            profile[i], nharm=nharm, final_nbin=200, imagefile=outroot + "_template.jpg"
        )

        template_func.append(get_template_func(template))
        mint.append(template.min())
        maxt.append(template.max())
        pulsed_frac.append((maxt[i] - mint[i]) / (maxt[i] + mint[i]))

        ph0.append(-phases_around_zero(additional_phase))
        parameters[f"Phase_{i}"] = ph0[i]

        for j, par in enumerate(parameter_names):
            if par == f"Phase_{i}":
                bounds[j] = (ph0[i] - 0.5, ph0[i] + 0.5)
                break

    results = optimize_solution(
        times_from_pepoch,
        parameters,
        parameter_names,
        input_mean_fit_pars,
        bounds,
        factors,
        template_func,
        nsteps=nsteps,
        minimize_first=minimize_first,
        nharm=nharm,
        outroot=outroot,
    )

    for i in range(n_files):
        if hasattr(model[i], "START"):
            results[f"Start_{i}"] = model[i].START.value
        else:
            results[f"Start_{i}"] = times_from_pepoch[i][0] / 86400 + pepoch[i]
        if hasattr(model[i], "STOP"):
            results[f"Stop_{i}"] = model[i].STOP.value
        else:
            results[f"Stop_{i}"] = times_from_pepoch[i][-1] / 86400 + pepoch[i]

    for i in range(n_files):
        results[f"PEPOCH_{i}"] = pepoch[i]

    for i in range(n_files):
        results[f"fname_{i}"] = fname[i]

    results["nharm"] = nharm
    results["emin"] = 0 if energy_range is None else energy_range[0]
    results["emax"] = np.inf if energy_range is None else energy_range[1]
    results["nsteps"] = nsteps

    for i in range(n_files):
        results[f"pf_{i}"] = pulsed_frac[i]

    for i in range(n_files):
        results[f"ctrate_{i}"] = times_from_pepoch[i].size / expo[i]

    results["ell1fit_version"] = version.version

    list_result = []
    for i in range(n_files):
        list_result.append(copy.deepcopy(results))
    keys = list(results.keys())

    results = Table(rows=[results])

    output_file = outroot + "_results.ecsv"

    if os.path.exists(output_file):
        old = Table.read(output_file)
        old.write("old_" + output_file, overwrite=True)
        results = vstack([old, results])

    results.write(output_file, overwrite=True)

    list_parameter_names += [
        "Phase",
        "PEPOCH",
        "Start",
        "Stop",
        "fname",
        "ctrate",
        "pf",
        "additional_phase",
    ]

    list_output_file = []
    for i in range(n_files):
        for par in list_parameter_names:
            for j in range(len(keys)):
                for k in range(n_files):
                    if par + f"_{k}" in keys[j]:
                        if k == i:
                            list_result[i][keys[j].replace(par + f"_{k}", par)] = list_result[i][
                                keys[j]
                            ]
                            # print(keys[j], "--->", keys[j].replace(par + f"_{k}", par))
                        del list_result[i][keys[j]]

        list_result[i] = Table(rows=[list_result[i]])

        list_output_file.append(outroot + f"_results_{i}.ecsv")

        if os.path.exists(list_output_file[i]):
            old = Table.read(list_output_file[i])
            old.write("old_" + list_output_file[i], overwrite=True)
            list_result[i] = vstack([old, list_result[i]])

        list_result[i].write(list_output_file[i], overwrite=True)

    return output_file


# if __name__ == "__main__":
#     import sys

#     fname = sys.argv[1]
#     parfile = sys.argv[2]
#     nsteps=100_000
#     if len(sys.argv) > 3:
#         nsteps = int(sys.argv[3])

#     print(f"Processing {fname}")
#     initial_pars = {"F0": (-np.inf, np.inf), "F1": (-np.inf, np.inf),
# "TASC": (-np.inf, np.inf), "A1": (-np.inf, np.inf), "EPS1": (-1, 1), "EPS2": (-1, 1)}
#     zoom = {"F0": 1.e-5, "F1": 1e-12, "TASC": 1, "A1": 1e-4, "EPS1": 1e-3, "EPS2": 1e-3}

#     minimize_first=True
#     nharm=1
#     energy_range = [3, 30]
#     if "3010" in fname or "80002092004" in fname:
#         energy_range = [8, 30]
#         nharm=2
#     if "80002092006" in fname:
#         initial_pars["F2"] = (-np.inf, np.inf)
#         zoom["F2"] = 1e-17

#     results = optimize_solution(
#         fname, parfile, initial_pars, zoom=zoom, nsteps=nsteps, energy_range=energy_range,
# minimize_first=minimize_first,
#         nharm=2
#     )

#     from scinum import Number

#     results = results[-1]
#     eps1 = eps2 = 0
#     for par in initial_pars.keys():
#         mean = results[par + "_50"] * results[par + "_factor"] + results[par + "_initial"]
#         errp = (results[par + "_84"] - results[par + "_50"]) * results[par + "_factor"]
#         errn = (results[par + "_50"] - results[par + "_16"]) * results[par + "_factor"]
#         if np.abs(errp - errn) / errp < 0.1:
#             errp = errn

#         lim = np.array([results[par + "_1"], results[par + "_99"]]) * results[par + "_factor"] +
# results[par + "_initial"]

#         if par == "TASC":
#             mean = float(results["PEPOCH"] + mean / 86400)
#         elif par == "PB":
#             mean = mean / 86400

#         if par in ("PB", "TASC"):
#             errp = errp / 86400
#             errn = errn / 86400
#             lim = lim / 86400

#         additional_label = ""
#         if lim[0] < 0 and lim[1] > 0:
#             additional_label = rf"({lim[0]:g} \le {par} \le {lim[1]:g})"

#         meas = Number(mean, (errn, errp), additional_label)
#         meas_str = meas.str("pdg", style="latex")
#         print(par, meas_str, additional_label)

#         if par == "EPS1":
#             eps1_uplim = np.max(np.abs(lim))
#         if par == "EPS2":
#             eps2_uplim = np.max(np.abs(lim))

#     print("ECC <", np.sqrt(eps1_uplim **2 + eps2_uplim**2))
