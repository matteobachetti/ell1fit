"""Phase and orbital timing utilities used by ell1fit."""

import warnings

import numpy as np
from numba import float32, float64, int64, njit, prange, vectorize


@njit
def interp_nb(x_vals, x, y):
    """Numba-friendly wrapper around numpy.interp.

    Parameters
    ----------
    x_vals : np.ndarray
        Coordinates where interpolation is evaluated.
    x : np.ndarray
        Monotonic sample coordinates.
    y : np.ndarray
        Sample values at x.

    Returns
    -------
    np.ndarray
        Interpolated values at x_vals.
    """
    return np.interp(x_vals, x, y)


@vectorize([(int64,), (float32,), (float64,)])
def phases_from_zero_to_one(phase):
    """Normalize pulse phases from 0 to 1.

    Examples
    --------
    >>> assert np.isclose(phases_from_zero_to_one(0.1), 0.1)
    >>> assert np.isclose(phases_from_zero_to_one(-0.9), 0.1)
    >>> assert np.isclose(phases_from_zero_to_one(0.9), 0.9)
    >>> assert np.isclose(phases_from_zero_to_one(3.1), 0.1)
    >>> assert np.allclose(phases_from_zero_to_one([0.1, 3.1, -0.9]), 0.1)
    """
    return phase - np.floor(phase)


@vectorize([(int64,), (float32,), (float64,)])
def phases_around_zero(phase):
    """Normalize pulse phases from -0.5 to 0.5.

    Examples
    --------
    >>> assert np.isclose(phases_around_zero(0.6), -0.4)
    >>> assert np.isclose(phases_around_zero(-0.9), 0.1)
    >>> assert np.isclose(phases_around_zero(3.9), -0.1)
    >>> assert np.allclose(phases_around_zero([0.6, -0.4]), -0.4)
    """
    ph = phase - np.floor(phase)
    while ph >= 0.5:
        ph -= 1.0
    while ph < -0.5:
        ph += 1.0
    return ph


@njit(fastmath=True, parallel=True)
def simple_circular_deorbit_numba(times, PB, A1, TASC, tolerance=1e-8):
    """Iteratively remove circular-orbit delays from event times."""
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
    return out_times


def add_circular_orbit_numba(times, PB, A1, TASC):
    """Apply circular-orbit delays to times using a sinusoidal model."""
    twopi = 2 * np.pi
    omega = twopi / PB
    return times + A1 * np.sin(omega * (times - TASC))


@njit(fastmath=True, parallel=True)
def simple_ell1_deorbit_numba(times, PB, A1, TASC, EPS1, EPS2, tolerance=1e-8):
    """Iteratively remove ELL1 orbital delays from event times."""
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
    return out_times


def add_ell1_orbit_numba(times, PB, A1, TASC, EPS1, EPS2):
    """Apply ELL1 orbital delays to times (forward model)."""
    twopi = 2 * np.pi
    omega = twopi / PB
    phase = omega * (times - TASC)
    twophase = 2 * phase
    k1 = EPS1 / 2
    k2 = EPS2 / 2
    return times + A1 * (np.sin(phase) + k1 * np.sin(twophase) + k2 * np.cos(twophase))


def _mjd_to_sec(mjd, mjdref):
    """Convert MJD timestamps to seconds from mjdref."""
    return ((mjd - mjdref) * 86400).astype(float)


def _sec_to_mjd(met, mjdref):
    """Convert seconds from mjdref back to MJD."""
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
    """Calculate pulse phase from the frequency and its derivatives."""
    if len(frequency_derivatives) == 1:
        return _fast_phase(times, frequency_derivatives[0])
    if len(frequency_derivatives) == 2:
        return _fast_phase_fdot(times, frequency_derivatives[0], frequency_derivatives[1])
    if len(frequency_derivatives) == 3:
        return _fast_phase_fddot(
            times,
            frequency_derivatives[0],
            frequency_derivatives[1],
            frequency_derivatives[2],
        )

    return _fast_phase_generic(times, np.array(frequency_derivatives))


def _calculate_phases(times_from_pepoch, pars_dict, tolerance=1e-8):
    """Compute pulse phases for each file given spin and ELL1 orbital parameters."""
    n_files = len(times_from_pepoch)
    list_phases_from_zero_to_one = []
    pb = pars_dict["PB"]
    for i in range(n_files):
        tasc_raw = _mjd_to_sec(pars_dict["TASC"], pars_dict[f"PEPOCH_{i}"])
        tasc = ((tasc_raw + 0.5 * pb) % pb) - 0.5 * pb
        if np.abs(tasc_raw - tasc) > 1e-9:
            warnings.warn("Wrapping TASC to the principal interval modulo PB")

        deorbit_times_from_pepoch = simple_ell1_deorbit_numba(
            times_from_pepoch[i],
            pb,
            pars_dict["A1"],
            tasc,
            pars_dict["EPS1"],
            pars_dict["EPS2"],
            tolerance=tolerance,
        )

        deorbited_pepoch = simple_ell1_deorbit_numba(
            np.array([0.0]),
            pb,
            pars_dict["A1"],
            tasc,
            pars_dict["EPS1"],
            pars_dict["EPS2"],
            tolerance=tolerance,
        )

        count = 0
        freq_ders = []
        while f"F{count}_{i}" in pars_dict:
            freq_ders.append(float(pars_dict[f"F{count}_{i}"]))
            count += 1

        phase_pepoch = fast_phase(deorbited_pepoch.astype(float), freq_ders)

        phases = (
            pars_dict[f"Phase_{i}"]
            - phase_pepoch
            + fast_phase(deorbit_times_from_pepoch.astype(float), freq_ders)
        )
        list_phases_from_zero_to_one.append(phases_from_zero_to_one(phases.astype(float)))
    return list_phases_from_zero_to_one


def folded_profile(times, parameters, weights=None, nbin=16, tolerance=1e-8):
    """Fold events into pulse profiles for one or multiple files."""
    n_files = len(times)
    phases = _calculate_phases(times, parameters, tolerance=tolerance)
    profile = []
    for i in range(n_files):
        if weights is None:
            profile.append(np.histogram(phases[i], bins=np.linspace(0, 1, nbin + 1))[0])
        else:
            profile.append(
                np.histogram(phases[i], bins=np.linspace(0, 1, nbin + 1), weights=weights[i])[0]
            )

    return profile
