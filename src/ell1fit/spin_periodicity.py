"""Search a spin-history quantity for a coherent periodicity, for ``ell1stat``.

The quantities searched are the local ``F1`` (a proxy for the accretion
torque) and the pulsed count rate (a proxy for the pulsed luminosity). Both are
sampled at a handful of epochs, usually clustered in campaigns years apart, so
the usual analytic false-alarm formulas -- which assume white noise and
reasonably even sampling -- do not apply. Everything here is calibrated on the
data's own sampling instead.

Periodogram
    A sinusoid plus a free constant fitted at each trial frequency (the
    "floating-mean" or generalised Lomb-Scargle periodogram), with power
    ``1 - chi2(f) / chi2_const``. It is the same number as astropy's
    ``LombScargle(fit_mean=True).power()``, computed here for all frequencies
    at once because the calibration below needs thousands of periodograms and
    the fitted amplitudes.

False-alarm probability
    The observation times are kept and the measured values shuffled among
    them. The fraction of shuffles whose highest peak in the searched range
    beats the real one is the false-alarm probability: the chance that data
    with no periodicity, sampled exactly like these, would produce as high a
    peak somewhere in the range. It makes no assumption about the noise
    distribution, but it does assume the values are exchangeable between
    epochs -- it knows nothing of slower trends.

Detectable amplitude
    Sinusoids of increasing semi-amplitude, at random periods within the range
    and random phases, are added to shuffled data; the amplitude detected (peak
    above the false-alarm threshold) in 90% of trials is reported. With sparse
    data, this is usually the more informative number.

Joint search
    Local ``F1`` and pulsed rate can share a period
    (:func:`search_joint_periodicity`): each keeps its own amplitude and phase,
    so a lag between torque and luminosity is allowed and measured. The
    shuffles keep each epoch's pair of values together; shuffling them
    independently would make every coincidence of peaks look significant
    whenever the two are correlated epoch by epoch, as a torque-luminosity
    relation makes them.

Coherence
    One sinusoid is fitted across the whole baseline, so a detection requires
    the period to be stable across it: for points ``T`` apart to stay in phase,
    the period must be steady to about ``P**2 / T``.
"""

import fnmatch

import numpy as np

__all__ = [
    "floating_mean_periodogram",
    "frequency_grid",
    "leave_one_out",
    "rate_values",
    "search_joint_periodicity",
    "search_periodicity",
    "spectral_window",
]


def floating_mean_periodogram(t, y, var, freq):
    """Power and best-fit coefficients of ``c0 + a cos(2 pi f t) + b sin(2 pi f t)``.

    Parameters
    ----------
    t, y, var : array-like
        Times, values and variances (``None`` for equal weights).
    freq : array-like
        Trial frequencies, in inverse units of ``t``.

    Returns
    -------
    power : ndarray
        ``1 - chi2(f) / chi2_const``, between 0 and 1.
    coeffs : ndarray, shape (len(freq), 3)
        ``(c0, a, b)`` at each frequency.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(t) if var is None else 1 / np.asarray(var, dtype=float)
    arg = 2 * np.pi * np.outer(freq, t)
    basis = np.stack([np.ones_like(arg), np.cos(arg), np.sin(arg)], axis=1)  # (nf, 3, n)
    normal = np.einsum("fin,fjn,n->fij", basis, basis, w)
    rhs = np.einsum("fin,n->fi", basis, w * y)
    coeffs = np.linalg.solve(normal, rhs[..., None])[..., 0]
    chi2 = np.sum(w * y**2) - np.einsum("fi,fi->f", coeffs, rhs)
    mean = np.sum(w * y) / np.sum(w)
    chi2_const = np.sum(w * (y - mean) ** 2)
    return 1 - chi2 / chi2_const, coeffs


def spectral_window(t, freq):
    """``|sum exp(2 pi i f t)|^2 / N^2``: 1 where the sampling alone repeats at ``f``."""
    phase = 2 * np.pi * np.outer(freq, np.asarray(t, dtype=float))
    return (np.cos(phase).sum(axis=1) ** 2 + np.sin(phase).sum(axis=1) ** 2) / len(t) ** 2


def rate_values(table, exclude=()):
    """Pulsed rates with epochs whose ``file`` or ``label`` matches any of the
    shell-style ``exclude`` patterns set to NaN; the table itself is untouched."""
    rates = np.array(table["pulsed_rate"], dtype=float)
    for i, (fname, label) in enumerate(zip(table["file"], table["label"])):
        if any(fnmatch.fnmatch(str(fname), p) or fnmatch.fnmatch(str(label), p) for p in exclude):
            rates[i] = np.nan
    return rates


def frequency_grid(t, min_period, max_period, oversample=5):
    """Trial frequencies (1/day) from ``1/max_period`` to ``1/min_period``, spaced
    ``1 / (oversample * baseline)``."""
    if not 0 < min_period < max_period:
        raise ValueError("Need 0 < min_period < max_period")
    df = 1 / (oversample * max(np.ptp(t), max_period))
    freq = np.arange(1 / max_period, 1 / min_period + df, df)
    return freq[freq <= 1 / min_period]


def _combined_power(t, ys, variances, freq, order=None):
    """Mean of the quantities' periodogram powers, and each one's coefficients.

    ``order`` permutes every quantity with the *same* permutation, so values
    measured at one epoch stay together.
    """
    powers, coeffs = [], []
    for y, var in zip(ys, variances):
        if order is not None:
            y = y[order]
            var = None if var is None else var[order]
        power, coeff = floating_mean_periodogram(t, y, var, freq)
        powers.append(power)
        coeffs.append(coeff)
    return np.mean(powers, axis=0), powers, coeffs


def _calibrated_search(t, ys, variances, freq, n_shuffle, fap_level, rng):
    """Combined periodogram, its best peak, and the shuffled-data false-alarm
    probability and detection threshold for that peak."""
    power, powers, coeffs = _combined_power(t, ys, variances, freq)
    best = int(np.argmax(power))
    max_shuffled = np.array(
        [
            _combined_power(t, ys, variances, freq, order=rng.permutation(t.size))[0].max()
            for _ in range(n_shuffle)
        ]
    )
    fap = (np.sum(max_shuffled >= power[best]) + 1) / (n_shuffle + 1)
    threshold = float(np.quantile(max_shuffled, 1 - fap_level))
    return power, powers, coeffs, best, fap, threshold


def _as_arrays(t, ys, variances):
    t = np.asarray(t, dtype=float)
    ys = [np.asarray(y, dtype=float) for y in ys]
    variances = [None if v is None else np.asarray(v, dtype=float) for v in variances]
    return t, ys, variances


def _search_summary(
    t, freq, power, best, fap, threshold, n_shuffle, fap_level, min_period, max_period
):
    return {
        "n": int(t.size),
        "min_period_days": float(min_period),
        "max_period_days": float(max_period),
        "n_frequencies": int(freq.size),
        "best_period_days": float(1 / freq[best]),
        "best_power": float(power[best]),
        "fap": float(fap),
        "n_shuffle": int(n_shuffle),
        "fap_level": float(fap_level),
        "power_threshold": threshold,
        "periods_days": 1 / freq,
        "power": power,
        "window": spectral_window(t, freq),
    }


def _phase_of_maximum(coeffs, freq):
    """Phase (cycles, in [0, 1)) at which ``a cos(2 pi f t) + b sin(2 pi f t)`` peaks,
    counted from ``t = 0``."""
    return (np.arctan2(coeffs[2], coeffs[1]) / (2 * np.pi)) % 1


def search_periodicity(
    t,
    y,
    var,
    min_period,
    max_period,
    oversample=5,
    n_shuffle=1000,
    fap_level=0.01,
    n_inject=100,
    detection_fraction=0.9,
    rng=None,
):
    """Periodogram of ``y(t)`` between two periods, calibrated by shuffling.

    Parameters
    ----------
    t : array-like
        Times, in days.
    y, var : array-like
        Values and variances (``var=None`` for equal weights).
    min_period, max_period : float
        Search range, in days.
    oversample : int
        Frequency steps per ``1 / baseline``.
    n_shuffle : int
        Shuffles for the false-alarm probability and the detection threshold.
    fap_level : float
        False-alarm probability defining a detection, for the detectable amplitude.
    n_inject : int
        Injections per trial amplitude; 0 skips the detectable amplitude (NaN).
    detection_fraction : float
        Fraction of injections that must be detected at the reported amplitude.

    Returns
    -------
    dict
        JSON-ready summary; also the arrays ``periods_days``, ``power`` and
        ``window`` and ``best_coeffs`` ``(c0, a, b)``, for plotting.
    """
    rng = np.random.default_rng(rng)
    t, (y,), (var,) = _as_arrays(t, [y], [var])
    freq = frequency_grid(t, min_period, max_period, oversample)
    power, _, (coeffs,), best, fap, threshold = _calibrated_search(
        t, [y], [var], freq, n_shuffle, fap_level, rng
    )

    detectable = float("nan")
    if n_inject > 0:
        amplitudes = np.geomspace(0.05, 20, 25) * np.std(y)
        detected = []
        for amplitude in amplitudes:
            hits = 0
            for _ in range(n_inject):
                order = rng.permutation(t.size)
                period = rng.uniform(min_period, max_period)
                signal = amplitude * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
                var_sh = None if var is None else var[order]
                power_sh = floating_mean_periodogram(t, y[order] + signal, var_sh, freq)[0]
                hits += power_sh.max() >= threshold
            detected.append(hits / n_inject)
        reached = np.nonzero(np.array(detected) >= detection_fraction)[0]
        detectable = float(amplitudes[reached[0]]) if reached.size else float("inf")

    result = _search_summary(
        t, freq, power, best, fap, threshold, n_shuffle, fap_level, min_period, max_period
    )
    result.update(
        best_amplitude=float(np.hypot(*coeffs[best, 1:])),
        detectable_amplitude=detectable,
        detection_fraction=float(detection_fraction),
        best_coeffs=coeffs[best],
    )
    return result


def search_joint_periodicity(
    t, ys, variances, min_period, max_period, oversample=5, n_shuffle=1000, fap_level=0.01, rng=None
):
    """One period shared by several quantities measured at the same epochs.

    Each quantity keeps its own mean, amplitude and phase; the joint power is
    the mean of their periodogram powers. The shuffles move all of an epoch's
    values together, so a correlation between the quantities at each epoch --
    which makes their periodogram peaks coincide whether or not anything is
    periodic -- is part of the null hypothesis, not mistaken for a signal.

    Parameters
    ----------
    t : array-like
        Times, in days, shared by all quantities.
    ys, variances : list of array-like
        Values and variances (``None`` for equal weights) of each quantity.

    Returns
    -------
    dict
        As :func:`search_periodicity` (without the detectable amplitude),
        plus ``best_amplitudes`` and ``phases_of_maximum`` (cycles from
        ``t = 0``) per quantity, ``phase_lags`` of each quantity's maximum
        after the first one's, in cycles within [-0.5, 0.5), and the arrays
        ``powers`` and ``best_coeffs`` per quantity.
    """
    rng = np.random.default_rng(rng)
    t, ys, variances = _as_arrays(t, ys, variances)
    freq = frequency_grid(t, min_period, max_period, oversample)
    power, powers, coeffs, best, fap, threshold = _calibrated_search(
        t, ys, variances, freq, n_shuffle, fap_level, rng
    )
    phases = [float(_phase_of_maximum(c[best], freq[best])) for c in coeffs]
    result = _search_summary(
        t, freq, power, best, fap, threshold, n_shuffle, fap_level, min_period, max_period
    )
    result.update(
        best_amplitudes=[float(np.hypot(*c[best, 1:])) for c in coeffs],
        phases_of_maximum=phases,
        phase_lags=[float((p - phases[0] + 0.5) % 1 - 0.5) for p in phases],
        powers=powers,
        best_coeffs=[c[best] for c in coeffs],
    )
    return result


def leave_one_out(t, ys, variances, min_period, max_period, oversample=5, n_shuffle=1000, rng=None):
    """Repeat a (single or joint) search with each epoch left out in turn.

    A candidate period carried by one or two epochs moves, or loses its
    significance, when they are dropped; one supported by the whole dataset
    does neither.

    Parameters
    ----------
    t, ys, variances
        As :func:`search_joint_periodicity`; a single quantity is a one-element list.

    Returns
    -------
    list of dict
        One per epoch: ``dropped_mjd``, ``best_period_days``, ``best_power``, ``fap``.
    """
    rng = np.random.default_rng(rng)
    t, ys, variances = _as_arrays(t, ys, variances)
    rows = []
    for i in range(t.size):
        keep = np.arange(t.size) != i
        freq = frequency_grid(t[keep], min_period, max_period, oversample)
        power, _, _, best, fap, _ = _calibrated_search(
            t[keep],
            [y[keep] for y in ys],
            [None if v is None else v[keep] for v in variances],
            freq,
            n_shuffle,
            0.5,
            rng,
        )
        rows.append(
            {
                "dropped_mjd": float(t[i]),
                "best_period_days": float(1 / freq[best]),
                "best_power": float(power[best]),
                "fap": float(fap),
            }
        )
    return rows
