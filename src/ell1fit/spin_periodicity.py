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

Coherence
    One sinusoid is fitted across the whole baseline, so a detection requires
    the period to be stable across it: for points ``T`` apart to stay in phase,
    the period must be steady to about ``P**2 / T``.
"""

import fnmatch

import numpy as np

__all__ = [
    "floating_mean_periodogram",
    "rate_values",
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


def _max_power(t, y, var, freq):
    return floating_mean_periodogram(t, y, var, freq)[0].max()


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
        Injections per trial amplitude.
    detection_fraction : float
        Fraction of injections that must be detected at the reported amplitude.

    Returns
    -------
    dict
        JSON-ready summary; also the arrays ``periods_days``, ``power`` and
        ``window`` and ``best_coeffs`` ``(c0, a, b)``, for plotting.
    """
    rng = np.random.default_rng(rng)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    var = None if var is None else np.asarray(var, dtype=float)
    if not 0 < min_period < max_period:
        raise ValueError("Need 0 < min_period < max_period")
    baseline = np.ptp(t)
    df = 1 / (oversample * max(baseline, max_period))
    freq = np.arange(1 / max_period, 1 / min_period + df, df)
    freq = freq[freq <= 1 / min_period]

    power, coeffs = floating_mean_periodogram(t, y, var, freq)
    best = int(np.argmax(power))

    def shuffled():
        order = rng.permutation(t.size)
        return y[order], None if var is None else var[order]

    max_shuffled = np.array([_max_power(t, *shuffled(), freq) for _ in range(n_shuffle)])
    fap = (np.sum(max_shuffled >= power[best]) + 1) / (n_shuffle + 1)
    threshold = float(np.quantile(max_shuffled, 1 - fap_level))

    amplitudes = np.geomspace(0.05, 20, 25) * np.std(y)
    detected = []
    for amplitude in amplitudes:
        hits = 0
        for _ in range(n_inject):
            y_sh, var_sh = shuffled()
            period = rng.uniform(min_period, max_period)
            signal = amplitude * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
            hits += _max_power(t, y_sh + signal, var_sh, freq) >= threshold
        detected.append(hits / n_inject)
    detected = np.array(detected)
    reached = np.nonzero(detected >= detection_fraction)[0]
    detectable = float(amplitudes[reached[0]]) if reached.size else float("inf")

    return {
        "n": int(t.size),
        "min_period_days": float(min_period),
        "max_period_days": float(max_period),
        "n_frequencies": int(freq.size),
        "best_period_days": float(1 / freq[best]),
        "best_power": float(power[best]),
        "best_amplitude": float(np.hypot(*coeffs[best, 1:])),
        "fap": float(fap),
        "n_shuffle": int(n_shuffle),
        "fap_level": float(fap_level),
        "power_threshold": threshold,
        "detectable_amplitude": detectable,
        "detection_fraction": float(detection_fraction),
        "periods_days": 1 / freq,
        "power": power,
        "window": spectral_window(t, freq),
        "best_coeffs": coeffs[best],
    }
