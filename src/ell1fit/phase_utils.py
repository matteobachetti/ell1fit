"""Phase and orbital timing utilities used by ell1fit."""

import logging
import re

import numpy as np
from ._numba_compat import float32, float64, int64, njit, prange, vectorize

#: Matches a frequency-derivative parameter name (``F0``, ``F1``, ...),
#: optionally prefixed with ``d`` for the local-coordinate fit variant
#: (``dF0``, ``dF1``, ...).
__all__ = [
    "NonInvertibleOrbitError",
    "_calculate_phases",
    "_mjd_to_sec",
    "_second_order_coefficients",
    "add_circular_orbit_numba",
    "add_ell1_orbit_numba",
    "fast_phase",
    "folded_profile",
    "interp_nb",
    "ELL1_TRUNCATION_COEFFICIENT",
    "ell1_truncation_error",
    "orbit_is_invertible",
    "phases_around_zero",
    "phases_from_zero_to_one",
    "simple_circular_deorbit_numba",
    "simple_ell1_deorbit_numba",
]


simple_freq_re = re.compile(r"^d?F([0-9]+)")

#: Safety cap on the deorbiting fixed-point iteration. Legitimate parameters
#: converge in a handful of passes -- a projected velocity of 1e-3 c needs about
#: five -- so this is never reached in normal use. It exists so that no input can
#: make the loop run forever; see :func:`orbit_is_invertible`.
MAX_DEORBIT_ITERATIONS = 1000


class NonInvertibleOrbitError(ValueError):
    """Raised when orbital parameters make arrival time non-invertible.

    Deorbiting solves ``t_emit = t_obs - A1 * sin(omega * t_emit)`` by fixed-point
    iteration, which converges only while the map is a contraction. See
    :func:`orbit_is_invertible` for what that means physically.
    """


#: Coefficient of the ELL1 truncation law, measured against exact Keplerian
#: orbits: the RMS phase error left by stopping the Roemer delay at ``O(e**2)``
#: is ``ELL1_TRUNCATION_COEFFICIENT * e**3 * A1 * F0`` cycles, after a fit has
#: absorbed what it can into ``A1``, ``TASC``, ``EPS1``, ``EPS2`` and a constant.
#: Measured at 0.2357 and **independent of omega** to four digits across ten
#: values, which is why one number suffices. See ``docs/ell1fit/design.rst``.
ELL1_TRUNCATION_COEFFICIENT = 0.236


def ell1_truncation_error(EPS1, EPS2, A1, F0):
    """Phase error, in cycles, from ELL1 truncating the orbit at ``O(e**2)``.

    ELL1 expands the Roemer delay in eccentricity; this package carries the
    ``O(e**2)`` terms, so what remains scales as :math:`e^3`. The residual lives
    almost entirely in the ``3 Phi`` harmonics, which are orthogonal to every
    direction the model can move in -- so exceeding this costs *sensitivity*,
    through unmodelled structure smearing the profile, rather than biasing the
    recovered eccentricity. At ``e = 0.01`` the recovered ``e`` is biased by
    only 2.7e-5 relative, some three orders of magnitude below its statistical
    error.

    Parameters
    ----------
    EPS1, EPS2 : float
        Laplace-Lagrange parameters; only their quadrature sum matters.
    A1 : float
        Projected semi-major axis, light-seconds.
    F0 : float
        Spin frequency, Hz.

    Returns
    -------
    float
        RMS phase error in cycles. Compare it against the precision the data
        support; :func:`ell1fit.pipeline._warn_on_eccentric_orbit` does.

    Examples
    --------
    >>> round(ell1_truncation_error(1.2e-3, -1.6e-3, 22.215, 7.5), 10)
    3.15e-07
    """
    eccentricity = np.hypot(EPS1, EPS2)
    return ELL1_TRUNCATION_COEFFICIENT * eccentricity**3 * np.abs(A1) * np.abs(F0)


def orbit_is_invertible(PB, A1, EPS1=0.0, EPS2=0.0):
    """Whether arrival time can be inverted for these orbital parameters.

    The deorbiting iteration contracts only while
    ``|A1| * omega * (1 + |EPS1| + |EPS2|) < 1``. That expression has a direct
    physical reading: ``A1`` is in light-seconds and ``omega`` in radians per
    second, so ``A1 * omega`` is the projected orbital velocity **in units of
    c**. The condition is therefore simply that the pulsar's projected motion is
    subluminal -- with the eccentricity terms adding their contribution.

    Beyond it the arrival-time map is not monotonic in emission time, so no
    inverse exists and no iteration scheme can find one. Real pulsars sit around
    ``1e-3``, many orders of magnitude inside the limit; a fit only reaches this
    region when an optimizer or sampler probes a wild trial position.

    Returns
    -------
    bool
        True when deorbiting is well posed.
    """
    if PB == 0 or not np.isfinite(PB) or not np.isfinite(A1):
        return False
    velocity_over_c = np.abs(A1) * 2 * np.pi / np.abs(PB)
    # bool() rather than leaking a numpy scalar: this is a predicate, and its
    # callers and its docstring both promise a plain bool.
    return bool(velocity_over_c * (1 + np.abs(EPS1) + np.abs(EPS2)) < 1.0)


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
def simple_circular_deorbit_numba(
    times, PB, A1, TASC, tolerance=1e-8, max_iter=MAX_DEORBIT_ITERATIONS
):
    """Iteratively remove circular-orbit delays from event times.

    The iteration count is capped so that no input can make this loop run
    forever; see :data:`MAX_DEORBIT_ITERATIONS` and :func:`orbit_is_invertible`.
    """
    twopi = 2 * np.pi
    omega = twopi / PB
    out_times = np.empty_like(times)
    for i in prange(times.size):
        t = times[i] - TASC
        out_times[i] = t - A1 * np.sin(omega * t)
        # Seed the previous value so the first comparison can never be a
        # no-op. A plain sentinel of 0 collides with a legitimate solution of
        # exactly 0 -- reached whenever an event sits precisely at TASC -- and
        # silently skips the loop entirely.
        old_out = out_times[i] + 2 * tolerance + 1.0
        n_iter = 0
        while np.abs(out_times[i] - old_out) > tolerance and n_iter < max_iter:
            old_out = out_times[i]
            out_times[i] = t - A1 * np.sin(omega * out_times[i])
            n_iter += 1
        out_times[i] += TASC
    return out_times


def add_circular_orbit_numba(times, PB, A1, TASC):
    """Apply circular-orbit delays to times using a sinusoidal model."""
    twopi = 2 * np.pi
    omega = twopi / PB
    return times + A1 * np.sin(omega * (times - TASC))


@njit(fastmath=True)
def _second_order_coefficients(EPS1, EPS2):
    """Harmonic coefficients of the Roemer delay including the o(e^2) terms.

    ELL1 as normally written keeps only first order in eccentricity. Wex and
    Zhu's o(e^2) extension -- the ``dre`` expression in tempo's ``bnryell1.f``,
    and what PINT computes -- adds terms in ``Phi`` and ``3 Phi``::

        delay / A1 = a1s sin(Phi)  + a1c cos(Phi)
                   + a2s sin(2Phi) + a2c cos(2Phi)
                   + a3s sin(3Phi) + a3c cos(3Phi)

    Collecting the block into six coefficients lets the kernels evaluate it with
    two transcendental calls instead of five, by getting the multiple angles
    from ``sin(Phi)`` and ``cos(Phi)`` through the standard identities. That
    matters because the deorbiting iteration is transcendental-bound: it is the
    reason the higher-order option is affordable at all.

    Setting ``EPS1 = EPS2 = 0`` gives ``(1, 0, 0, 0, 0, 0)``, a circular orbit.

    Returns
    -------
    tuple of float
        ``(a1s, a1c, a2s, a2c, a3s, a3c)``.
    """
    e1e1, e2e2, e1e2 = EPS1 * EPS1, EPS2 * EPS2, EPS1 * EPS2
    return (
        1.0 - (3.0 * e1e1 + 5.0 * e2e2) / 8.0,
        e1e2 / 4.0,
        EPS2 / 2.0,
        -EPS1 / 2.0,
        -3.0 * (e1e1 - e2e2) / 8.0,
        -3.0 * e1e2 / 4.0,
    )


@njit(fastmath=True)
def _ell1_shape(sin_phase, cos_phase, a1s, a1c, a2s, a2c, a3s, a3c):
    """Roemer delay divided by ``A1``, to second order, from one sine and cosine.

    Multiple angles come from the identities ``sin 2x = 2 sin x cos x``,
    ``cos 2x = 1 - 2 sin^2 x``, ``sin 3x = sin x (3 - 4 sin^2 x)`` and
    ``cos 3x = cos x (4 cos^2 x - 3)``, which are exact.
    """
    ss = sin_phase * sin_phase
    cc = cos_phase * cos_phase
    return (
        a1s * sin_phase
        + a1c * cos_phase
        + a2s * (2.0 * sin_phase * cos_phase)
        + a2c * (1.0 - 2.0 * ss)
        + a3s * (sin_phase * (3.0 - 4.0 * ss))
        + a3c * (cos_phase * (4.0 * cc - 3.0))
    )


@njit(fastmath=True, parallel=True)
def simple_ell1_deorbit_numba(
    times, PB, A1, TASC, EPS1, EPS2, tolerance=1e-8, max_iter=MAX_DEORBIT_ITERATIONS
):
    """Iteratively remove ELL1 orbital delays from event times.

    Solves ``t_emit = t_obs - delay(t_emit)`` by fixed-point iteration, with the
    Roemer delay taken to **second order in eccentricity** -- see
    :func:`_second_order_coefficients` for the expression and its provenance.

    The iteration count is capped so that no input can make this loop run
    forever. Reaching the cap means the parameters are outside the invertible
    region and the returned times are meaningless -- callers should screen with
    :func:`orbit_is_invertible` first, which :func:`_calculate_phases` does.
    That screen is derived from the first-order delay; the second-order terms
    move the contraction bound by ``O(e**2)``, well inside the margin it already
    leaves.
    """
    twopi = 2 * np.pi
    omega = twopi / PB
    out_times = np.empty_like(times)
    a1s, a1c, a2s, a2c, a3s, a3c = _second_order_coefficients(EPS1, EPS2)
    for i in prange(times.size):
        t = times[i] - TASC
        # Circular first guess; the EPS terms are applied inside the loop, so
        # the loop must run at least once or they are never applied at all.
        out_times[i] = t - A1 * np.sin(omega * t)
        # Seeding this to a plain 0 collided with a legitimate solution of
        # exactly 0, reached whenever an event sits precisely at TASC. The
        # comparison was then false on entry, the loop was skipped, and the
        # returned value was missing the whole EPS1 cos(2 phi) term -- which is
        # at its maximum right there. Offsetting guarantees one iteration.
        old_out = out_times[i] + 2 * tolerance + 1.0
        n_iter = 0
        while np.abs(out_times[i] - old_out) > tolerance and n_iter < max_iter:
            old_out = out_times[i]
            phase = omega * out_times[i]
            out_times[i] = t - A1 * _ell1_shape(
                np.sin(phase), np.cos(phase), a1s, a1c, a2s, a2c, a3s, a3c
            )
            n_iter += 1
        out_times[i] += TASC
    return out_times


def add_ell1_orbit_numba(times, PB, A1, TASC, EPS1, EPS2):
    """Apply ELL1 orbital delays to times (forward model).

    Second order in eccentricity; see :func:`_second_order_coefficients`.
    """
    twopi = 2 * np.pi
    omega = twopi / PB
    phase = omega * (times - TASC)
    coefficients = _second_order_coefficients(EPS1, EPS2)
    return times + A1 * _ell1_shape(np.sin(phase), np.cos(phase), *coefficients)


def _mjd_to_sec(mjd, mjdref):
    """Convert MJD timestamps to seconds from ``mjdref``.

    Accepts plain Python floats as well as numpy values. The previous
    implementation called ``.astype`` on the result, which works only for numpy
    types and raised ``AttributeError`` on a float -- something that never
    surfaced in normal use because PINT hands back ``np.float64``, but which bit
    immediately when driving the pipeline from a hand-built parameter dict.
    """
    return np.asarray((mjd - mjdref) * 86400, dtype=float)[()]


def _sec_to_mjd(met, mjdref):
    """Convert seconds from mjdref back to MJD."""
    return met / 86400 + mjdref


@njit(parallel=True)
def _fast_phase_fdot(ts, mean_f, mean_fdot):
    """Spin phase from frequency and its first derivative.

    Specialised rather than deferring to :func:`_fast_phase_generic` because
    this is the common case and the general version costs an extra array
    multiply per term. See :func:`fast_phase` for the dispatch.
    """
    phases = ts * mean_f + 0.5 * ts * ts * mean_fdot
    return phases


ONE_SIXTH = 1 / 6


@njit(parallel=True)
def _fast_phase_fddot(ts, mean_f, mean_fdot, mean_fddot):
    """Spin phase from frequency and its first two derivatives."""
    tssq = ts * ts
    phases = ts * mean_f + 0.5 * tssq * mean_fdot + ONE_SIXTH * tssq * ts * mean_fddot
    return phases


@njit(parallel=True)
def _fast_phase(ts, mean_f):
    """Spin phase from frequency alone, for a model with no derivatives."""
    phases = ts * mean_f
    return phases


@njit(parallel=True)
def _fast_phase_generic(times, frequency_derivatives):
    """Spin phase from an arbitrary number of frequency derivatives.

    Evaluates the Taylor series ``sum_k F_k t^(k+1) / (k+1)!`` by accumulating
    the running power of ``t`` and the running factorial, so each term costs one
    multiply rather than a fresh ``t**k``.

    The dominant term is ``F0 * t``, which for a long baseline can reach 1e8
    cycles or more; float64 carries about 16 significant digits, leaving roughly
    1e-8 cycles of resolution there. Measured against an 80-bit reference the
    error is 1.6e-10 cycles over a 100 ks observation and 4.3e-8 over a year --
    comfortably below the ~1e-3 cycle precision a fit achieves, but the reason
    times are kept relative to each file's own ``PEPOCH`` rather than a common
    epoch.
    """
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


def _calculate_phases(times_from_pepoch, parameters, tolerance=1e-8):
    """Compute pulse phases for each file given spin and ELL1 orbital parameters."""
    n_files = len(times_from_pepoch)
    list_phases_from_zero_to_one = []
    pb = parameters["PB"]

    if not orbit_is_invertible(pb, parameters["A1"], parameters["EPS1"], parameters["EPS2"]):
        raise NonInvertibleOrbitError(
            f"Orbital parameters imply a projected velocity of "
            f"{np.abs(parameters['A1']) * 2 * np.pi / np.abs(pb):.4g} c "
            f"(PB={pb!r} s, A1={parameters['A1']!r} lt-s): arrival time is not "
            "invertible, so pulse phases are undefined here."
        )

    for i in range(n_files):
        # The binary is global -- one PB, TASC, A1, EPS1, EPS2 -- but those
        # values are referenced to one epoch, and orbital derivatives carry
        # them away from it. Each file adds a fixed offset that makes them
        # valid at its own PEPOCH. The offsets are exactly zero, so this loop
        # is unchanged, whenever the model sets no orbital derivative; direct
        # callers building a parameter dictionary by hand need not supply them.
        # See :func:`ell1fit.models._orbital_epoch_offsets`.
        pb_i = pb + parameters.get(f"PB_offset_{i}", 0.0)
        a1_i = parameters["A1"] + parameters.get(f"A1_offset_{i}", 0.0)
        eps1_i = parameters["EPS1"] + parameters.get(f"EPS1_offset_{i}", 0.0)
        eps2_i = parameters["EPS2"] + parameters.get(f"EPS2_offset_{i}", 0.0)

        tasc_raw = _mjd_to_sec(
            parameters["TASC"] + parameters.get(f"TASC_offset_{i}", 0.0),
            parameters[f"PEPOCH_{i}"],
        )
        tasc = ((tasc_raw + 0.5 * pb_i) % pb_i) - 0.5 * pb_i
        if np.abs(tasc_raw - tasc) > 1e-9:
            # Normal operation, not an anomaly: TASC is only defined modulo PB,
            # and it lands more than half an orbit from PEPOCH whenever the two
            # epochs were not deliberately aligned. Logged rather than warned
            # because this fires on essentially every call.
            logging.info("Wrapping TASC to the principal interval modulo PB")

        deorbit_times_from_pepoch = simple_ell1_deorbit_numba(
            times_from_pepoch[i],
            pb_i,
            a1_i,
            tasc,
            eps1_i,
            eps2_i,
            tolerance=tolerance,
        )

        deorbited_pepoch = simple_ell1_deorbit_numba(
            np.array([0.0]),
            pb_i,
            a1_i,
            tasc,
            eps1_i,
            eps2_i,
            tolerance=tolerance,
        )

        count = 0
        freq_ders = []
        while f"F{count}_{i}" in parameters:
            freq_ders.append(float(parameters[f"F{count}_{i}"]))
            count += 1

        phase_pepoch = fast_phase(deorbited_pepoch.astype(float), freq_ders)

        phases = (
            parameters[f"Phase_{i}"]
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
