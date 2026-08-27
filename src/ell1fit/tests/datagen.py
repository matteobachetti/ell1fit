"""Synthetic multi-epoch ELL1 pulsar event generator, for testing.

This module produces event lists and parameter files from a *known* injected
timing solution, so tests can assert that the pipeline recovers the truth
rather than merely reproducing whatever it produced last time.

Independence from the code under test
-------------------------------------
The forward model here is written in plain numpy and deliberately does **not**
import ``ell1fit.phase_utils``. If the generator called the package's own
kernels, a sign error or a convention mismatch shared by both would cancel out
and the recovery tests would happily confirm a wrong answer. The two
implementations agreeing is the signal we want; sharing code would destroy it.

Multi-epoch by construction
---------------------------
Every epoch gets its own ``PEPOCH``, its own ``TASC`` (shifted by a whole
number of orbits to sit near that epoch), and its own ``F0`` (propagated
through ``F1``). This exercises the per-file ``F0_i``/``PEPOCH_i``/``Phase_i``
bookkeeping, the modulo-``PB`` wrapping of ``TASC``, and the reference-model
epoch alignment -- none of which a single-epoch dataset can reach.

Conventions
-----------
Matching the package: ``PB`` is handled in seconds internally but written to
parfiles in days; ``A1`` is in light-seconds; ``TASC`` and ``PEPOCH`` are MJD;
orbital phase is measured from ``TASC``; spin phase is measured from
``PEPOCH``; and the pulsed fraction is ``(max - min) / (max + min)`` of the
profile, the same definition used in :mod:`ell1fit.pipeline`.
"""

import dataclasses

import numpy as np

SEC_PER_DAY = 86400.0


@dataclasses.dataclass(frozen=True)
class InjectedSolution:
    """A ground-truth timing solution to generate events from.

    Attributes
    ----------
    F0, F1 : float
        Spin frequency (Hz) and its derivative (Hz/s), defined at ``pepoch_ref``.
    PB : float
        Orbital period, in days.
    A1 : float
        Projected semi-major axis, in light-seconds.
    TASC : float
        Time of ascending node, MJD.
    EPS1, EPS2 : float
        ELL1 Laplace-Lagrange eccentricity parameters, ``e sin(omega)`` and
        ``e cos(omega)``. The defaults give ``e = 2.0e-3`` at
        ``omega = 143.1 deg``, chosen so that the eccentricity is a firm
        detection rather than a marginal one. The default fixture reaches
        ``sigma(EPS) = 1.45e-4``, which makes this a **12.5 sigma** measurement
        where the previous ``e = 2.6e-4`` was 1.8 sigma -- too weak for anything
        to tell a correct eccentricity model from a broken one. It is still an
        order of magnitude inside the range where the second-order expansion is
        faithful: the truncation error here is 3e-7 cycles, against a limit of
        ``e = 2.9e-2`` at 1e-3 cycles.
    PBDOT : float
        Rate of change of the orbital period, dimensionless (s/s). Zero by
        default, in which case every quantity below reduces exactly to its
        constant-period form.
    A1DOT : float
        Rate of change of the projected semi-major axis, in light-seconds per
        second. Zero by default. Like ``PBDOT`` it is an *epoch-local*
        quantity here: the ``A1`` written into each epoch's parfile is the one
        in force at that epoch's ``TASC`` alias, so a fit that ignores the
        drift sees a different ``A1`` at each epoch, which is exactly the
        signal ``A1DOT`` is measured from.
    exact_kepler : bool
        Generate arrival times from an *exact* Keplerian orbit rather than from
        the ELL1 first-order expansion. ``EPS1``/``EPS2`` still define the
        truth, by way of ``e`` and ``omega``; what changes is that the orbit is
        no longer truncated. This is the Blandford-Teukolsky description that
        ELL1 approximates, and fitting data generated this way is the only test
        here that can see the truncation at all -- see
        :func:`kepler_orbital_delay`.
    pepoch_ref : float
        Reference epoch (MJD) at which ``F0``/``F1`` are defined.
    """

    F0: float = 7.5
    F1: float = -3.2e-13
    PB: float = 2.532971
    A1: float = 22.215
    TASC: float = 56682.0669
    EPS1: float = 1.2e-3
    EPS2: float = -1.6e-3
    PBDOT: float = 0.0
    A1DOT: float = 0.0
    exact_kepler: bool = False
    pepoch_ref: float = 56682.0

    @property
    def ECC(self):
        """Orbital eccentricity, ``sqrt(EPS1**2 + EPS2**2)``."""
        return float(np.hypot(self.EPS1, self.EPS2))

    @property
    def OM(self):
        """Longitude of periastron in radians, ``atan2(EPS1, EPS2)``."""
        return float(np.arctan2(self.EPS1, self.EPS2))

    @property
    def PB_sec(self):
        """Orbital period in seconds."""
        return self.PB * SEC_PER_DAY

    def spin_at(self, pepoch):
        """Propagate ``F0``/``F1`` from ``pepoch_ref`` to ``pepoch``.

        Returns
        -------
        tuple of float
            ``(F0, F1)`` valid at ``pepoch``.
        """
        dt = (pepoch - self.pepoch_ref) * SEC_PER_DAY
        return self.F0 + self.F1 * dt, self.F1

    def tasc_near(self, pepoch):
        """Return the ``TASC`` whole-orbit alias closest to ``pepoch``.

        With ``PBDOT = 0`` this is the same physical solution, but it produces a
        genuinely different ``TASC`` value per epoch -- which is the point.

        A nonzero ``PBDOT`` makes the orbit count quadratic in time,
        :math:`N = x - \\tfrac{1}{2}\\dot{P_b}x^2` with :math:`x = \\Delta t/P_b`,
        so the ascending nodes are no longer evenly spaced and the alias is
        ``n * PB + n**2 * PB * PBDOT / 2`` from the reference. The expression
        reduces exactly to ``n * PB`` when ``PBDOT`` is zero.
        """
        x = (pepoch - self.TASC) / self.PB
        n_orbits = np.round(x - 0.5 * self.PBDOT * x**2)
        return self.TASC + n_orbits * self.PB + n_orbits**2 * self.PB * self.PBDOT / 2

    def A1_near(self, pepoch):
        """Projected semi-major axis, in light-seconds, at the alias near ``pepoch``.

        The same rule as :meth:`PB_sec_near`: a parfile's ``A1`` is the one in
        force at the ``TASC`` it quotes, so the drift is measured from the
        reference ``TASC`` to the nearest ascending node, not to ``pepoch``
        itself. The two differ by up to half an orbit, which at any credible
        ``A1DOT`` is far below the precision of a fit -- but matching the
        convention keeps the generator and PINT describing the same solution
        rather than two that agree only approximately.
        """
        elapsed = (self.tasc_near(pepoch) - self.TASC) * SEC_PER_DAY
        return self.A1 + self.A1DOT * elapsed

    def PB_sec_near(self, pepoch):
        """Orbital period, in seconds, valid at the alias near ``pepoch``.

        The period a parfile quotes is the one in force at the ``TASC`` it
        quotes, so an epoch-local ``TASC`` has to be written with an
        epoch-local ``PB``.
        """
        elapsed = (self.tasc_near(pepoch) - self.TASC) * SEC_PER_DAY
        return self.PB_sec + self.PBDOT * elapsed


def orbital_delay(t_from_tasc_sec, solution, pb_sec=None, a1=None):
    """Roemer delay of the ELL1 model, in seconds, to second order in e.

    This is the *forward* model: given a time in the pulsar frame, it returns
    the delay to add to obtain the observed arrival time.

    Parameters
    ----------
    t_from_tasc_sec : np.ndarray
        Pulsar-frame time measured from ``TASC``, in seconds.
    solution : InjectedSolution
        The injected truth.
    a1 : float or None, optional
        Projected semi-major axis in force at that ``TASC``, in light-seconds.
        Defaults to the reference value; supplying the epoch-local one is what
        makes a nonzero ``A1DOT`` come out right.
    pb_sec : float or None, optional
        Orbital period in force at that ``TASC``. Defaults to the reference
        period. Supplying the epoch-local one is what makes a nonzero
        ``PBDOT`` come out right: measured from the nearest ascending node,
        the neglected quadratic term is under ``1e-10`` orbits across an
        observation, so the local period *is* the exact model there.

    Returns
    -------
    np.ndarray
        Delay in seconds.
    """
    if pb_sec is None:
        pb_sec = solution.PB_sec
    if a1 is None:
        a1 = solution.A1
    phase = 2 * np.pi * t_from_tasc_sec / pb_sec
    # tempo's bnryell1.f `dre`, with EPS1 = e sin(omega), EPS2 = e cos(omega):
    # first order plus the Wex-Zhu o(e^2) block, matching the order the package
    # computes. Written out as published rather than copied from phase_utils --
    # the package once paired these terms the other way round, and a generator
    # that shared the mistake confirmed it instead of catching it. Note this is
    # the literal harmonic form, where phase_utils folds the same expression
    # into six coefficients so it needs only one sine and one cosine.
    e1, e2 = solution.EPS1, solution.EPS2
    return a1 * (
        np.sin(phase)
        - 0.5 * (e1 * np.cos(2 * phase) - e2 * np.sin(2 * phase))
        - (1 / 8)
        * (
            -2 * e1 * e2 * np.cos(phase)
            + 6 * e1 * e2 * np.cos(3 * phase)
            + 3 * e1 * e1 * np.sin(phase)
            + 5 * e2 * e2 * np.sin(phase)
            + 3 * e1 * e1 * np.sin(3 * phase)
            - 3 * e2 * e2 * np.sin(3 * phase)
        )
    )


def kepler_orbital_delay(t_from_tasc_sec, solution, pb_sec=None, a1=None):
    """Roemer delay of an *exact* Keplerian orbit, in seconds.

    ELL1 is a first-order expansion of this. Generating events from the exact
    orbit and fitting them with ELL1 is therefore the only way to see the
    truncation error, and the only way to check that ``EPS1``/``EPS2`` come back
    at ``e sin(omega)`` and ``e cos(omega)`` rather than at whatever the
    expansion happens to be self-consistent with.

    Kepler's equation ``M = E - e sin(E)`` is solved by Newton iteration, and
    the projected separation follows from the standard relations
    ``r cos(nu) = a (cos E - e)`` and ``r sin(nu) = a sqrt(1 - e^2) sin E``::

        delay = A1 * [sin(omega) (cos E - e)
                      + cos(omega) sqrt(1 - e^2) sin E]

    The mean anomaly is measured from periastron while the ELL1 phase is
    measured from the ascending node, so ``M = Phi - omega``; PINT states the
    same relation as ``ELL1_T0 = TASC + PB/(2 pi) arctan(eps1/eps2)``.

    At ``e = 0`` this reduces to ``A1 sin(Phi)`` identically.

    Parameters
    ----------
    t_from_tasc_sec : np.ndarray
        Pulsar-frame time measured from ``TASC``, in seconds.
    solution : InjectedSolution
        The injected truth.
    pb_sec : float or None, optional
        Orbital period in force at that ``TASC``. See :func:`orbital_delay`.

    Returns
    -------
    np.ndarray
        Delay in seconds.
    """
    if pb_sec is None:
        pb_sec = solution.PB_sec
    if a1 is None:
        a1 = solution.A1
    e, om = solution.ECC, solution.OM

    phi = 2 * np.pi * t_from_tasc_sec / pb_sec
    mean_anomaly = phi - om

    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(100):
        step = (eccentric_anomaly - e * np.sin(eccentric_anomaly) - mean_anomaly) / (
            1 - e * np.cos(eccentric_anomaly)
        )
        eccentric_anomaly = eccentric_anomaly - step
        if np.max(np.abs(step)) < 1e-15:
            break

    return a1 * (
        np.sin(om) * (np.cos(eccentric_anomaly) - e)
        + np.cos(om) * np.sqrt(1 - e**2) * np.sin(eccentric_anomaly)
    )


def pulse_shape(phase, duty=0.12):
    """A wrapped-Gaussian pulse normalized to the range ``[0, 1]``.

    A single narrow peak, deliberately non-sinusoidal so that templates with
    ``nharm > 1`` are actually exercised (the shipped test data is a pure
    sinusoid, which never leaves the ``nharm == 1`` branch).

    Parameters
    ----------
    phase : np.ndarray
        Pulse phase; only the fractional part matters.
    duty : float, optional
        Gaussian sigma in phase units. Smaller means sharper.

    Returns
    -------
    np.ndarray
        Values spanning exactly ``[0, 1]``.
    """
    ph = phase - np.floor(phase)
    # Sum three images so the peak wraps smoothly across the 0/1 boundary.
    g = sum(np.exp(-0.5 * ((ph + k - 0.5) / duty) ** 2) for k in (-1, 0, 1))
    return (g - g.min()) / (g.max() - g.min())


def pulsed_fraction_at(energy, pf_ref=0.25, e_ref=3.0, index=0.6, pf_max=0.85):
    """Energy-dependent pulsed fraction, rising with energy.

    Gives ``--use-weight`` and ``--use-pi`` something real to find: a flat
    pulsed fraction would make energy weighting a no-op and the tests vacuous.

    Returns
    -------
    np.ndarray
        Pulsed fraction in ``[0, pf_max]``.
    """
    pf = pf_ref * (energy / e_ref) ** index
    return np.clip(pf, 0.0, pf_max)


def generate_epoch(
    solution,
    pepoch,
    duration=None,
    n_events=5000,
    phase0=0.35,
    duty=0.12,
    energy_range=(0.5, 10.0),
    photon_index=1.8,
    n_gtis=4,
    gti_duty=0.6,
    rng=None,
):
    """Generate one observation's worth of events from the injected solution.

    Events are drawn by rejection sampling in the pulsar frame and then pushed
    through the forward orbital model, so the arrival times carry a genuine
    ELL1 modulation that the pipeline has to undo.

    Parameters
    ----------
    solution : InjectedSolution
        The injected truth.
    pepoch : float
        This epoch's reference epoch, MJD.
    duration : float or None, optional
        Observation span in seconds (wall clock, before GTI filtering). Defaults
        to one full orbital period -- see :func:`make_multi_epoch_dataset`.
    n_events : int, optional
        Approximate number of events surviving GTI filtering.
    phase0 : float, optional
        Pulse phase offset for this epoch.
    duty : float, optional
        Pulse sharpness, passed to :func:`pulse_shape`.
    energy_range : tuple of float, optional
        ``(emin, emax)`` in keV.
    photon_index : float, optional
        Power-law photon index of the energy distribution.
    n_gtis : int, optional
        Number of good-time intervals the observation is broken into.
    gti_duty : float, optional
        Fraction of the span that is on-source.
    rng : np.random.Generator or None, optional
        Seeded generator. A fresh default generator is used if omitted.

    Returns
    -------
    dict
        ``times_from_pepoch``, ``energy``, ``pi``, ``gtis_from_pepoch``,
        ``pepoch``, ``F0``, ``F1``, ``TASC``, ``PB``, ``A1``, and ``phase0``.
        ``TASC``, ``PB`` and ``A1`` are the epoch-local set, which differ per
        epoch once ``PBDOT`` or ``A1DOT`` is nonzero.
    """
    if rng is None:
        rng = np.random.default_rng()
    if duration is None:
        # One full orbit *inside the good-time intervals*, not merely one orbit
        # of wall clock: the last GTI ends at ((n - 1) + duty) / n of the span,
        # so a span of exactly PB would leave the final tenth of the orbit
        # unsampled.
        duration = solution.PB_sec * n_gtis / (n_gtis - 1 + gti_duty)

    F0, F1 = solution.spin_at(pepoch)
    tasc = solution.tasc_near(pepoch)
    tasc_sec = (tasc - pepoch) * SEC_PER_DAY
    pb_sec = solution.PB_sec_near(pepoch)
    a1 = solution.A1_near(pepoch)

    gtis = _build_gtis(duration, n_gtis, gti_duty)
    live_time = np.sum(np.diff(gtis, axis=1))

    # Oversample: rejection sampling and GTI filtering both throw events away.
    # The mean acceptance of the profile is 1/(1+pf) times the GTI duty cycle,
    # so 4x plus a floor is a comfortable margin.
    n_draw = int(4 * n_events / max(gti_duty, 0.05)) + 1000

    accepted = {"t": [], "e": []}
    n_have = 0
    while n_have < n_events:
        t_pulsar = rng.uniform(0.0, duration, n_draw)
        energy = _draw_powerlaw(rng, n_draw, energy_range, photon_index)

        # Spin phase in the pulsar frame, measured from PEPOCH.
        spin_phase = phase0 + F0 * t_pulsar + 0.5 * F1 * t_pulsar**2
        pf = pulsed_fraction_at(energy)
        # Profile with (max - min) / (max + min) == pf, by construction.
        rate = 1.0 + pf * (2.0 * pulse_shape(spin_phase, duty=duty) - 1.0)
        keep = rng.uniform(0.0, 1.0 + pf.max(), n_draw) < rate

        t_pulsar = t_pulsar[keep]
        energy = energy[keep]

        # Forward orbital model: pulsar frame -> observed arrival time.
        delay = kepler_orbital_delay if solution.exact_kepler else orbital_delay
        t_obs = t_pulsar + delay(t_pulsar - tasc_sec, solution, pb_sec=pb_sec, a1=a1)

        in_gti = _in_gtis(t_obs, gtis)
        accepted["t"].append(t_obs[in_gti])
        accepted["e"].append(energy[in_gti])
        n_have += int(np.sum(in_gti))

    times = np.concatenate(accepted["t"])
    energy = np.concatenate(accepted["e"])

    order = np.argsort(times)
    times = times[order][:n_events]
    energy = energy[order][:n_events]

    return {
        "times_from_pepoch": times,
        "energy": energy,
        # A simple linear channel mapping, enough for --use-pi to be meaningful.
        "pi": np.floor(energy / 0.04).astype(int),
        "gtis_from_pepoch": gtis,
        "pepoch": pepoch,
        "F0": F0,
        "F1": F1,
        "TASC": tasc,
        "PB": pb_sec / SEC_PER_DAY,
        "A1": a1,
        "phase0": phase0,
        "live_time": live_time,
    }


def _draw_powerlaw(rng, size, energy_range, photon_index):
    """Draw energies from a power law by inverse-transform sampling."""
    emin, emax = energy_range
    gamma = 1.0 - photon_index
    u = rng.uniform(0.0, 1.0, size)
    return (emin**gamma + u * (emax**gamma - emin**gamma)) ** (1.0 / gamma)


def _build_gtis(duration, n_gtis, gti_duty):
    """Split a span into evenly spaced good-time intervals."""
    edges = np.linspace(0.0, duration, n_gtis + 1)
    length = np.diff(edges) * gti_duty
    return np.column_stack([edges[:-1], edges[:-1] + length])


def _in_gtis(times, gtis):
    """Boolean mask of events falling inside any good-time interval."""
    mask = np.zeros(times.size, dtype=bool)
    for start, stop in gtis:
        mask |= (times >= start) & (times < stop)
    return mask


PARFILE_TEMPLATE = """# Synthetic test data generated by ell1fit.tests.datagen
PSR                                 Synth{index}
EPHEM                               DE421
RAJ                      9:55:51.04010000
DECJ                    69:40:45.49010000
PMRA                                  0.0
PMDEC                                 0.0
F0                   {F0:.16g} {F0_unc:.3g}
F1                   {F1:.16g} {F1_unc:.3g}
PEPOCH               {PEPOCH:.16g}
PLANET_SHAPIRO                          N
BINARY                               ELL1
PB                   {PB:.16g} {PB_unc:.3g}
PBDOT                {PBDOT:.16g}
A1                   {A1:.16g} {A1_unc:.3g}
A1DOT                {A1DOT:.16g}
TASC                 {TASC:.16g} {TASC_unc:.3g}
EPS1                 {EPS1:.16g}
EPS2                 {EPS2:.16g}
TZRMJD               {PEPOCH:.16g}
TZRSITE                               ssb
TZRFRQ                                inf
"""


def write_parfile(path, epoch, solution, index=0, offsets=None, uncertainties=None):
    """Write a PINT-readable ELL1 parfile for one epoch.

    Parameters
    ----------
    path : str
        Destination path.
    epoch : dict
        An entry returned by :func:`generate_epoch`.
    solution : InjectedSolution
        The injected truth.
    index : int, optional
        Used only to give each pulsar a distinct name.
    offsets : dict or None, optional
        Parameter-name to additive offset, applied to the written values. This
        is how a test deliberately mis-sets a parameter -- e.g.
        ``{"PB": 3e-6}`` -- to check that the fit pulls it back.
    uncertainties : dict or None, optional
        Parameter-name to uncertainty, overriding the defaults.

    Returns
    -------
    str
        The path written.
    """
    offsets = dict(offsets or {})
    unc = {
        "F0": 1e-9,
        "F1": 1e-16,
        "PB": 1e-6,
        "A1": 1e-3,
        "TASC": 1e-6,
    }
    unc.update(uncertainties or {})

    values = {
        "F0": epoch["F0"],
        "F1": epoch["F1"],
        # TASC, PB and A1 are the epoch-local set: a parfile's period and
        # projected semi-major axis are those in force at the TASC it quotes.
        # With PBDOT = A1DOT = 0 these are solution.PB and solution.A1.
        "PB": epoch["PB"],
        "A1": epoch["A1"],
        "TASC": epoch["TASC"],
        "EPS1": solution.EPS1,
        "EPS2": solution.EPS2,
        "PBDOT": solution.PBDOT,
        "A1DOT": solution.A1DOT,
    }

    # PINT reads a PBDOT or A1DOT above 1e-7 in magnitude as being quoted in
    # the parfile convention of 1e-12 units, and silently multiplies it by
    # 1e-12. No physical value is anywhere near that, but a test reaching for
    # an exaggerated one would get a solution twelve orders of magnitude away
    # from the one it asked for, and the generator would still look right.
    for name in ("PBDOT", "A1DOT"):
        if abs(values[name]) > 1e-7:
            raise ValueError(
                f"{name}={values[name]:g} exceeds 1e-7, which PINT rescales by 1e-12 on read"
            )
    for name, offset in offsets.items():
        if name not in values:
            raise KeyError(f"Cannot offset unknown parameter {name!r}")
        values[name] += offset

    text = PARFILE_TEMPLATE.format(
        index=index,
        PEPOCH=epoch["pepoch"],
        F0_unc=unc["F0"],
        F1_unc=unc["F1"],
        PB_unc=unc["PB"],
        A1_unc=unc["A1"],
        TASC_unc=unc["TASC"],
        **values,
    )
    with open(path, "w") as fobj:
        fobj.write(text)
    return path


def write_eventfile(path, epoch, mjdref=56000.0):
    """Write one epoch's events to a HENDRICS-readable file.

    Returns
    -------
    str
        The path written.
    """
    from hendrics.io import save_events
    from stingray.events import EventList

    pepoch_met = (epoch["pepoch"] - mjdref) * SEC_PER_DAY

    events = EventList(
        time=epoch["times_from_pepoch"] + pepoch_met,
        gti=epoch["gtis_from_pepoch"] + pepoch_met,
        mjdref=mjdref,
        energy=epoch["energy"],
    )
    events.pi = epoch["pi"]
    events.instr = "synth"
    events.mission = "synth"
    save_events(events, path)
    return path


def make_multi_epoch_dataset(
    outdir,
    solution=None,
    epoch_offsets=(0.0, 37.0, 91.0),
    n_events=5000,
    duration=None,
    phase0=(0.35, 0.35, 0.35),
    duty=0.12,
    seed=20260820,
    offsets=None,
    uncertainties=None,
    prefix="synth",
):
    """Generate a complete multi-epoch dataset: event files plus parfiles.

    Parameters
    ----------
    outdir : str
        Directory to write into.
    solution : InjectedSolution or None, optional
        The injected truth. A default solution is used if omitted.
    epoch_offsets : sequence of float, optional
        Days from ``solution.pepoch_ref`` for each epoch. The defaults span
        three months, so ``F1`` has a visible effect and each epoch lands on a
        different orbital alias of ``TASC``.
    n_events : int, optional
        Events per epoch.
    duration : float or None, optional
        Observation span per epoch, in seconds. Defaults to **one full orbital
        period**, which is what it takes to constrain the orbit.

        ``A1`` fixes the amplitude of a sinusoid in orbital phase and ``EPS1``/
        ``EPS2`` the amplitudes of its second harmonic, so a span covering only
        part of an orbit measures them from an arc rather than a cycle. Measured
        at fixed total counts, going from 0.46 to 1.0 orbits improves
        ``sigma(EPS)`` by a factor of 1.8 for the same photons. The tell is the
        asymmetry: over a partial orbit ``sigma(EPS1)`` and ``sigma(EPS2)``
        differ by 11%, because the ``sin 2 Phi`` and ``cos 2 Phi`` directions are
        sampled unequally; over a full orbit they agree to 0.3%. Two orbits buy
        nothing further.

        Longer spans are free here: the generator draws a fixed number of events
        regardless, so a wider span lowers the count rate rather than raising the
        cost of anything.
    phase0 : sequence of float, optional
        Per-epoch pulse phase offset.
    duty : float, optional
        Pulse sharpness, passed to :func:`pulse_shape`. The default is a narrow
        peak with real harmonic content; a large value (~0.3) gives an almost
        pure sinusoid, which is what a single-harmonic model is *correct* for.
    seed : int, optional
        Seed for the generator, so datasets are reproducible.
    offsets : dict or None, optional
        Parameter offsets written into *every* parfile, to deliberately start
        the fit away from the truth. See :func:`write_parfile`.
    uncertainties : dict or None, optional
        Parameter uncertainties written into every parfile, overriding the
        defaults. Loosening these keeps the priors from fighting the data when
        a test deliberately offsets a starting value.
    prefix : str, optional
        Base name for the generated files.

    Returns
    -------
    dict
        ``event_files``, ``par_files``, ``epochs``, and ``solution``.
    """
    import os

    if solution is None:
        solution = InjectedSolution()

    rng = np.random.default_rng(seed)

    event_files = []
    par_files = []
    epochs = []

    for i, day_offset in enumerate(epoch_offsets):
        pepoch = solution.pepoch_ref + day_offset
        epoch = generate_epoch(
            solution,
            pepoch,
            duration=duration,
            n_events=n_events,
            phase0=phase0[i % len(phase0)],
            duty=duty,
            rng=rng,
        )
        epochs.append(epoch)

        event_path = os.path.join(outdir, f"{prefix}{i}.nc")
        par_path = os.path.join(outdir, f"{prefix}{i}.par")
        write_eventfile(event_path, epoch)
        write_parfile(
            par_path,
            epoch,
            solution,
            index=i,
            offsets=offsets,
            uncertainties=uncertainties,
        )

        event_files.append(event_path)
        par_files.append(par_path)

    return {
        "event_files": event_files,
        "par_files": par_files,
        "epochs": epochs,
        "solution": solution,
    }
