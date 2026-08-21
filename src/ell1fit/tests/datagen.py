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
        ELL1 Laplace-Lagrange eccentricity parameters.
    PBDOT : float
        Rate of change of the orbital period, dimensionless (s/s). Zero by
        default, in which case every quantity below reduces exactly to its
        constant-period form.
    pepoch_ref : float
        Reference epoch (MJD) at which ``F0``/``F1`` are defined.
    """

    F0: float = 7.5
    F1: float = -3.2e-13
    PB: float = 2.532971
    A1: float = 22.215
    TASC: float = 56682.0669
    EPS1: float = 1.5e-4
    EPS2: float = -2.1e-4
    PBDOT: float = 0.0
    pepoch_ref: float = 56682.0

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

    def PB_sec_near(self, pepoch):
        """Orbital period, in seconds, valid at the alias near ``pepoch``.

        The period a parfile quotes is the one in force at the ``TASC`` it
        quotes, so an epoch-local ``TASC`` has to be written with an
        epoch-local ``PB``.
        """
        elapsed = (self.tasc_near(pepoch) - self.TASC) * SEC_PER_DAY
        return self.PB_sec + self.PBDOT * elapsed


def orbital_delay(t_from_tasc_sec, solution, pb_sec=None):
    """Roemer delay of the ELL1 model, in seconds.

    This is the *forward* model: given a time in the pulsar frame, it returns
    the delay to add to obtain the observed arrival time.

    Parameters
    ----------
    t_from_tasc_sec : np.ndarray
        Pulsar-frame time measured from ``TASC``, in seconds.
    solution : InjectedSolution
        The injected truth.
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
    phase = 2 * np.pi * t_from_tasc_sec / pb_sec
    return solution.A1 * (
        np.sin(phase)
        + 0.5 * solution.EPS1 * np.sin(2 * phase)
        + 0.5 * solution.EPS2 * np.cos(2 * phase)
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
    duration=100_000.0,
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
    duration : float, optional
        Observation span in seconds (wall clock, before GTI filtering).
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
        ``pepoch``, ``F0``, ``F1``, ``TASC``, ``PB``, and ``phase0``. ``TASC``
        and ``PB`` are the epoch-local pair, which differ per epoch once
        ``PBDOT`` is nonzero.
    """
    if rng is None:
        rng = np.random.default_rng()

    F0, F1 = solution.spin_at(pepoch)
    tasc = solution.tasc_near(pepoch)
    tasc_sec = (tasc - pepoch) * SEC_PER_DAY
    pb_sec = solution.PB_sec_near(pepoch)

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
        t_obs = t_pulsar + orbital_delay(t_pulsar - tasc_sec, solution, pb_sec=pb_sec)

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
        # TASC and PB are the epoch-local pair: a parfile's period is the one
        # in force at the TASC it quotes. With PBDOT = 0 this is solution.PB.
        "PB": epoch["PB"],
        "A1": solution.A1,
        "TASC": epoch["TASC"],
        "EPS1": solution.EPS1,
        "EPS2": solution.EPS2,
        "PBDOT": solution.PBDOT,
    }
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
    duration=100_000.0,
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
    duration : float, optional
        Observation span per epoch, in seconds.
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
