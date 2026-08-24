"""Tests for energy-dependent event weighting.

The weights only ever enter the likelihood through their *shape*: the weighted
profile and its noise level both scale linearly with them, so a global rescaling
cancels. Every assertion here is therefore about shape -- how the recovered
curve tracks the injected pulsed fraction, and how much pulsed signal the
weights recover -- never about absolute values.
"""

import numpy as np
import pytest
from stingray.pulse.pulsar import z_n_binned_events, z_n_gauss

from ..templates import estimate_weighted_profile_std
from ..weighting import _pulse_modulation, pf_weight_versus_energy


F0 = 1.0


def _parameters(n_files=1):
    """Minimal parameter dictionary: unit spin frequency, no binary."""
    parameters = {"PB": 1e6, "A1": 0.0, "EPS1": 0.0, "EPS2": 0.0, "TASC": 0.0}
    for i in range(n_files):
        parameters[f"PEPOCH_{i}"] = 55000.0
        parameters[f"F0_{i}"] = F0
        parameters[f"Phase_{i}"] = 0.0
    return parameters


def _simulate(pulsed_fraction, n_events=200_000, e_range=(3.0, 60.0), seed=42):
    """Draw events whose pulsed fraction follows ``pulsed_fraction(energy)``.

    Energies follow a falling power law, so the counts per unit log energy vary
    by orders of magnitude across the band -- the condition that makes a fixed
    energy binning misbehave.
    """
    rng = np.random.default_rng(seed)
    emin, emax = e_range
    # dN/dE ~ E^-2, sampled by inverting its cumulative distribution. The
    # highest-energy decade ends up with a thousandth of the counts of the
    # lowest, which is the whole difficulty this module has to cope with.
    u = rng.random(n_events)
    energies = 1.0 / (1.0 / emin - u * (1.0 / emin - 1.0 / emax))

    amplitude = np.asarray(pulsed_fraction(energies), dtype=float)
    # Rejection-free sampling of 1 + a cos(2 pi phi) by inverting on a fine grid
    # is overkill here; accept-reject is exact and fast enough at this size.
    phases = np.empty(n_events)
    todo = np.arange(n_events)
    while todo.size:
        trial = rng.random(todo.size)
        keep = rng.random(todo.size) < 0.5 * (1 + amplitude[todo] * np.cos(2 * np.pi * trial))
        phases[todo[keep]] = trial[keep]
        todo = todo[~keep]

    times = (phases + np.arange(n_events)) / F0
    return [times], [energies]


def _alignment(weights, truth):
    """Cosine similarity between a weight curve and the true pulsed fraction.

    Phase precision goes as ``1 / alignment``, and it is maximised at 1 by
    ``w`` proportional to the truth -- so this is the figure of merit the whole
    module exists to maximise, and it is blind to the overall scale, exactly as
    the likelihood is.
    """
    weights = np.clip(np.asarray(weights, dtype=float), 0, None)
    norm = np.sqrt(np.sum(weights**2)) * np.sqrt(np.sum(truth**2))
    return float(np.sum(weights * truth) / norm) if norm > 0 else 0.0


def _weighted_z2(phases, weights, nbin=32, nharm=1):
    profile = np.histogram(phases, bins=np.linspace(0, 1, nbin + 1), weights=weights)[0]
    error = estimate_weighted_profile_std(weights, nbin=nbin, ntrials=200)
    return float(z_n_gauss(profile, err=error, n=nharm))


def test_modulation_has_zero_mean_and_unit_mean_square():
    """The projection estimator relies on both normalizations."""
    rng = np.random.default_rng(0)
    phases = rng.random(200_000)
    phases = phases[rng.random(phases.size) < 0.5 * (1 + 0.3 * np.cos(2 * np.pi * phases))]

    modulation = _pulse_modulation(phases, nharm=1)
    grid = np.linspace(0, 1, 4096, endpoint=False)
    values = modulation(grid)

    assert np.mean(values) == pytest.approx(0.0, abs=1e-10)
    assert np.mean(values**2) == pytest.approx(1.0, rel=1e-6)


def test_weights_track_a_rising_pulsed_fraction():
    """Weights must follow the injected trend, not the count spectrum."""
    times, energies = _simulate(lambda e: np.clip(0.02 * (e / 3.0), 0, 0.5))
    weights = pf_weight_versus_energy(times, energies, _parameters())[0]

    assert weights.min() >= 0.0
    assert weights.max() == pytest.approx(1.0)

    order = np.argsort(energies[0])
    sorted_weights = weights[order]
    assert sorted_weights[-1] > 5 * sorted_weights[0]

    # Point-by-point monotonicity would be a test of the noise, not the trend:
    # below 5 keV the injected amplitude is a few percent and the recovered
    # curve wiggles within its own uncertainty there. What has to hold is that
    # the recovered shape is aligned with the truth, and much better aligned
    # than not weighting at all.
    truth = np.clip(0.02 * (energies[0] / 3.0), 0, 0.5)
    assert _alignment(weights, truth) > 0.99
    assert _alignment(np.ones_like(weights), truth) < 0.95


def test_weights_are_flat_when_the_pulsed_fraction_is():
    """An energy-independent pulse must not produce structured weights.

    This is the failure the previous recipe was prone to: per-band amplitudes
    estimated from Z^2 are rectified, so bands with few counts were pushed up
    and neighbouring bands could differ by a factor of two on noise alone.
    """
    times, energies = _simulate(lambda e: np.full_like(e, 0.10))
    weights = pf_weight_versus_energy(times, energies, _parameters())[0]

    # Peak normalization puts the maximum at 1; a flat truth should keep the
    # bulk of the events near it rather than spreading over all of [0, 1].
    assert np.median(weights) > 0.7
    # And no event may be discarded outright. The previous recipe padded its
    # interpolation with a zero at either end of the band, so the lowest- and
    # highest-energy events were given weight 0 whatever the data said.
    assert weights.min() > 0.3


def test_weighting_recovers_pulsed_signal():
    """Weighting must raise the weighted Z^2 above the unweighted one."""
    from ..phase_utils import _calculate_phases

    times, energies = _simulate(lambda e: np.clip(0.01 * (e / 3.0) ** 1.5, 0, 0.6))
    parameters = _parameters()
    weights = pf_weight_versus_energy(times, energies, parameters)[0]
    phases = np.asarray(_calculate_phases(times, parameters)[0], dtype=float)

    unweighted = float(z_n_binned_events(np.histogram(phases, bins=np.linspace(0, 1, 33))[0], 1))
    weighted = _weighted_z2(phases, weights)

    assert weighted > 1.2 * unweighted


def test_weights_stay_in_the_unit_interval():
    """The Pletsch-Clarke likelihood requires ``0 <= w <= 1``."""
    times, energies = _simulate(lambda e: np.clip(0.3 * np.cos(np.log(e / 10.0)), 0, 0.5))
    weights = pf_weight_versus_energy(times, energies, _parameters())[0]

    assert np.all(np.isfinite(weights))
    assert weights.min() >= 0.0
    assert weights.max() <= 1.0


def test_too_few_events_falls_back_to_uniform_weights():
    """An energy-resolved pulsed fraction is not measurable from a handful."""
    times, energies = _simulate(lambda e: np.full_like(e, 0.3), n_events=40)
    with pytest.warns(UserWarning, match="too few"):
        weights = pf_weight_versus_energy(times, energies, _parameters())[0]

    assert np.all(weights == 1.0)


def test_non_positive_energies_do_not_break_the_fit():
    """``--use-pi`` supplies channel numbers, and channel zero is legal."""
    times, energies = _simulate(lambda e: np.clip(0.02 * (e / 3.0), 0, 0.5), n_events=50_000)
    channels = [np.round((energies[0] - energies[0].min()) * 10.0)]

    weights = pf_weight_versus_energy(times, channels, _parameters())[0]

    assert channels[0].min() == 0.0
    assert np.all(np.isfinite(weights))
    assert weights.max() == pytest.approx(1.0)
