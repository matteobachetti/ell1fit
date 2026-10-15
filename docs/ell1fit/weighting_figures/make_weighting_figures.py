"""Render the example weighting diagnostics shown in ``docs/ell1fit/weighting.rst``.

The figures are produced by the pipeline's own code path,
:func:`ell1fit.pipeline._build_profiles_and_weights`, on a simulated
observation, so they are exactly what ``ell1fit --use-weight`` writes next to
its results. Only the format differs: JPG at 150 dpi, ample for a web page and
a fraction of the size of the 300 dpi default.

Run from anywhere; the JPGs are written next to this script::

    python docs/ell1fit/weighting_figures/make_weighting_figures.py

The output is deterministic (fixed seed), so rerunning it after a change to the
plotting code shows the change and nothing else.
"""

import functools
import os

import numpy as np

import ell1fit.pipeline as pipeline
import ell1fit.weighting as weighting
from ell1fit.plotting import save_figure, set_figure_format

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "example")
DPI = 150
SEED = 20261015
N_EVENTS = 60_000
E_RANGE = (3.0, 60.0)
F0 = 1.0


def pulsed_fraction(energy):
    """A curved, non-monotonic pulsed fraction, like a real accreting pulsar.

    It rises from 5% at 3 keV to about 30% near 15 keV and falls again above,
    with a dip at the 6.4 keV iron line, whose photons come from reprocessing
    and carry little pulsation.
    """
    log_e = np.log(energy)
    hump = 0.05 + 0.27 * np.exp(-0.5 * ((log_e - np.log(15.0)) / 0.55) ** 2)
    iron_dip = 1 - 0.6 * np.exp(-0.5 * ((energy - 6.4) / 0.35) ** 2)
    return hump * iron_dip


def draw_energies(rng, size):
    """Cutoff power law, dN/dE ~ E^-1 exp(-E/12 keV), by inverting its CDF on a grid."""
    grid = np.geomspace(*E_RANGE, 4000)
    density = grid**-1.0 * np.exp(-grid / 12.0)
    cdf = np.concatenate(([0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))))
    return np.interp(rng.random(size), cdf / cdf[-1], grid)


def draw_phases(rng, amplitude):
    """Phases following ``1 + a cos(2 pi phi)``, by accept-reject (exact)."""
    phases = np.empty(amplitude.size)
    todo = np.arange(amplitude.size)
    while todo.size:
        trial = rng.random(todo.size)
        keep = rng.random(todo.size) < 0.5 * (1 + amplitude[todo] * np.cos(2 * np.pi * trial))
        phases[todo[keep]] = trial[keep]
        todo = todo[~keep]
    return phases


def main():
    rng = np.random.default_rng(SEED)
    energies = draw_energies(rng, N_EVENTS)
    phases = draw_phases(rng, pulsed_fraction(energies))
    # One event per spin period: an isolated pulsar at F0 = 1 Hz, observed for
    # N_EVENTS seconds. Only the phases matter to the weighting.
    times = (phases + np.arange(N_EVENTS)) / F0

    parameters = {
        "PB": 1e6,
        "A1": 0.0,
        "EPS1": 0.0,
        "EPS2": 0.0,
        "TASC": 0.0,
        "PEPOCH_0": 55000.0,
        "F0_0": F0,
        "Phase_0": 0.0,
    }

    # The pipeline saves at 300 dpi; lower it for the web without touching the
    # plotting code, so the figures stay identical in everything else.
    low_dpi_save = functools.partial(save_figure, dpi=DPI)
    weighting._save_figure = low_dpi_save
    pipeline._save_figure = low_dpi_save

    set_figure_format("jpg")
    pipeline._build_profiles_and_weights(
        [times],
        parameters,
        [energies],
        n_files=1,
        get_outroot=lambda i: ROOT,
        use_weight=True,
        nbin=32,
        tolerance=1e-8,
    )
    for suffix in ("_pf_weight_spectrum", "_weighted_profile_comparison"):
        print("wrote", ROOT + suffix + ".jpg")


if __name__ == "__main__":
    main()
