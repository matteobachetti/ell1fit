r"""The orbit summary figure: a physical corner plot beside its eccentricity.

The figure is checked through its own axes rather than by looking at the saved
image. What matters and is testable is structural: that exactly the parameters
the chain explored get panels, in a fixed order, with the physical centre they
were offset by written on the axis; that the two blocks sit side by side rather
than on top of one another -- ``corner`` ends by calling ``subplots_adjust`` on
whatever figure it is handed, which is precisely the way this layout could
silently collapse; and that the corner panels stay square whatever the
eccentricity panel does to the figure's height.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ..eccentricity import eccentricity_summary  # noqa: E402
from ..orbit_plot import ORBITAL_PARAMETERS, plot_orbit_summary  # noqa: E402


SEED = 20260905
NSAMPLES = 4000

#: A plausible low-mass X-ray binary: tens of light-seconds, a period of a few
#: days, an epoch as an MJD, and an eccentricity a few sigma from zero.
CENTRES = {"A1": 22.225, "PB": 218668.4, "TASC": 56696.34, "EPS1": 3.0e-4, "EPS2": -1.0e-4}
WIDTHS = {"A1": 3e-4, "PB": 0.35, "TASC": 1.2 / 86400, "EPS1": 8e-5, "EPS2": 8e-5}


def _samples(parameters, seed=SEED):
    rng = np.random.default_rng(seed)
    return {
        par: CENTRES[par] + WIDTHS[par] * rng.normal(size=NSAMPLES)
        for par in parameters
        if par in CENTRES
    }


@pytest.fixture
def captured_figure(monkeypatch):
    """Hold on to the figure the plot saves, so its axes can be inspected."""
    held = {}

    def _savefig(self, *args, **kwargs):
        held["figure"] = self

    monkeypatch.setattr(Figure, "savefig", _savefig)
    return held


def _corner_axes(figure, n):
    """The n*n corner axes, which are added before the eccentricity panel."""
    return figure.axes[: n * n]


def test_the_file_is_written(tmp_path):
    fname = str(tmp_path / "orbit.jpg")

    assert plot_orbit_summary(_samples(ORBITAL_PARAMETERS), fname=fname) == fname
    assert (tmp_path / "orbit.jpg").exists()


def test_only_the_parameters_the_chain_explored_get_panels(tmp_path, captured_figure):
    plot_orbit_summary(_samples(["A1", "EPS1", "EPS2"]), fname=str(tmp_path / "o.jpg"))

    figure = captured_figure["figure"]
    assert len(figure.axes) == 3 * 3 + 1


def test_parameters_outside_the_orbital_set_are_ignored(tmp_path, captured_figure):
    """A fit of F0 and F1 alongside the orbit does not put them in this figure."""
    samples = _samples(["A1", "EPS1", "EPS2"])
    samples["F0"] = np.full(NSAMPLES, 1.37)
    samples["F1"] = np.full(NSAMPLES, -1e-12)

    plot_orbit_summary(samples, fname=str(tmp_path / "o.jpg"))

    assert len(captured_figure["figure"].axes) == 3 * 3 + 1


def test_panels_follow_the_documented_order_not_the_dictionary(tmp_path, captured_figure):
    shuffled = {par: s for par, s in reversed(list(_samples(ORBITAL_PARAMETERS).items()))}

    plot_orbit_summary(shuffled, fname=str(tmp_path / "o.jpg"))

    figure = captured_figure["figure"]
    bottom_row = _corner_axes(figure, 5)[-5:]
    assert [ax.get_xlabel().split()[0] for ax in bottom_row] == list(ORBITAL_PARAMETERS)


def test_each_axis_carries_the_physical_centre_it_was_offset_by(tmp_path, captured_figure):
    plot_orbit_summary(_samples(["A1", "EPS1", "EPS2"]), fname=str(tmp_path / "o.jpg"))

    labels = [ax.get_xlabel() for ax in _corner_axes(captured_figure["figure"], 3)[-3:]]
    assert labels[0].startswith("A1 - 22.225")
    assert "lt-s" in labels[0]
    assert "EPS1" in labels[1] and "EPS2" in labels[2]


def test_the_two_blocks_sit_side_by_side(tmp_path, captured_figure):
    """corner's own subplots_adjust must not drag the corner over the histogram."""
    plot_orbit_summary(_samples(ORBITAL_PARAMETERS), fname=str(tmp_path / "o.jpg"))

    figure = captured_figure["figure"]
    corner_right = max(ax.get_position().x1 for ax in _corner_axes(figure, 5))
    eccentricity_left = figure.axes[-1].get_position().x0
    assert corner_right < eccentricity_left


@pytest.mark.parametrize("parameters", [["EPS1", "EPS2"], ["A1", "PB", "TASC", "EPS1", "EPS2"]])
def test_the_corner_panels_stay_square(tmp_path, captured_figure, parameters):
    """However tall the eccentricity makes the figure, the panels do not stretch."""
    plot_orbit_summary(_samples(parameters), fname=str(tmp_path / "o.jpg"))

    figure = captured_figure["figure"]
    fig_width, fig_height = figure.get_size_inches()
    for ax in _corner_axes(figure, len(parameters)):
        box = ax.get_position()
        assert np.isclose(box.width * fig_width, box.height * fig_height, rtol=1e-6)


def test_a_supplied_summary_is_the_one_shown(tmp_path, captured_figure):
    samples = _samples(ORBITAL_PARAMETERS)
    summary = eccentricity_summary(samples["EPS1"], samples["EPS2"], upper_limit_level=0.99)

    plot_orbit_summary(samples, fname=str(tmp_path / "o.jpg"), summary=summary)

    title = captured_figure["figure"].axes[-1].get_title()
    assert title.replace("\n", "; ") == summary["ECC_summary"]


@pytest.mark.parametrize("parameters", [["A1", "PB", "EPS1"], ["A1", "PB", "TASC"]])
def test_without_both_eps_there_is_nothing_to_pair_the_corner_with(tmp_path, parameters):
    with pytest.raises(ValueError, match="No eccentricity posterior to show"):
        plot_orbit_summary(_samples(parameters), fname=str(tmp_path / "o.jpg"))
