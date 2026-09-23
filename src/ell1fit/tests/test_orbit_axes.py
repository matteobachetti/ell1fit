r"""Choosing a readable centre and unit for an orbital-parameter axis.

A posterior on ``A1`` is a spike 1e-6 light-seconds wide sitting at 26.5
light-seconds. Plotted raw it is a vertical line at an unreadable tick label,
so each axis is drawn as an *offset*: the posterior mean is subtracted and what
remains is divided by a unit small enough that the standard deviation covers a
few units rather than a millionth of one. What is pinned down here is that the
unit chosen is the readable one -- hours and minutes for a wide orbital period,
microseconds for a sharp epoch -- and that the centre is printed to enough
digits to be worth printing at all.
"""

import numpy as np
import pytest

from ..orbit_plot import axis_scale


def _samples(centre, sigma, size=5000, seed=20260905):
    """Gaussian samples with essentially exact mean and standard deviation."""
    raw = np.random.default_rng(seed).normal(size=size)
    return centre + sigma * (raw - raw.mean()) / raw.std()


@pytest.mark.parametrize(
    "sigma_seconds, expected",
    [
        (3.0e5, "d"),
        (7200.0, "h"),
        (120.0, "min"),
        (45.0, "s"),
        (0.5, "ms"),
        (1e-4, "µs"),
        (1e-8, "ns"),
        (1e-15, "ns"),
    ],
)
def test_a_time_axis_gets_the_readable_time_unit(sigma_seconds, expected):
    """PB is stored in seconds; its residuals are shown in whatever reads best."""
    scale = axis_scale("PB", _samples(218668.4, sigma_seconds))

    assert scale.unit == expected
    assert 1.0 <= sigma_seconds / scale.scale or expected == "ns"


def test_the_chosen_unit_makes_the_width_a_readable_number():
    """One standard deviation lands between 1 and the next unit up, everywhere."""
    for sigma in np.logspace(-9, 5, 60):
        scale = axis_scale("PB", _samples(218668.4, sigma))
        assert 1.0 <= sigma / scale.scale < 1000.0


def test_residuals_are_centred_on_the_mean_and_expressed_in_that_unit():
    sigma = 0.5
    samples = _samples(218668.4, sigma)

    scale = axis_scale("PB", samples)
    residuals = scale.apply(samples)

    assert scale.unit == "ms"
    assert np.isclose(residuals.mean(), 0.0, atol=1e-9)
    assert np.isclose(residuals.std(), 500.0)


def test_tasc_is_stored_in_days_but_read_in_seconds():
    """An epoch is quoted as an MJD; its uncertainty is never quoted in days."""
    sigma_days = 1.0 / 86400.0
    samples = _samples(57000.25, sigma_days)

    scale = axis_scale("TASC", samples)

    assert scale.unit == "s"
    assert np.isclose(scale.apply(samples).std(), 1.0)
    assert "MJD" in scale.label
    assert "57000.25" in scale.label


def test_pb_is_stored_in_seconds_but_its_centre_is_quoted_in_days():
    """Parfiles carry PB in days; that is the number to recognise on the axis."""
    scale = axis_scale("PB", _samples(218668.4, 0.5))

    assert "2.53088" in scale.label
    assert " d" in scale.label
    assert "(ms)" in scale.label


def test_a1_keeps_light_seconds_with_a_metric_prefix():
    scale = axis_scale("A1", _samples(26.5, 2.5e-6))

    assert scale.unit == "µlt-s"
    assert np.isclose(scale.apply(_samples(26.5, 2.5e-6)).std(), 2.5)
    assert "26.5" in scale.label
    assert "lt-s" in scale.label


def test_a_width_just_under_a_unit_drops_to_the_finer_one():
    """The boundary is decided by the width alone, with no rounding slack.

    A posterior 0.999 seconds wide is shown as 999 ms rather than as 0.999 s,
    which is the readable choice and the one that keeps the rule stateable in
    one sentence: the unit is the largest the width still covers.
    """
    assert axis_scale("PB", _samples(218668.4, 0.999)).unit == "ms"
    assert axis_scale("PB", _samples(218668.4, 1.001)).unit == "s"


def test_a_dimensionless_parameter_gets_a_power_of_ten():
    scale = axis_scale("EPS1", _samples(1e-3, 4e-6))

    assert np.isclose(scale.scale, 1e-6)
    assert "10^{-6}" in scale.label
    assert np.isclose(scale.apply(_samples(1e-3, 4e-6)).std(), 4.0)


def test_the_centre_is_printed_finely_enough_to_place_the_residuals():
    """The printed centre must recover the mean to well inside the width.

    A centre rounded coarsely is worse than useless: the reader adds it to a
    residual read off the axis and lands somewhere the posterior is not.
    """
    samples = _samples(57000.2512345678, 1e-5)

    scale = axis_scale("TASC", samples)

    centre_text = scale.label.split(" - ")[1].split(" MJD")[0]
    assert abs(float(centre_text) - samples.mean()) < 1e-5 / 100


def test_an_unknown_parameter_still_gets_a_usable_axis():
    """No convention recorded: fall back to a bare power of ten, no unit name."""
    scale = axis_scale("F1", _samples(-1e-12, 3e-16))

    assert np.isclose(scale.scale, 1e-16)
    assert "F1" in scale.label
    assert np.isclose(scale.apply(_samples(-1e-12, 3e-16)).std(), 3.0)


def test_a_degenerate_posterior_does_not_blow_up():
    """A parameter that never moved has no width to choose a unit from."""
    scale = axis_scale("A1", np.full(100, 26.5))

    assert np.isfinite(scale.scale) and scale.scale > 0
    assert np.all(scale.apply(np.full(100, 26.5)) == 0.0)
