"""Tests for :mod:`ell1fit.information_report`, the ``ell1info`` output.

The interesting function here is the predicted eccentricity limit, which turns
a 2x2 information block into the bound a fit would report without running the
fit. It has a closed form only when the two components are equally well
measured, and :mod:`ell1fit.observed_information` shows they usually are not,
so it is checked against the analytic case and then for the right behaviour
away from it.
"""

import numpy as np
import pytest

from ..information import InformationBudget, schur_information
from ..information_report import (
    RAYLEIGH_THREE_SIGMA,
    format_budget,
    predicted_eccentricity_limit,
)


SHARED = ["A1", "EPS1", "EPS2"]


def _budget(information, names=SHARED, n=2):
    """A budget whose total is the given matrix, split over ``n`` equal segments."""
    information = np.asarray(information, dtype=float)
    segments = [schur_information(information / n, len(names), name=f"seg{i}") for i in range(n)]
    return InformationBudget(names, segments)


def test_rayleigh_constant_is_the_three_sigma_point():
    """99.73% of a 2-D Gaussian's mass lies within 3.439 sigma of the origin."""
    assert RAYLEIGH_THREE_SIGMA == pytest.approx(np.sqrt(-2.0 * np.log(1.0 - 0.9973)), rel=1e-6)


def test_isotropic_limit_matches_the_closed_form():
    """Equal, uncorrelated components give exactly the Rayleigh answer."""
    sigma = 4.0e-4
    information = np.diag([1.0, sigma**-2, sigma**-2])
    limit = predicted_eccentricity_limit(_budget(information), seed=99)
    assert limit == pytest.approx(RAYLEIGH_THREE_SIGMA * sigma, rel=0.02)


def test_anisotropic_limit_lies_between_the_two_axes():
    """A badly-sampled direction dominates the bound but does not fully set it."""
    small, large = 2.0e-4, 8.0e-4
    information = np.diag([1.0, small**-2, large**-2])
    limit = predicted_eccentricity_limit(_budget(information), seed=99)
    assert RAYLEIGH_THREE_SIGMA * small < limit < RAYLEIGH_THREE_SIGMA * large


def test_a_worse_measurement_gives_a_weaker_limit():
    """Halving the information must loosen the bound, not tighten it."""
    sigma = 4.0e-4
    good = predicted_eccentricity_limit(_budget(np.diag([1.0, sigma**-2, sigma**-2])), seed=1)
    bad = predicted_eccentricity_limit(
        _budget(np.diag([1.0, 0.25 * sigma**-2, 0.25 * sigma**-2])), seed=1
    )
    assert bad == pytest.approx(2.0 * good, rel=0.03)


def test_eccentricity_limit_needs_both_components():
    budget = _budget(np.diag([1.0, 1.0]), names=["A1", "EPS1"])
    with pytest.raises(ValueError, match="EPS1 and EPS2"):
        predicted_eccentricity_limit(budget)


def test_format_budget_names_every_segment_and_parameter():
    budget = _budget(np.diag([1.0, 2.0, 3.0]), n=3)
    text = format_budget(budget)
    for name in SHARED:
        assert name in text
    for i in range(3):
        assert f"seg{i}" in text


def test_format_budget_reports_the_cost_of_freeing_the_eccentricity():
    """Holding EPS fixed can only help A1, and the report says by how much."""
    information = np.array([[1.0, 0.8, 0.0], [0.8, 2.0, 0.0], [0.0, 0.0, 3.0]])
    text = format_budget(_budget(information))
    assert "EPS1, EPS2 fixed" in text


def test_format_budget_marks_the_segments_carrying_most_of_the_information():
    budget = _budget(np.diag([1.0, 1.0, 1.0]), n=4)
    text = format_budget(budget, parameter="A1")
    assert "90%" in text
