"""Tests for :mod:`ell1fit.observed_information`, the Hessian half of the budget.

:mod:`ell1fit.information` is tested on matrices written down by hand. Here the
matrices come from the real likelihood evaluated on simulated events, so the
assertions are about the two things that connect the arithmetic to the data:
that the curvature is the curvature the likelihood actually has, and that it
moves the way the physics says it should when orbital coverage or the spin
priors change.
"""

import numpy as np
import pytest

from ..information import InformationBudget
from ..observed_information import (
    segment_information,
    segment_log_likelihood,
    split_shared_and_local,
)
from .datagen import InjectedSolution, make_multi_epoch_dataset
from .helpers import build_pipeline_state


ORBITAL = ["A1", "EPS1", "EPS2"]
FIT = ("F0", "F1", "A1", "EPS1", "EPS2")


def _state(tmp_path, duration_orbits, fit_parameters=FIT, n_events=4000, uncertainties=None, **kw):
    """Simulate one epoch spanning a given fraction of an orbit and set up a fit."""
    dataset = make_multi_epoch_dataset(
        str(tmp_path),
        epoch_offsets=(0.0,),
        phase0=(0.35,),
        n_events=n_events,
        duration=duration_orbits * InjectedSolution().PB * 86400.0,
        uncertainties=uncertainties,
        seed=20260925,
    )
    return build_pipeline_state(dataset, fit_parameters=fit_parameters, **kw)


def test_split_shared_and_local_separates_by_trailing_index():
    """``A1`` is one global number; ``F0_3`` belongs to file 3 alone."""
    names = ["A1", "EPS1", "F0_0", "F1_0", "Phase_0", "F0_1", "F1_1", "Phase_1"]
    shared, local = split_shared_and_local(names)
    assert shared == ["A1", "EPS1"]
    assert local == {0: ["F0_0", "F1_0", "Phase_0"], 1: ["F0_1", "F1_1", "Phase_1"]}


def test_split_shared_and_local_does_not_confuse_file_1_with_file_10():
    """``F0_1`` and ``F0_10`` differ by the whole trailing integer, not a prefix."""
    _, local = split_shared_and_local([f"F0_{i}" for i in (1, 10)])
    assert local == {1: ["F0_1"], 10: ["F0_10"]}


@pytest.fixture(scope="module")
def full_orbit(tmp_path_factory):
    """One epoch covering a full orbit, with A1 and the eccentricity shared."""
    observations, setup = _state(tmp_path_factory.mktemp("full"), 1.0, ignore_uncertainties=True)
    segments = segment_information(observations, setup)
    return observations, setup, InformationBudget(ORBITAL, segments)


def test_segments_are_named_after_their_files(full_orbit):
    _, _, budget = full_orbit
    assert len(budget.segments) == 1
    assert "synth" in budget.segments[0].name


def test_local_spin_fit_dilutes_but_does_not_erase(full_orbit):
    """Fitting F0 and F1 per segment eats part of the orbital signal, not all of it."""
    _, _, budget = full_orbit
    (segment,) = budget.segments
    assert np.all(segment.dilution > 0.0)
    assert np.all(segment.dilution < 1.0)


def test_the_eccentricity_direction_a_segment_samples_well_beats_a1(full_orbit):
    """``A1`` rides on sin(Phi); the eccentricity rides on sin and cos of 2 Phi.

    A wave of half the period departs from a quadratic in time much sooner, and
    the per-segment {Phase, F0, F1} fit is exactly a quadratic in phase, so it
    absorbs more of the ``A1`` signature than of the eccentricity's. The claim
    has to be made about the *better* of the two components, not both: see
    :func:`test_one_segment_sees_the_two_eccentricity_components_unequally`.
    """
    _, _, budget = full_orbit
    a1, eps1, eps2 = budget.segments[0].dilution
    assert max(eps1, eps2) > 2.0 * a1


def test_one_segment_sees_the_two_eccentricity_components_unequally(full_orbit):
    """A single arc pins one eccentricity direction and barely constrains the other.

    ``EPS1`` and ``EPS2`` multiply cos(2 Phi) and sin(2 Phi), and which of those
    looks more like a quadratic over the arc observed depends entirely on where
    in orbital phase the arc sits. One observation therefore measures one
    combination of the two far better than the other, however long it is, and it
    takes segments at different orbital phases to close the gap. That is a
    scheduling statement, not a sensitivity one, and it does not show up in any
    exposure table.
    """
    _, _, budget = full_orbit
    eccentricity_block = budget.segments[0].effective[1:, 1:]
    assert np.linalg.cond(eccentricity_block) > 1.5


def test_curvature_matches_a_direct_scan_of_the_likelihood(full_orbit):
    """The reported information is the second derivative the likelihood really has.

    Scanning ``A1`` with everything else held and fitting a parabola to the
    log likelihood must reproduce ``raw[A1, A1]``, which is what the
    finite-difference Hessian claims it is.
    """
    observations, setup, budget = full_orbit
    target = segment_log_likelihood(observations, setup, 0)
    a1 = setup.parameters["A1"]
    step = 0.3 * setup.factors[list(setup.parameter_names).index("A1")]

    offsets = np.linspace(-step, step, 9)
    values = np.array([target({"A1": a1 + d}) for d in offsets])
    # A parabola fitted to log L has curvature -I/2 in its quadratic coefficient.
    # The two routes agree to a few parts in a thousand rather than exactly: a
    # least-squares parabola over the whole scan absorbs some of the quartic
    # term, where a three-point difference at the end points does not.
    quadratic = np.polyfit(offsets, values, 2)[0]
    assert -2.0 * quadratic == pytest.approx(budget.segments[0].raw[0, 0], rel=5e-3)


def test_more_orbital_coverage_means_more_surviving_information(tmp_path_factory):
    """Same photons, wider orbital arc: more of the A1 signature outlives the spin fit.

    The count rate falls as the span grows because the generator draws a fixed
    number of events, so this isolates coverage from exposure.
    """
    dilutions = {}
    for label, orbits in (("partial", 0.35), ("full", 1.0)):
        observations, setup = _state(
            tmp_path_factory.mktemp(label), orbits, ignore_uncertainties=True
        )
        dilutions[label] = segment_information(observations, setup)[0].dilution[0]
    assert dilutions["full"] > dilutions["partial"]


def test_tight_spin_priors_leave_more_information_for_the_orbit(tmp_path_factory):
    """A prior that pins F0 and F1 stops them absorbing the orbital signature.

    Which prior the per-segment spin parameters get is therefore not a harmless
    detail: a narrow one buys precision on the shared orbital parameters
    directly, whether or not that narrowness is deserved.
    """
    wide_obs, wide_setup = _state(tmp_path_factory.mktemp("wide"), 1.0, ignore_uncertainties=True)
    narrow_obs, narrow_setup = _state(
        tmp_path_factory.mktemp("narrow"),
        1.0,
        uncertainties={"F0": 1e-10, "F1": 1e-16},
    )
    wide = segment_information(wide_obs, wide_setup)[0]
    narrow = segment_information(narrow_obs, narrow_setup)[0]
    assert narrow.effective[0, 0] > 10.0 * wide.effective[0, 0]
    assert narrow.dilution[0] > wide.dilution[0]


def test_flat_priors_cannot_change_the_shared_block(tmp_path_factory):
    """A1 and the eccentricity have uniform priors, which carry no curvature."""
    observations, setup = _state(tmp_path_factory.mktemp("flat"), 1.0, ignore_uncertainties=True)
    with_priors = segment_information(observations, setup, include_priors=True)[0]
    without = segment_information(observations, setup, include_priors=False)[0]
    assert with_priors.raw == pytest.approx(without.raw, rel=1e-6)


def test_unknown_shared_parameter_is_refused(tmp_path_factory):
    observations, setup = _state(tmp_path_factory.mktemp("unknown"), 1.0, ignore_uncertainties=True)
    with pytest.raises(ValueError, match="PBDOT"):
        segment_information(observations, setup, shared=["A1", "PBDOT"])


def test_a_setup_with_no_shared_parameters_is_refused(tmp_path_factory):
    """Fitting only per-file spin parameters leaves nothing for a budget to be about."""
    observations, setup = _state(
        tmp_path_factory.mktemp("noshared"),
        1.0,
        fit_parameters=("F0", "F1"),
        ignore_uncertainties=True,
    )
    with pytest.raises(ValueError, match="no shared"):
        segment_information(observations, setup)


def test_two_epochs_give_two_named_segments(tmp_path_factory):
    """The budget is per input file, in input order."""
    tmp_path = tmp_path_factory.mktemp("two")
    dataset = make_multi_epoch_dataset(
        str(tmp_path),
        epoch_offsets=(0.0, 37.0),
        phase0=(0.35, 0.35),
        n_events=3000,
        seed=20260925,
    )
    observations, setup = build_pipeline_state(
        dataset, fit_parameters=FIT, ignore_uncertainties=True
    )
    segments = segment_information(observations, setup)
    assert len(segments) == 2
    assert [s.name for s in segments] == [str(f) for f in observations.files]
