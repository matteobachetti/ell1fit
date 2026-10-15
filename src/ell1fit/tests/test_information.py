"""Tests for :mod:`ell1fit.information`, the per-segment information budget.

Nothing here touches event data. Every assertion is about the linear algebra
that turns a set of per-segment Hessians into "how much does each observation
actually tell us about the shared orbital parameters", checked against a
directly-assembled arrow matrix whose Schur complement is computed the slow,
obvious way.
"""

import numpy as np
import pytest

from ..information import (
    InformationBudget,
    SegmentInformation,
    schur_information,
)


SEED = 20260925
SHARED = ["A1", "EPS1", "EPS2"]


def _random_information(rng, size):
    """A random symmetric positive-definite matrix of the given size."""
    root = rng.normal(size=(size, size))
    return root @ root.T + size * np.eye(size)


@pytest.fixture
def segments():
    """Three segments sharing 3 parameters, with 2, 3 and 4 nuisance parameters."""
    rng = np.random.default_rng(SEED)
    out = []
    for name, n_nuisance in zip("abc", (2, 3, 4)):
        joint = _random_information(rng, len(SHARED) + n_nuisance)
        out.append(schur_information(joint, len(SHARED), name=name))
    return out


def test_schur_information_splits_raw_from_effective():
    """``raw`` is the shared block untouched; ``effective`` is it minus the nuisance coupling."""
    joint = np.array(
        [
            [10.0, 2.0, 1.0],
            [2.0, 8.0, 3.0],
            [1.0, 3.0, 4.0],
        ]
    )
    info = schur_information(joint, 1, name="one")
    assert info.raw == pytest.approx(np.array([[10.0]]))
    # [2, 1] @ inv([[8, 3], [3, 4]]) @ [2, 1]^T
    coupling = np.array([2.0, 1.0]) @ np.linalg.solve(
        np.array([[8.0, 3.0], [3.0, 4.0]]), np.array([2.0, 1.0])
    )
    assert info.effective == pytest.approx(np.array([[10.0 - coupling]]))


def test_effective_never_exceeds_raw(segments):
    """Fitting local spin parameters can only remove information, never add it."""
    for segment in segments:
        assert np.all(np.diag(segment.effective) <= np.diag(segment.raw) + 1e-12)
        assert np.all(segment.dilution > 0.0)
        assert np.all(segment.dilution <= 1.0 + 1e-12)


def test_total_equals_the_full_arrow_matrix_schur_complement(segments):
    """The per-segment sum is exact, because no nuisance block couples two segments.

    This is the property the whole module rests on: assembling the full
    (shared + every segment's nuisance) information matrix and eliminating all
    the nuisance parameters at once gives the same answer as summing each
    segment's own Schur complement.
    """
    n_shared = len(SHARED)
    sizes = [segment.n_nuisance for segment in segments]
    total_size = n_shared + sum(sizes)
    arrow = np.zeros((total_size, total_size))

    start = n_shared
    for segment, size in zip(segments, sizes):
        stop = start + size
        arrow[:n_shared, :n_shared] += segment.raw
        arrow[:n_shared, start:stop] = segment.coupling
        arrow[start:stop, :n_shared] = segment.coupling.T
        arrow[start:stop, start:stop] = segment.nuisance
        start = stop

    shared = arrow[:n_shared, :n_shared]
    cross = arrow[:n_shared, n_shared:]
    nuisance = arrow[n_shared:, n_shared:]
    expected = shared - cross @ np.linalg.solve(nuisance, cross.T)

    budget = InformationBudget(SHARED, segments)
    assert budget.total == pytest.approx(expected)


def test_uncertainties_invert_the_total(segments):
    """Sigma on each shared parameter is the square root of the inverse's diagonal."""
    budget = InformationBudget(SHARED, segments)
    expected = np.sqrt(np.diag(np.linalg.inv(budget.total)))
    assert list(budget.uncertainties().values()) == pytest.approx(expected)
    assert list(budget.uncertainties()) == SHARED


def test_holding_a_parameter_fixed_shrinks_the_others(segments):
    """Fixing EPS1/EPS2 drops their rows before inversion, so A1 can only improve."""
    budget = InformationBudget(SHARED, segments)
    free = budget.uncertainties()["A1"]
    held = budget.uncertainties(fixed=["EPS1", "EPS2"])["A1"]
    assert held <= free
    assert list(budget.uncertainties(fixed=["EPS1", "EPS2"])) == ["A1"]
    # With no covariance left to marginalise over, it is just 1/sqrt(I_A1A1).
    assert held == pytest.approx(1.0 / np.sqrt(budget.total[0, 0]))


def test_fixing_an_unknown_parameter_is_an_error(segments):
    budget = InformationBudget(SHARED, segments)
    with pytest.raises(ValueError, match="TASC"):
        budget.uncertainties(fixed=["TASC"])


def test_without_matches_a_budget_built_without_that_segment(segments):
    """Leave-one-out is a subtraction, and agrees with rebuilding from the rest."""
    budget = InformationBudget(SHARED, segments)
    dropped = budget.without("b")
    rebuilt = InformationBudget(SHARED, [s for s in segments if s.name != "b"])
    assert dropped.total == pytest.approx(rebuilt.total)
    assert [s.name for s in dropped.segments] == ["a", "c"]


def test_without_an_unknown_segment_is_an_error(segments):
    budget = InformationBudget(SHARED, segments)
    with pytest.raises(ValueError, match="zzz"):
        budget.without("zzz")


def test_dropping_a_segment_cannot_improve_a_measurement(segments):
    """Throwing data away never shrinks an error bar -- the point of the exercise."""
    budget = InformationBudget(SHARED, segments)
    for segment in segments:
        reduced = budget.without(segment.name).uncertainties()
        for par, value in budget.uncertainties().items():
            assert reduced[par] >= value - 1e-12


def test_ranking_is_sorted_and_normalised(segments):
    """Contributions to one parameter are fractions of the total, largest first."""
    budget = InformationBudget(SHARED, segments)
    ranking = budget.ranking("A1")
    fractions = [fraction for _, fraction in ranking]
    assert fractions == sorted(fractions, reverse=True)
    assert sum(fractions) == pytest.approx(1.0)
    assert sorted(name for name, _ in ranking) == ["a", "b", "c"]


def test_correlation_has_unit_diagonal_and_matches_the_covariance(segments):
    budget = InformationBudget(SHARED, segments)
    correlation = budget.correlation()
    covariance = np.linalg.inv(budget.total)
    sigma = np.sqrt(np.diag(covariance))
    assert np.diag(correlation) == pytest.approx(np.ones(len(SHARED)))
    assert correlation == pytest.approx(covariance / np.outer(sigma, sigma))


def test_a_singular_nuisance_block_names_the_segment():
    """A segment too short to pin its own spin parameters must say which one it is."""
    joint = np.eye(4)
    joint[2:, 2:] = 0.0  # two nuisance parameters, no curvature at all
    with pytest.raises(np.linalg.LinAlgError, match="stubborn"):
        schur_information(joint, 2, name="stubborn")


def test_segment_information_rejects_an_asymmetric_matrix():
    joint = np.array([[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        schur_information(joint, 1, name="lopsided")


def test_a_segment_with_no_nuisance_parameters_is_undiluted():
    """With nothing local to fit, the raw information survives intact."""
    joint = np.array([[4.0, 1.0], [1.0, 9.0]])
    info = schur_information(joint, 2, name="bare")
    assert info.n_nuisance == 0
    assert info.effective == pytest.approx(joint)
    assert info.dilution == pytest.approx([1.0, 1.0])


def test_budget_rejects_segments_of_the_wrong_width(segments):
    with pytest.raises(ValueError, match="3 shared parameters"):
        InformationBudget(["A1", "EPS1"], segments)


def test_duplicate_segment_names_are_rejected(segments):
    with pytest.raises(ValueError, match="twice"):
        InformationBudget(SHARED, [segments[0], segments[0]])


def test_cumulative_reaches_one_and_counts_segments(segments):
    """The cumulative curve answers 'how many segments carry 90% of this?'."""
    budget = InformationBudget(SHARED, segments)
    names, cumulative = budget.cumulative("A1")
    assert cumulative[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cumulative) >= 0.0)
    assert names == [name for name, _ in budget.ranking("A1")]
    assert budget.n_segments_for("A1", 1.0) == 3


def test_n_segments_for_needs_a_fraction_below_one(segments):
    budget = InformationBudget(SHARED, segments)
    with pytest.raises(ValueError, match="between 0 and 1"):
        budget.n_segments_for("A1", 1.5)


def test_segment_information_is_immutable(segments):
    with pytest.raises(AttributeError):
        segments[0].name = "renamed"


def test_segment_repr_names_itself_and_its_size(segments):
    text = repr(segments[0])
    assert "a" in text
    assert "3" in text  # three shared parameters


def test_dilution_of_a_heavily_coupled_segment_is_small():
    """When the local fit can nearly reproduce the shared signal, little survives."""
    # A shared parameter whose column is almost exactly a nuisance column.
    design = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-3]])
    joint = design.T @ design
    info = schur_information(joint, 1, name="degenerate")
    assert info.dilution[0] < 1e-3


class TestSegmentInformationConstruction:
    """The dataclass is normally built by :func:`schur_information`, but not always."""

    def test_direct_construction_keeps_what_it_is_given(self):
        raw = np.array([[2.0]])
        effective = np.array([[1.0]])
        coupling = np.array([[1.0]])
        nuisance = np.array([[1.0]])
        info = SegmentInformation("x", raw, effective, coupling, nuisance)
        assert info.n_shared == 1
        assert info.n_nuisance == 1
        assert info.dilution == pytest.approx([0.5])
