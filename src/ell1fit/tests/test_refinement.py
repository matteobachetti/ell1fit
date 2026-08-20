"""Tests for iterative template refinement.

The substantive claim being tested is that refolding with an improved solution
and rebuilding the template reduces bias when the input parfile is wrong. That
is measured properly -- over many realizations -- in
``tools/`` experiments rather than here; what these tests pin down is the
behaviour a future change could silently break:

* one iteration is exactly equivalent to not refining,
* refinement improves the fold rather than degrading it,
* the best iterate is kept even when the last one is worse,
* non-convergence is reported rather than passing silently.
"""

import dataclasses

import numpy as np
import pytest

from ell1fit.ell1fit import _prepare_fit_setup
from ell1fit.events import _load_events_for_all_files
from ell1fit.likelihoods import pletsch_clarke_likelihood
from ell1fit.models import _build_parameters_from_models, _load_and_validate_models
from ell1fit.phase_utils import folded_profile, phases_around_zero
from ell1fit.refinement import _profile_score, refine_templates_and_solution
from ell1fit.setup_types import ObservationSet
from ell1fit.templates import create_template_from_profile_harm, get_template_func

from .datagen import make_multi_epoch_dataset

EPOCH_OFFSETS = (0.0, 37.0)
NHARM = 2
NBIN = max(32, NHARM * 8)


@pytest.fixture(scope="module")
def offset_setup(tmp_path_factory):
    """Pipeline state built from a parfile whose ``A1`` is deliberately wrong.

    The templates here are folded with that wrong solution, so they carry the
    smearing that refinement exists to remove.
    """
    outdir = str(tmp_path_factory.mktemp("refine"))
    dataset = make_multi_epoch_dataset(
        outdir,
        epoch_offsets=EPOCH_OFFSETS,
        n_events=2500,
        duration=100_000.0,
        offsets={"A1": 0.02},
        uncertainties={"A1": 1e-1, "F0": 1e-7},
        prefix="refine",
    )

    models, pepoch, ref_model = _load_and_validate_models(dataset["par_files"])

    def get_outroot(file_n=None):
        return f"{outdir}/refine_{file_n}"

    times, obs_length, energies, exposures = _load_events_for_all_files(
        dataset["event_files"], None, pepoch, get_outroot
    )
    observations = ObservationSet(
        files=dataset["event_files"],
        models=models,
        ref_model=ref_model,
        pepoch=pepoch,
        times_from_pepoch=times,
        energies=energies,
        exposures=exposures,
        observation_length=obs_length,
    )

    parameters_with_unc, parameters = _build_parameters_from_models(models, ref_model, obs_length)

    profiles = folded_profile(times, parameters, nbin=NBIN)
    template_funcs = []
    for i, profile in enumerate(profiles):
        template, phase = create_template_from_profile_harm(
            profile, nharm=NHARM, final_nbin=200, plot=False
        )
        template_funcs.append(get_template_func(template))
        offset = -phases_around_zero(phase)
        parameters[f"Phase_{i}"] = offset
        parameters_with_unc[f"Phase_{i}"][0] = offset

    setup = _prepare_fit_setup(
        parameters,
        ["A1", "F0"],
        pletsch_clarke_likelihood,
        parameters_with_unc,
        obs_length,
        models,
        template_funcs=template_funcs,
        weights=None,
        tolerance=1e-8,
    )
    return dataset, observations, setup


def test_single_iteration_is_a_no_op(offset_setup):
    """``--template-iterations 1`` must not touch the setup at all.

    This is what makes the default safe to ship: enabling the feature without
    asking for it changes nothing.
    """
    _, observations, setup = offset_setup

    refined, history = refine_templates_and_solution(
        observations, setup, nbin=NBIN, nharm=NHARM, max_iterations=1
    )

    assert refined is setup, "one iteration must return the identical object"
    assert history == []


def test_refinement_improves_the_fold(offset_setup):
    """Refolding with the refined solution must concentrate the pulse better."""
    _, observations, setup = offset_setup

    before = _profile_score(
        folded_profile(
            observations.times_from_pepoch, setup.parameters, nbin=NBIN, tolerance=setup.tolerance
        ),
        NHARM,
    )

    refined, history = refine_templates_and_solution(
        observations, setup, nbin=NBIN, nharm=NHARM, max_iterations=3
    )

    after = _profile_score(
        folded_profile(
            observations.times_from_pepoch,
            refined.parameters,
            nbin=NBIN,
            tolerance=refined.tolerance,
        ),
        NHARM,
    )

    assert history, "refinement should have run at least one pass"
    assert after > before, (
        f"refinement degraded the fold: Z^2 went from {before:.1f} to {after:.1f}"
    )


def test_refinement_moves_a1_toward_the_truth(offset_setup):
    """The wrongly-set parameter must end up closer to its injected value."""
    dataset, observations, setup = offset_setup
    truth = dataset["solution"].A1

    started_at = setup.parameters["A1"]
    assert abs(started_at - truth) > 1e-2, "the fixture parfile was not actually offset"

    refined, _ = refine_templates_and_solution(
        observations, setup, nbin=NBIN, nharm=NHARM, max_iterations=3
    )

    assert abs(refined.parameters["A1"] - truth) < abs(started_at - truth), (
        f"A1 did not improve: started {started_at!r}, ended "
        f"{refined.parameters['A1']!r}, truth {truth!r}"
    )


def test_best_iterate_is_kept_not_the_last(offset_setup, monkeypatch):
    """A pass that degrades the fold must not be the one returned.

    Refinement is not guaranteed to improve monotonically, so the loop scores
    each pass and keeps the best. Here the score is forced to collapse after the
    first pass; the returned setup must be the good one.
    """
    _, observations, setup = offset_setup

    # The first call scores the starting fold; every later call scores a
    # refinement pass. Making the baseline unbeatable means no pass should win.
    scores = iter([500.0, 1.0, 1.0, 1.0])

    monkeypatch.setattr(
        "ell1fit.refinement._profile_score",
        lambda profiles, nharm: next(scores, 1.0),
    )

    refined, history = refine_templates_and_solution(
        observations, setup, nbin=NBIN, nharm=NHARM, max_iterations=3
    )

    assert history, "refinement should still have run its passes"
    assert refined is setup, "a pass that degraded the fold was allowed to win"


def test_non_convergence_is_reported(offset_setup, caplog):
    """Hitting the iteration cap without converging must warn, not pass silently."""
    _, observations, setup = offset_setup

    with caplog.at_level("WARNING"):
        _, history = refine_templates_and_solution(
            observations,
            setup,
            nbin=NBIN,
            nharm=NHARM,
            max_iterations=2,
            # Impossible threshold, so convergence can never be declared.
            tolerance=0.0,
        )

    assert len(history) == 2
    assert not any(entry["converged"] for entry in history)
    assert any("did not converge" in message for message in caplog.messages)


def test_history_records_each_pass(offset_setup):
    """The history must expose enough to inspect convergence, not just assert it."""
    _, observations, setup = offset_setup

    _, history = refine_templates_and_solution(
        observations, setup, nbin=NBIN, nharm=NHARM, max_iterations=3
    )

    for entry in history:
        assert set(entry) == {"iteration", "score", "max_shift", "converged"}
        assert np.isfinite(entry["score"])
        assert np.isfinite(entry["max_shift"])


def test_setup_is_not_mutated_in_place(offset_setup):
    """Refinement must build new bundles rather than mutate the caller's.

    FitSetup is frozen precisely so one pass's state cannot leak into the next;
    this pins that the loop honours it.
    """
    _, observations, setup = offset_setup
    original_templates = setup.template_funcs
    original_baseline = list(setup.baseline_values)

    refined, _ = refine_templates_and_solution(
        observations, setup, nbin=NBIN, nharm=NHARM, max_iterations=3
    )

    assert setup.template_funcs is original_templates
    assert list(setup.baseline_values) == original_baseline
    assert dataclasses.is_dataclass(refined)
