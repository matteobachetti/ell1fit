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

from ell1fit.refinement import _profile_score, refine_templates_and_solution

from ell1fit.phase_utils import folded_profile

from .datagen import make_multi_epoch_dataset
from .helpers import build_pipeline_state

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
    observations, setup = build_pipeline_state(dataset, fit_parameters=("A1", "F0"), nharm=NHARM)
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
    # ">=" rather than ">" because that is exactly what the keep-the-best-iterate
    # logic guarantees: the returned setup can never score below the starting
    # one, but nothing promises some pass will beat it on a given realization.
    # How *much* it improves is a statistical claim, measured offline over many
    # realizations rather than asserted from one.
    assert after >= before, (
        f"refinement degraded the fold: Z^2 went from {before:.1f} to {after:.1f}"
    )


def test_refinement_keeps_a1_near_the_truth(offset_setup):
    """Refinement must not send the wrongly-set parameter off into the weeds.

    Note what this does *not* assert: that ``A1`` ends closer to the truth than
    it began. That is true of the distribution but not of any single fit --
    measured over 40 realizations, refinement cuts the bias 5.4x (from 3.4 sigma
    to consistent with zero), yet only about half of individual realizations
    land closer than they started, because the per-fit scatter is comparable to
    the offset being corrected. Asserting the per-realization version produced a
    test that passed locally on two architectures and failed intermittently in
    CI.

    So the statistical claim is measured offline, and what is pinned here is the
    guard-rail: refinement stays in the right neighbourhood rather than
    diverging.
    """
    dataset, observations, setup = offset_setup
    truth = dataset["solution"].A1

    started_at = setup.parameters["A1"]
    assert abs(started_at - truth) > 1e-2, "the fixture parfile was not actually offset"

    refined, _ = refine_templates_and_solution(
        observations, setup, nbin=NBIN, nharm=NHARM, max_iterations=3
    )

    # Generous: several times the per-fit scatter, so only a genuine divergence
    # trips it.
    assert abs(refined.parameters["A1"] - truth) < 0.1, (
        f"A1 diverged: started {started_at!r}, ended {refined.parameters['A1']!r}, truth {truth!r}"
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


def test_comparison_phaseogram_reference_survives_refinement(offset_setup, monkeypatch, tmp_path):
    """The "before" panel must show the starting solution, not the refined one.

    ``optimize_solution`` used to derive its left-hand panel by evaluating the
    posterior at local coordinates zero, which silently means "whatever the
    baseline currently holds". Refinement re-centres the baseline on its own
    result, so the comparison became the refined solution plotted against
    itself -- two identical panels, and a diagnostic that always looks perfect
    however badly the fit went.

    This pins that the reference passed in is the one actually plotted, and
    that it still differs from the fitted phases after refinement has moved the
    solution.
    """
    import numpy as np

    from ell1fit import fitting
    from ell1fit.phase_utils import _calculate_phases

    _, observations, setup = offset_setup

    # The "before" phases, captured while the baseline is still the start.
    reference = _calculate_phases(
        observations.times_from_pepoch, setup.parameters, tolerance=setup.tolerance
    )

    refined, history = refine_templates_and_solution(
        observations, setup, nbin=NBIN, nharm=NHARM, max_iterations=3
    )
    assert history, "refinement should have run, or this proves nothing"

    captured = {}

    def fake_plot(reference_phases, fitted_phases, times, outroots, suffix=""):
        captured[suffix] = (reference_phases, fitted_phases)

    monkeypatch.setattr(fitting, "_plot_phaseogram_set", fake_plot)
    monkeypatch.setattr(
        fitting, "safe_run_sampler",
        lambda *a, **k: {f"d{p}_{q}": 0.0 for p in refined.parameter_names
                         for q in (1, 10, 16, 50, 84, 90, 99)},
    )

    fitting.optimize_solution(
        observations, refined, nsteps=10,
        outroots=[str(tmp_path / f"p{i}") for i in range(observations.n_files + 1)],
        reference_phases=reference,
    )

    assert captured, "the phaseogram comparison was never drawn"
    for suffix, (ref_used, fitted_used) in captured.items():
        for i in range(observations.n_files):
            assert np.array_equal(ref_used[i], reference[i]), (
                f"panel '{suffix}' did not plot the reference it was given"
            )
            assert not np.allclose(ref_used[i], fitted_used[i], rtol=0, atol=1e-9), (
                f"panel '{suffix}' shows the same phases on both sides: the "
                "before-and-after comparison is uninformative"
            )
