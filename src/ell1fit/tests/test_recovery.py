"""End-to-end tests that the pipeline recovers a *known* injected solution.

Why these assertions are statistical and not golden numbers
-----------------------------------------------------------
The obvious way to protect a numerical pipeline against regressions is to pin
its output: run it once, check the numbers in, and assert equality forever
after. That is a poor fit here, for two reasons.

First, the MCMC path runs through the RNG stream, and
``get_autocorr_time`` -> burn-in -> thinning -> percentiles amplifies any
difference into a visible one. Pinned percentiles would fire on every numpy or
emcee upgrade, and the fix would always be "regenerate the reference" -- which
trains everyone to regenerate it without looking.

Second, and worse: a golden-number test cannot distinguish *correct* from
*consistently wrong*. If a bias predates the reference file, the test locks the
bias in permanently and reports success.

So these tests assert physics instead: the fit must land on the injected truth,
to within the uncertainty it quotes for itself. That survives RNG and library
drift, and it is the only formulation that can detect bias -- which is the whole
point of the adaptive-template work these tests exist to validate.

For verifying that a *refactor* changed nothing, bit-comparison is the right
tool, but it belongs in a same-machine before/after dump, not in a checked-in
test. See ``tools/refactor_net.py``.

A trap for anyone adding a ``TASC`` assertion here
--------------------------------------------------
``TASC`` is only defined modulo ``PB``, and
:func:`ell1fit.models._load_and_validate_models` re-references the shared model
to the mean ``PEPOCH``. With a multi-epoch dataset that shifts the reported
value by a whole number of orbits -- 17 of them, about 43 days, for the epochs
used here -- so comparing a fitted ``TASC`` directly against the injected one
fails by an enormous margin while the fit is in fact correct. Reduce the
difference modulo ``PB`` before asserting on it.
"""

import numpy as np
import pytest
from astropy.table import Table

from ell1fit.ell1fit import main as main_ell1fit
from ell1fit.phase_utils import folded_profile
from stingray.pulse.pulsar import z_n_binned_events

from .datagen import make_multi_epoch_dataset

EPOCH_OFFSETS = (0.0, 37.0)


@pytest.fixture(scope="module")
def rich_dataset(tmp_path_factory):
    """A high-count dataset for the detection check, which does no fitting.

    Kept separate from :func:`dataset` because generating events is cheap while
    fitting them is not: the detection check wants a comfortable margin over
    the noise floor, and the fitting tests want to stay fast.
    """
    outdir = str(tmp_path_factory.mktemp("detect"))
    return make_multi_epoch_dataset(
        outdir,
        epoch_offsets=EPOCH_OFFSETS,
        n_events=5000,
        duration=100_000.0,
        prefix="detect",
    )


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A two-epoch synthetic dataset whose parfiles are deliberately wrong.

    ``A1`` is written 4e-3 lt-s away from the truth -- several times the
    uncertainty the fit achieves -- so that a fit which merely sat still at its
    starting point would fail. The parfile uncertainties are deliberately loose
    so the priors do not fight the data; the point here is to test the
    likelihood, not the priors.
    """
    outdir = str(tmp_path_factory.mktemp("recovery"))
    return make_multi_epoch_dataset(
        outdir,
        epoch_offsets=EPOCH_OFFSETS,
        n_events=1500,
        duration=60_000.0,
        offsets={"A1": 4e-3},
        uncertainties={"A1": 1e-2, "F0": 1e-8},
    )


def _parameter_dict(solution, epochs, **override):
    """Build the parameter mapping :func:`folded_profile` expects.

    Values are numpy scalars because ``phase_utils._mjd_to_sec`` calls
    ``.astype`` on its result and so rejects plain Python floats.
    """
    parameters = {
        "PB": np.float64(solution.PB_sec),
        "A1": np.float64(solution.A1),
        "TASC": np.float64(solution.TASC),
        "EPS1": np.float64(solution.EPS1),
        "EPS2": np.float64(solution.EPS2),
        "PBDOT": np.float64(0.0),
    }
    for i, epoch in enumerate(epochs):
        parameters[f"F0_{i}"] = np.float64(epoch["F0"])
        parameters[f"F1_{i}"] = np.float64(epoch["F1"])
        parameters[f"PEPOCH_{i}"] = np.float64(epoch["pepoch"])
        parameters[f"Phase_{i}"] = np.float64(0.0)
    parameters.update({k: np.float64(v) for k, v in override.items()})
    return parameters


def _fitted(table, parameter):
    """Return ``(median, sigma, point_estimate)`` in physical units.

    The pipeline reports every fitted parameter in the local coordinate system
    ``physical = local * factor + initial``, so undo that here.
    """
    initial = table[f"d{parameter}_initial"]
    factor = table[f"d{parameter}_factor"]
    median = table[f"d{parameter}_50"] * factor + initial
    sigma = (table[f"d{parameter}_84"] - table[f"d{parameter}_16"]) / 2 * factor
    point_estimate = table[f"rough_d{parameter}"] * factor + initial
    return median, sigma, point_estimate


def test_injected_signal_is_detectable_at_the_truth(rich_dataset):
    """The generator's forward model must invert under the package's deorbit.

    This is the cross-check that makes every other test in this file
    meaningful: :mod:`ell1fit.tests.datagen` implements the ELL1 Roemer delay
    independently, in plain numpy. If the two implementations disagree about a
    sign or a reference epoch, the pulse will not appear here.

    The threshold is set against the noise floor rather than against a measured
    value: under the null hypothesis :math:`Z^2_2` follows a chi-squared
    distribution with 4 degrees of freedom, whose 99.9th percentile is about
    18.5. This dataset yields 65-85, so the margin is comfortable. Mutating the
    generator's orbital delay by as little as 2% in amplitude, or flipping its
    sign, drops it to 2-12 and trips this assertion.
    """
    solution = rich_dataset["solution"]
    epochs = rich_dataset["epochs"]
    times = [epoch["times_from_pepoch"] for epoch in epochs]

    def z2(**override):
        parameters = _parameter_dict(solution, epochs, **override)
        return np.array(
            [z_n_binned_events(p, 2) for p in folded_profile(times, parameters, nbin=32)]
        )

    at_truth = z2()
    # A quarter of a light-second of orbital error smears the pulse away.
    wrong_a1 = z2(A1=solution.A1 + 0.5)
    no_orbit = z2(A1=0.0)
    wrong_tasc = z2(TASC=solution.TASC + 0.05)

    # 20 is just above the chi^2(4) 99.9th percentile of ~18.5; see the docstring.
    assert np.all(at_truth > 20), f"pulse not detected at the injected truth: {at_truth}"
    assert np.all(at_truth > 4 * wrong_a1), f"{at_truth} vs {wrong_a1}"
    assert np.all(at_truth > 4 * no_orbit), f"{at_truth} vs {no_orbit}"
    assert np.all(at_truth > 4 * wrong_tasc), f"{at_truth} vs {wrong_tasc}"


def test_pipeline_recovers_injected_solution(dataset, tmp_path):
    """The fit must pull back to the truth from a deliberately wrong parfile."""
    solution = dataset["solution"]
    outroot = str(tmp_path / "recovery")

    main_ell1fit(
        dataset["event_files"]
        + ["-p"]
        + dataset["par_files"]
        + ["-P", "F0,A1", "-N", "2", "--minimize-first", "--nsteps", "300", "-o", outroot]
    )

    table = Table.read(outroot + "_A1_F0_N2_results.ecsv")[-1]

    expected = {"A1": solution.A1}
    for i, day_offset in enumerate(EPOCH_OFFSETS):
        expected[f"F0_{i}"] = solution.spin_at(solution.pepoch_ref + day_offset)[0]

    failures = []
    for parameter, truth in expected.items():
        median, sigma, point_estimate = _fitted(table, parameter)
        assert sigma > 0, f"{parameter} has a non-positive uncertainty {sigma}"
        pull = (median - truth) / sigma
        if abs(pull) > 4:
            failures.append(
                f"{parameter}: fitted {median!r} +- {sigma:.3g}, "
                f"injected {truth!r}, pull {pull:.2f}"
            )
        assert np.isfinite(point_estimate), f"{parameter} point estimate is not finite"

    assert not failures, "parameters inconsistent with the injected truth:\n" + "\n".join(failures)


def test_fit_moves_away_from_the_wrong_starting_value(dataset, tmp_path):
    """A1 starts 4e-3 lt-s off; the fit must actually move, not sit still.

    Without this, ``test_pipeline_recovers_injected_solution`` could pass simply
    because the parfile already held the answer.
    """
    solution = dataset["solution"]
    outroot = str(tmp_path / "moved")

    main_ell1fit(
        dataset["event_files"]
        + ["-p"]
        + dataset["par_files"]
        + ["-P", "F0,A1", "-N", "2", "--minimize-first", "--nsteps", "300", "-o", outroot]
    )

    table = Table.read(outroot + "_A1_F0_N2_results.ecsv")[-1]
    median, sigma, _ = _fitted(table, "A1")

    started_at = table["dA1_initial"]
    assert abs(started_at - solution.A1) > 3e-3, "the test parfile was not actually offset"
    assert abs(median - solution.A1) < abs(started_at - solution.A1), (
        f"fit did not improve on its starting point: started {started_at!r}, "
        f"ended {median!r}, truth {solution.A1!r}"
    )
