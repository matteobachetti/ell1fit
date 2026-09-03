"""Tests for carrying one global orbital solution to each file's own epoch.

The fit uses a single binary -- one ``PB``, ``TASC``, ``A1``, ``EPS1``,
``EPS2`` -- but those values are referenced to one epoch, and orbital
derivatives carry them away from it. :func:`ell1fit.models._orbital_epoch_offsets`
supplies the fixed per-file corrections that keep them valid everywhere.

The reference used here is the exact ELL1 orbit count,

.. math:: N(t) = x - \\tfrac{1}{2}\\,\\dot{P_b}\\,x^2, \\quad x = (t - T_{asc})/P_b

written out directly rather than obtained from PINT or from the code under
test, so agreement is evidence rather than a tautology -- the same reasoning
that keeps :mod:`ell1fit.tests.datagen` independent of ``phase_utils``.
"""

import os
import re

import numpy as np
import pytest

from ..models import _build_parameters_from_models, _load_and_validate_models
from ..phase_utils import _calculate_phases, phases_around_zero

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

OFFSET_NAMES = ("TASC_offset", "PB_offset", "A1_offset", "EPS1_offset", "EPS2_offset")


def _write_parfile(path, pepoch, **overrides):
    """Copy the shipped parfile, moving ``PEPOCH`` and setting extra values."""
    text = open(os.path.join(datadir, "events0.par")).read()
    text = re.sub(r"^PEPOCH.*$", f"PEPOCH {pepoch}", text, flags=re.M)
    for key, value in overrides.items():
        pattern = rf"^{key}\s.*$"
        line = f"{key} {value}"
        text = (
            re.sub(pattern, line, text, flags=re.M)
            if re.search(pattern, text, re.M)
            else (text + f"\n{line}\n")
        )
    open(path, "w").write(text)
    return str(path)


def _parfiles(tmp_path, years_apart, **overrides):
    """Two parfiles for the same solution, ``years_apart`` years of epoch apart."""
    pepochs = [56357.0, 56357.0 + 365.25 * years_apart]
    return [
        _write_parfile(tmp_path / f"epoch{i}.par", pepoch, **overrides)
        for i, pepoch in enumerate(pepochs)
    ]


def _load(tmp_path, years_apart, **overrides):
    """Build the pipeline's parameter dictionary for two separated epochs."""
    parfiles = _parfiles(tmp_path, years_apart, **overrides)
    model, pepoch, ref_model = _load_and_validate_models(parfiles)
    _, parameters = _build_parameters_from_models(model, ref_model, [1e5] * len(parfiles))
    return parameters, pepoch


def _worst_phase_error(parameters, pepoch, pbdot, f0=7.5, span=1e5, use_offsets=True):
    """Largest pulse-phase departure from the exact ELL1 orbit, in cycles.

    A constant offset is removed first: the free ``Phase_i`` absorbs one, so it
    is not an error the fit would ever see.
    """
    pb, tasc, a1 = parameters["PB"], parameters["TASC"], parameters["A1"]
    worst = 0.0

    for i, epoch in enumerate(pepoch):
        times = np.linspace(0, span, 2000)
        x = ((float(epoch) - tasc) * 86400 + times) / pb
        phi_exact = 2 * np.pi * (x - 0.5 * pbdot * x**2)

        pb_i = pb + (parameters[f"PB_offset_{i}"] if use_offsets else 0.0)
        tasc_offset = parameters[f"TASC_offset_{i}"] if use_offsets else 0.0
        raw = (tasc + tasc_offset - float(epoch)) * 86400
        tasc_i = ((raw + 0.5 * pb_i) % pb_i) - 0.5 * pb_i
        phi_local = 2 * np.pi * (times - tasc_i) / pb_i

        difference = a1 * (np.sin(phi_local) - np.sin(phi_exact))
        worst = max(worst, np.max(np.abs(difference - difference.mean())) * f0)

    return worst


def test_offsets_are_exactly_zero_without_orbital_derivatives(tmp_path):
    """No derivative in the parfile must mean no correction at all.

    Exact zeros, not small numbers: this is what keeps every existing result
    bit-for-bit unchanged, and ``tools/refactor_net.py`` depends on it.
    """
    parameters, pepoch = _load(tmp_path, years_apart=3)

    for i in range(len(pepoch)):
        for name in OFFSET_NAMES:
            value = parameters[f"{name}_{i}"]
            assert value == 0.0, f"{name}_{i} is {value!r}, expected exactly 0.0"
            assert not np.signbit(value), f"{name}_{i} is negative zero"


@pytest.mark.parametrize("pbdot", [1e-11, 1e-10, 1e-9])
@pytest.mark.parametrize("years_apart", [1, 10])
def test_pbdot_epoch_propagation_tracks_the_exact_orbit(tmp_path, pbdot, years_apart):
    """The per-file solution must follow the exact quadratic orbit count.

    Also asserts the converse, which is what makes this test mutation-proof:
    with the offsets zeroed -- the behaviour before they existed -- the same
    configuration is wrong by more than the ~1e-3 cycles a fit resolves. If the
    offsets stop being applied, the second assertion fails.
    """
    parameters, pepoch = _load(tmp_path, years_apart, PBDOT=pbdot)

    corrected = _worst_phase_error(parameters, pepoch, pbdot, use_offsets=True)
    uncorrected = _worst_phase_error(parameters, pepoch, pbdot, use_offsets=False)

    assert corrected < 1e-6, f"epoch propagation leaves {corrected:.3g} cycles"
    assert uncorrected > 1e-5, (
        f"the uncorrected error is only {uncorrected:.3g} cycles here, so this "
        "configuration no longer demonstrates anything"
    )
    assert corrected < uncorrected / 100


def test_pb_offset_alone_achieves_nothing(tmp_path):
    """The ``TASC`` correction is the one that does the work.

    Wrapping ``TASC`` modulo the *local* period re-adds ``n * PB_i``, while the
    exact model accumulates ``n * PB + n**2 * PB * PBDOT / 2``. Correcting the
    period on its own flips the sign of the residual while leaving its magnitude
    alone, so it buys nothing: measured here, it improves the error by 0.15%,
    against a factor of 300,000 when both corrections are applied. Anyone
    tempted to ship the obvious half of this fix should see this fail.
    """
    pbdot = 1e-10
    parameters, pepoch = _load(tmp_path, years_apart=3, PBDOT=pbdot)

    half_corrected = dict(parameters)
    for i in range(len(pepoch)):
        half_corrected[f"TASC_offset_{i}"] = 0.0

    both = _worst_phase_error(parameters, pepoch, pbdot)
    period_only = _worst_phase_error(half_corrected, pepoch, pbdot)
    neither = _worst_phase_error(parameters, pepoch, pbdot, use_offsets=False)

    assert period_only == pytest.approx(neither, rel=0.05)
    assert both < neither / 1000


def test_a1dot_is_honoured_as_an_input(tmp_path):
    """``A1DOT`` in the parfile must reach the per-file ``A1``.

    Unlike the other derivatives it does not arrive as a fixed offset: it is
    the *trial* ``A1DOT`` times a fixed per-file lever arm ``binary_dt_i``, so
    that it can be fitted. The size is unchanged -- ``A1DOT`` times the epoch
    separation -- and so is the sign structure, since the lever arms straddle
    the reference epoch.
    """
    a1dot = 1e-13  # lt-s per second
    years_apart = 4
    parameters, pepoch = _load(tmp_path, years_apart, A1DOT=a1dot)

    separation = (float(pepoch[1]) - float(pepoch[0])) * 86400
    drift = [parameters["A1DOT"] * parameters[f"binary_dt_{i}"] for i in range(2)]

    assert parameters["A1DOT"] == pytest.approx(a1dot)
    # PINT propagates over dt_integer_orbits -- the separation rounded to a
    # whole number of orbits -- so the two differ by up to one period, here
    # 0.17% of the four-year baseline.
    assert drift[1] - drift[0] == pytest.approx(a1dot * separation, rel=5e-3)
    assert drift[0] < 0 < drift[1]

    # The offset that used to carry it is now empty: A1 is propagated by A1DOT
    # alone, so nothing is left for the fixed correction to do.
    assert parameters["A1_offset_0"] == 0.0
    assert parameters["A1_offset_1"] == 0.0


def test_a1dot_reproduces_pints_own_propagated_a1(tmp_path):
    """The lever arm must give exactly the ``A1`` PINT would have propagated.

    The whole point of expressing the drift as ``A1DOT * binary_dt_i`` rather
    than as a precomputed offset is that the trial value can move. That is only
    legitimate if, *at* the parfile's own ``A1DOT``, the phases are the ones
    PINT's ``change_binary_epoch`` implies -- which is what this compares,
    at the phase level rather than on the parameter, so a convention error
    anywhere between the two cannot hide.

    The injected ``A1DOT`` is deliberately enormous (1e-8 lt-s/s moves ``A1``
    by 1.3 lt-s over the baseline): a value at the real target precision would
    leave the two phase sets equal to within numerical noise whether or not the
    lever arm were applied at all.
    """
    import copy

    a1dot = 1e-8  # lt-s per second: a gross, unmistakable drift
    parfiles = _parfiles(tmp_path, years_apart=4, A1DOT=a1dot)
    model, pepoch, ref_model = _load_and_validate_models(parfiles)
    _, parameters = _build_parameters_from_models(model, ref_model, [1e5] * len(parfiles))

    times = [np.linspace(0, 1e5, 500) for _ in pepoch]
    phases = _calculate_phases(times, parameters)

    for i, epoch in enumerate(pepoch):
        propagated = copy.deepcopy(ref_model)
        propagated.change_binary_epoch(epoch)

        # One file, with PINT's own propagated A1 substituted for the global
        # value and no lever arm at all.
        single = {
            "PB": parameters["PB"],
            "A1": float(propagated.A1.value),
            "TASC": parameters["TASC"],
            "EPS1": parameters["EPS1"],
            "EPS2": parameters["EPS2"],
            "PEPOCH_0": parameters[f"PEPOCH_{i}"],
            "Phase_0": parameters[f"Phase_{i}"],
            "TASC_offset_0": parameters[f"TASC_offset_{i}"],
            "PB_offset_0": parameters[f"PB_offset_{i}"],
        }
        count = 0
        while f"F{count}_{i}" in parameters:
            single[f"F{count}_0"] = parameters[f"F{count}_{i}"]
            count += 1

        reference = _calculate_phases([times[i]], single)[0]
        assert np.allclose(phases_around_zero(phases[i] - reference), 0, atol=1e-9)


def test_offsets_are_present_for_every_file(tmp_path):
    """Every file gets a full set of offsets, whether or not they are nonzero."""
    parameters, pepoch = _load(tmp_path, years_apart=2, PBDOT=1e-10)

    for i in range(len(pepoch)):
        for name in OFFSET_NAMES:
            assert f"{name}_{i}" in parameters


def test_offsets_do_not_collide_with_result_field_names(tmp_path):
    """Offset keys must not shadow the fit's own output fields.

    ``optimize_solution`` merges the parameter dictionary into the results, so a
    key that matches a result field silently overwrites it. An earlier naming of
    these offsets as ``dA1_1`` did exactly that, replacing the first percentile
    of the fitted ``A1``. Percentile fields are ``d<par>_<percentile>``.
    """
    parameters, _ = _load(tmp_path, years_apart=2, PBDOT=1e-10)
    percentiles = {"1", "10", "16", "50", "84", "90", "99"}

    for key in parameters:
        if not key.startswith("d"):
            continue
        _, _, suffix = key.rpartition("_")
        assert suffix not in percentiles, f"{key} can shadow a percentile result field"


def test_calculate_phases_actually_applies_the_offsets(tmp_path):
    """The offsets must reach the phases, not merely sit in the dictionary.

    Every other test here checks that the offsets are computed *correctly*; this
    one checks that ``_calculate_phases`` uses them. Without it, deleting the
    lookups in ``phase_utils`` leaves the whole suite green -- which is exactly
    what happened when this module was first written.

    The size of the change is asserted, not just its presence: it must match the
    error that the uncorrected model was measured to carry.
    """
    pbdot = 1e-10
    parameters, pepoch = _load(tmp_path, years_apart=3, PBDOT=pbdot)
    times = [np.linspace(0, 1e5, 2000) for _ in pepoch]

    without_offsets = {k: v for k, v in parameters.items() if "_offset_" not in k}
    corrected = _calculate_phases(times, parameters)
    uncorrected = _calculate_phases(times, without_offsets)

    worst = 0.0
    for a, b in zip(corrected, uncorrected):
        difference = phases_around_zero(a - b)
        worst = max(worst, np.max(np.abs(difference - difference.mean())))

    predicted = _worst_phase_error(parameters, pepoch, pbdot, use_offsets=False)
    assert worst == pytest.approx(predicted, rel=0.2), (
        f"the offsets moved the phases by {worst:.3g} cycles, but the error they "
        f"correct was measured at {predicted:.3g}"
    )


@pytest.fixture(scope="module")
def pbdot_datasets(tmp_path_factory):
    """Two epochs four years either side of the reference, with and without PBDOT.

    Both are generated from the same seed, so the pair isolates the effect of
    the derivative itself. Neither epoch sits at the reference: an epoch that
    did would be generated identically whatever ``PBDOT`` is, and would test
    only half of the propagation.

    ``PBDOT = 5e-9`` is larger than a typical redback shows. It is chosen to
    make the effect unmistakable in a test that must not be flaky, not because
    it is representative; :doc:`the design notes </ell1fit/design>` carry the
    measured scaling for realistic values.
    """
    from .datagen import InjectedSolution, make_multi_epoch_dataset

    common = dict(
        epoch_offsets=(-1461.0, 1461.0),
        n_events=6000,
        phase0=(0.35, 0.35),
        seed=7,
    )
    root = tmp_path_factory.mktemp("pbdot")
    with_pbdot = root / "with"
    without_pbdot = root / "without"
    with_pbdot.mkdir()
    without_pbdot.mkdir()

    return {
        "with_pbdot": make_multi_epoch_dataset(
            str(with_pbdot), solution=InjectedSolution(PBDOT=5e-9), **common
        ),
        "without_pbdot": make_multi_epoch_dataset(
            str(without_pbdot), solution=InjectedSolution(PBDOT=0.0), **common
        ),
    }


def _folded_z2(dataset, use_offsets=True, nharm=2, nbin=32):
    """Fold each epoch with its parfile solution and return the per-epoch Z^2."""
    from stingray.pulse.pulsar import z_n_binned_events

    from ..phase_utils import folded_profile

    model, pepoch, ref_model = _load_and_validate_models(dataset["par_files"])
    _, parameters = _build_parameters_from_models(model, ref_model, [1e5] * len(pepoch))
    if not use_offsets:
        parameters = {k: v for k, v in parameters.items() if "_offset_" not in k}

    times = [epoch["times_from_pepoch"] for epoch in dataset["epochs"]]
    profiles = folded_profile(times, parameters, nbin=nbin)
    return [float(z_n_binned_events(profile, nharm)) for profile in profiles]


def test_pbdot_events_fold_as_sharply_as_without_pbdot(pbdot_datasets):
    """Propagation must recover the whole signal, not merely improve it.

    The control is the same data generated with ``PBDOT = 0``: if the epoch
    offsets are right, a pulsar whose orbit is decaying folds exactly as well as
    one whose orbit is not.
    """
    corrected = _folded_z2(pbdot_datasets["with_pbdot"])
    control = _folded_z2(pbdot_datasets["without_pbdot"])

    for epoch, (got, want) in enumerate(zip(corrected, control)):
        assert got == pytest.approx(want, rel=0.15), (
            f"epoch {epoch}: Z^2 {got:.1f} against {want:.1f} without PBDOT"
        )


def test_ignoring_the_epoch_offsets_smears_pbdot_data(pbdot_datasets):
    """The converse: without the offsets, the pulse is lost, not just blurred.

    This is the end-to-end statement of the whole change, made on generated
    events rather than on parameter dictionaries. Measured over three seeds, the
    uncorrected fold retains 3-15% of the corrected Z^2 -- for four harmonics'
    worth of degrees of freedom, that is consistent with no detection at all.
    """
    corrected = _folded_z2(pbdot_datasets["with_pbdot"])
    uncorrected = _folded_z2(pbdot_datasets["with_pbdot"], use_offsets=False)

    for epoch, (good, bad) in enumerate(zip(corrected, uncorrected)):
        assert bad < 0.4 * good, (
            f"epoch {epoch}: ignoring the offsets still gives Z^2 {bad:.1f} "
            f"against {good:.1f}, so this dataset no longer demonstrates anything"
        )


def test_generated_parfiles_carry_the_injected_pbdot(pbdot_datasets):
    """The generator must write ``PBDOT`` out, or the pipeline cannot honour it.

    It also writes each epoch's *local* period, since a parfile's ``PB`` is the
    one in force at the ``TASC`` it quotes. Both epochs therefore carry a
    different ``PB``, which is what a constant-``PB`` generator would get wrong.
    """
    dataset = pbdot_datasets["with_pbdot"]
    model, _, _ = _load_and_validate_models(dataset["par_files"])

    for one in model:
        assert one.PBDOT.value == pytest.approx(5e-9)

    periods = [float(one.PB.value) for one in model]
    solution = dataset["solution"]
    separation = (dataset["epochs"][1]["TASC"] - dataset["epochs"][0]["TASC"]) * 86400
    assert periods[1] - periods[0] == pytest.approx(solution.PBDOT * separation / 86400, rel=1e-6)
