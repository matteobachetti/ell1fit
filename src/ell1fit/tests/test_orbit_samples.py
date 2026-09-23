r"""Turning a flattened chain back into physical samples of any parameter.

The sampler works in local coordinates -- offsets from a starting value, in
units of that parameter's preconditioned scale -- so every plot or summary that
wants light-seconds, seconds or MJD has to undo that first. The conversion
itself is one multiply and one add; what these tests pin down is everything
around it: which column belongs to which parameter, which parameters may be
missing without that being an error, and which mismatches must stay errors.
"""

import numpy as np
import pytest
from astropy.table import Table

from ..eccentricity import (
    ParameterNotSampled,
    eps_samples_from_chain,
    physical_samples_from_chain,
)


SEED = 20260905

#: One column per parameter, with a deliberately different scale each, so a
#: swapped column or a shared factor would show up as a wrong answer.
INITIAL = {"A1": 26.5, "PB": 218668.4, "TASC": 57000.25, "EPS1": 1e-3, "EPS2": -2e-3}
FACTOR = {"A1": 1e-6, "PB": 1e-3, "TASC": 1e-8, "EPS1": 1e-5, "EPS2": 2e-5}


def _chain_and_row(parameters, size=2000, seed=SEED):
    """A local-coordinate chain, plus the results row that describes it."""
    rng = np.random.default_rng(seed)
    local = rng.normal(size=(size, len(parameters)))
    row = {}
    for i, par in enumerate(parameters):
        for perc in (16, 50, 84):
            row[f"d{par}_{perc}"] = float(np.percentile(local[:, i], perc))
        row[f"d{par}_initial"] = INITIAL[par]
        row[f"d{par}_factor"] = FACTOR[par]
    return local, row, [f"d{par}" for par in parameters]


def test_each_column_is_undone_with_its_own_initial_and_factor():
    parameters = ["A1", "PB", "EPS1", "EPS2"]
    local, row, labels = _chain_and_row(parameters)

    samples = physical_samples_from_chain(row, local, parameters, labels=labels)

    assert list(samples) == parameters
    for i, par in enumerate(parameters):
        assert np.allclose(samples[par], INITIAL[par] + local[:, i] * FACTOR[par])


def test_the_eps_pair_helper_is_the_general_converter():
    """``eps_samples_from_chain`` must keep returning exactly what it did."""
    parameters = ["A1", "EPS1", "EPS2"]
    local, row, labels = _chain_and_row(parameters)

    eps1, eps2 = eps_samples_from_chain(row, local, labels=labels)
    samples = physical_samples_from_chain(row, local, ["EPS1", "EPS2"], labels=labels)

    assert np.array_equal(eps1, samples["EPS1"])
    assert np.array_equal(eps2, samples["EPS2"])


def test_columns_are_found_without_labels_too():
    """An HDF5 chain carries no names; the recorded percentiles identify it."""
    parameters = ["A1", "PB", "EPS1", "EPS2"]
    local, row, _ = _chain_and_row(parameters)

    samples = physical_samples_from_chain(row, local, parameters, labels=None)

    for i, par in enumerate(parameters):
        assert np.allclose(samples[par], INITIAL[par] + local[:, i] * FACTOR[par])


@pytest.mark.parametrize("with_labels", [True, False])
def test_a_parameter_that_was_not_fitted_is_left_out_when_not_strict(with_labels):
    parameters = ["A1", "EPS1", "EPS2"]
    local, row, labels = _chain_and_row(parameters)

    samples = physical_samples_from_chain(
        row,
        local,
        ["A1", "PB", "TASC", "EPS1", "EPS2"],
        labels=labels if with_labels else None,
        strict=False,
    )

    assert list(samples) == parameters


@pytest.mark.parametrize("with_labels", [True, False])
def test_a_parameter_that_was_not_fitted_is_an_error_when_strict(with_labels):
    parameters = ["A1", "EPS1", "EPS2"]
    local, row, labels = _chain_and_row(parameters)

    with pytest.raises(ParameterNotSampled, match="PB"):
        physical_samples_from_chain(
            row, local, ["A1", "PB"], labels=labels if with_labels else None, strict=True
        )


def test_a_sampled_parameter_with_no_recorded_scaling_is_treated_as_missing():
    """Without ``initial`` and ``factor`` there is nothing to convert with."""
    parameters = ["A1", "EPS1", "EPS2"]
    local, row, labels = _chain_and_row(parameters)
    del row["dA1_initial"]

    with pytest.raises(ParameterNotSampled, match="dA1_initial"):
        physical_samples_from_chain(row, local, ["A1"], labels=labels)

    assert list(physical_samples_from_chain(row, local, ["A1"], labels=labels, strict=False)) == []


def test_a_chain_from_a_different_fit_is_still_an_error_when_not_strict():
    """Forgiving a missing parameter must not forgive a mismatched file."""
    local, row, _ = _chain_and_row(["EPS1", "EPS2"])
    row["dEPS1_16"] += 50.0
    row["dEPS1_50"] += 50.0
    row["dEPS1_84"] += 50.0

    with pytest.raises(ValueError, match="No column of the chain matches"):
        physical_samples_from_chain(row, local, ["EPS1"], strict=False)


def test_an_astropy_row_works_as_well_as_a_dict():
    parameters = ["A1", "EPS1", "EPS2"]
    local, row, labels = _chain_and_row(parameters)

    samples = physical_samples_from_chain(Table(rows=[row])[0], local, parameters, labels=labels)

    assert np.allclose(samples["A1"], INITIAL["A1"] + local[:, 0] * FACTOR["A1"])
