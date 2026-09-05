"""Tests for :mod:`ell1fit.limits`, the signed-parameter posterior summary.

Everything here is checked against samples drawn from a distribution whose
percentiles are known analytically, so the assertions are about the function's
arithmetic rather than about any sampler's behaviour.
"""

import numpy as np
import pytest

from ..limits import signed_parameter_summary


#: Enough samples that a percentile of a standard normal is reproduced to
#: better than the tolerances used below, without slowing the suite down.
N_SAMPLES = 400_000
SEED = 20260905


@pytest.fixture(scope="module")
def standard_normal():
    return np.random.default_rng(SEED).normal(0.0, 1.0, N_SAMPLES)


def test_upper_limit_is_the_magnitude_percentile(standard_normal):
    """95% of a standard normal's mass has |x| < 1.96."""
    summary = signed_parameter_summary(standard_normal, "PBDDOT", detected=False)
    assert summary["PBDDOT_upper_limit"] == pytest.approx(1.959964, rel=0.01)
    assert summary["PBDDOT_upper_limit_level"] == 0.95


def test_upper_limit_level_is_configurable(standard_normal):
    summary = signed_parameter_summary(
        standard_normal, "PBDDOT", upper_limit_level=0.99, detected=False
    )
    assert summary["PBDDOT_upper_limit"] == pytest.approx(2.575829, rel=0.01)
    assert summary["PBDDOT_upper_limit_level"] == 0.99


def test_one_and_two_sigma_intervals_are_gaussian_equivalent(standard_normal):
    """The intervals are quoted at the credible levels a Gaussian 1 and 2 sigma
    carry (68.27% and 95.45%), not at the rounded 16/84 and 2.5/97.5."""
    summary = signed_parameter_summary(standard_normal, "PBDDOT", detected=False)
    assert summary["PBDDOT_1sigma_lo"] == pytest.approx(-1.0, abs=0.01)
    assert summary["PBDDOT_1sigma_hi"] == pytest.approx(1.0, abs=0.01)
    assert summary["PBDDOT_2sigma_lo"] == pytest.approx(-2.0, abs=0.02)
    assert summary["PBDDOT_2sigma_hi"] == pytest.approx(2.0, abs=0.02)


def test_median_is_reported(standard_normal):
    summary = signed_parameter_summary(standard_normal + 7.0, "PBDOT", detected=True)
    assert summary["PBDOT_50"] == pytest.approx(7.0, abs=0.01)


def test_zero_centred_posterior_has_no_significance(standard_normal):
    summary = signed_parameter_summary(standard_normal, "PBDDOT", detected=False)
    assert summary["PBDDOT_significance_sigma"] < 0.1
    assert summary["PBDDOT_zero_credibility"] < 0.1


def test_offset_posterior_significance_counts_standard_deviations(standard_normal):
    """A posterior five standard deviations away from zero excludes it at 5 sigma."""
    summary = signed_parameter_summary(standard_normal + 5.0, "PBDOT", detected=True)
    assert summary["PBDOT_significance_sigma"] == pytest.approx(5.0, rel=0.02)
    assert summary["PBDOT_zero_credibility"] > 0.999999


def test_significance_is_sign_blind(standard_normal):
    """A negative PBDOT is just as detected as a positive one."""
    positive = signed_parameter_summary(standard_normal + 5.0, "PBDOT", detected=True)
    negative = signed_parameter_summary(-standard_normal - 5.0, "PBDOT", detected=True)
    assert negative["PBDOT_significance_sigma"] == pytest.approx(
        positive["PBDOT_significance_sigma"], rel=1e-6
    )


def test_detected_parameter_quotes_no_upper_limit(standard_normal):
    """A limit is not the thing to quote once the parameter is measured."""
    summary = signed_parameter_summary(standard_normal + 5.0, "PBDOT", detected=True)
    assert np.isnan(summary["PBDOT_upper_limit"])
    assert summary["PBDOT_detected"] is True
    assert "PBDOT =" in summary["PBDOT_summary"]
    assert "upper limit" not in summary["PBDOT_summary"]


def test_undetected_parameter_summary_reads_as_a_limit(standard_normal):
    summary = signed_parameter_summary(standard_normal, "PBDDOT", detected=False)
    assert summary["PBDDOT_detected"] is False
    assert "|PBDDOT| <" in summary["PBDDOT_summary"]
    assert "upper limit" in summary["PBDDOT_summary"]
    assert "not a measurement" in summary["PBDDOT_summary"]


def test_unit_appears_in_the_summary_line(standard_normal):
    summary = signed_parameter_summary(standard_normal, "PBDDOT", detected=False, unit="1/yr")
    assert "1/yr" in summary["PBDDOT_summary"]


def test_two_sigma_interval_is_reported_for_an_undetected_parameter(standard_normal):
    """The limit alone hides the sign information; the interval carries it."""
    summary = signed_parameter_summary(standard_normal, "PBDDOT", detected=False)
    assert "2 sigma" in summary["PBDDOT_summary"]


def test_sample_count_is_recorded(standard_normal):
    summary = signed_parameter_summary(standard_normal, "PBDDOT", detected=False)
    assert summary["PBDDOT_nsamples"] == N_SAMPLES


def test_too_few_samples_is_an_error():
    with pytest.raises(ValueError, match="at least"):
        signed_parameter_summary([1.0, 2.0], "PBDOT", detected=False)


def test_non_finite_samples_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        signed_parameter_summary([1.0, 2.0, np.nan, 4.0], "PBDOT", detected=False)


def test_every_value_is_json_serializable(standard_normal):
    """The summary goes straight into ell1decay's results JSON."""
    import json

    summary = signed_parameter_summary(standard_normal, "PBDDOT", detected=False)
    json.dumps(summary)
