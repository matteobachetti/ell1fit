"""Tests for the rule-based priors and for the ``--prior`` command-line overrides.

A prior that silently becomes something other than what was asked for changes
the answer and leaves no trace in the output, so most of these tests are about
malformed input being refused rather than guessed at.
"""

import numpy as np
import pytest
from scipy.stats import norm

from ell1fit.prior_transform import transform_spec_for_prior
from ell1fit.priors import assign_logpriors, parse_prior_spec, parse_prior_specs
from ell1fit.scaling import TARGET_LOCAL_SIGMA

from .datagen import make_multi_epoch_dataset
from .helpers import build_pipeline_state


#: A minimal ``{name: [value, uncertainty]}`` dictionary covering the parameters
#: the prior rules need to look up (``PB`` is read when wrapping ``TASC``).
PARAMETERS = {
    "F0_0": [0.5, np.nan],
    "F1_0": [-1e-12, np.nan],
    "F1_1": [-1e-12, np.nan],
    "PB": [16000.0, np.nan],
    "A1": [1.5, np.nan],
    "TASC": [57000.0, np.nan],
}


def test_absolute_uniform_spec_sets_exactly_those_bounds():
    """``F1:uniform:-1e-10,1e-10`` means that interval literally, not one around F1."""
    spec = parse_prior_spec("F1:uniform:-1e-10,1e-10")
    (logp,) = assign_logpriors(["F1_0"], PARAMETERS, user_priors=[spec])

    assert logp.phys_bounds == (-1e-10, 1e-10)
    assert logp(0.0) == 0
    assert logp(2e-10) == -np.inf


def test_relative_uniform_spec_is_centred_on_the_input_value():
    """``+-WIDTH`` brackets the parameter's input value rather than zero."""
    spec = parse_prior_spec("F1:uniform:+-1e-10")
    (logp,) = assign_logpriors(["F1_0"], PARAMETERS, user_priors=[spec])

    value = PARAMETERS["F1_0"][0]
    assert logp.phys_bounds == pytest.approx((value - 1e-10, value + 1e-10))


def test_normal_spec_produces_a_gaussian_log_prior():
    """``PB:normal:mu,sigma`` evaluates as the Gaussian log-density it names."""
    spec = parse_prior_spec("PB:normal:16000.0,0.5")
    (logp,) = assign_logpriors(["PB"], PARAMETERS, user_priors=[spec])

    assert logp(16000.3) == pytest.approx(norm(loc=16000.0, scale=0.5).logpdf(16000.3))


def test_bare_name_covers_every_per_file_expansion():
    """``F1`` applies to ``F1_0`` and ``F1_1`` alike, since spin parameters are per file."""
    spec = parse_prior_spec("F1:uniform:-1e-10,1e-10")
    logps = assign_logpriors(["F1_0", "F1_1"], PARAMETERS, user_priors=[spec])

    assert [p.phys_bounds for p in logps] == [(-1e-10, 1e-10)] * 2


def test_exact_name_wins_over_the_bare_one():
    """A ``F1_0`` spec overrides the blanket ``F1`` spec for that file only."""
    specs = parse_prior_specs(["F1:uniform:-1e-10,1e-10", "F1_0:uniform:-1e-12,1e-12"])
    first, second = assign_logpriors(["F1_0", "F1_1"], PARAMETERS, user_priors=specs)

    assert first.phys_bounds == (-1e-12, 1e-12)
    assert second.phys_bounds == (-1e-10, 1e-10)


def test_tasc_override_stays_periodic():
    """TASC is an epoch modulo one orbit, so an override must still wrap."""
    spec = parse_prior_spec("TASC:normal:+-1e-5")
    (logp,) = assign_logpriors(["TASC"], PARAMETERS, user_priors=[spec])

    period = PARAMETERS["PB"][0] / 86400.0
    assert logp(57000.0) == pytest.approx(logp(57000.0 + period))


def test_override_beats_the_rule_it_replaces():
    """Without an override F1 gets the improper catch-all uniform; with one it does not."""
    (default,) = assign_logpriors(["F1_0"], PARAMETERS)
    assert default.phys_bounds == (-np.inf, np.inf)

    spec = parse_prior_spec("F1:uniform:-1e-10,1e-10")
    (overridden,) = assign_logpriors(["F1_0"], PARAMETERS, user_priors=[spec])
    assert np.all(np.isfinite(overridden.phys_bounds))


def test_a_bounded_override_makes_nested_sampling_possible():
    """An improper prior has no evidence; bounding F1 from the command line fixes that."""
    (default,) = assign_logpriors(["F1_0"], PARAMETERS)
    with pytest.raises(ValueError, match="improper"):
        transform_spec_for_prior(default)

    spec = parse_prior_spec("F1:uniform:-1e-10,1e-10")
    (overridden,) = assign_logpriors(["F1_0"], PARAMETERS, user_priors=[spec])
    assert transform_spec_for_prior(overridden) is not None


@pytest.mark.parametrize(
    "text,message",
    [
        ("F1:uniform", "NAME:SHAPE:ARGS"),
        ("F1:lorentzian:1,2", "unknown shape"),
        ("F1:uniform:bananas,2", "not numeric"),
        ("F1:uniform:1,2,3", "expected 2 number"),
        ("F1:uniform:+-1,2", "expected 1 number"),
        ("F1:uniform:2,1", "lower bound"),
        ("F1:normal:0,-1", "standard deviation must be positive"),
        ("F1:uniform:+--1e-10", "width must be positive"),
        ("F1:uniform:-inf,inf", "must be finite"),
        (":uniform:1,2", "no parameter name"),
    ],
)
def test_malformed_specs_are_refused_with_a_readable_message(text, message):
    """Every way of writing a spec wrongly names what is wrong with it."""
    with pytest.raises(ValueError, match=message):
        parse_prior_spec(text)


def test_two_priors_for_the_same_parameter_are_refused():
    """A repeated name means the command line disagrees with itself."""
    with pytest.raises(ValueError, match="Two priors given for F1"):
        parse_prior_specs(["F1:uniform:-1e-10,1e-10", "F1:normal:0,1e-11"])


def test_a_prior_on_an_unfitted_parameter_is_refused():
    """A typo in a prior would otherwise run to completion and answer another question."""
    spec = parse_prior_spec("F2:uniform:-1e-10,1e-10")
    with pytest.raises(ValueError, match="not being fitted"):
        assign_logpriors(["F1_0"], PARAMETERS, user_priors=[spec])


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """One short single-epoch observation, enough to build a real fit setup."""
    return make_multi_epoch_dataset(
        str(tmp_path_factory.mktemp("priors")),
        epoch_offsets=(0.0,),
        n_events=3000,
        phase0=(0.35,),
        prefix="prio",
    )


def test_a_narrow_prior_sets_the_local_scale(dataset):
    """A prior tighter than the heuristic scale must shrink the sampler's step.

    The factors are derived from uncertainty heuristics that know nothing about
    the priors, so without this the walkers would be spread over a region the
    prior excludes.
    """
    _, plain = build_pipeline_state(dataset, fit_parameters=("F0", "A1"), nharm=2)
    spec = parse_prior_spec("A1:uniform:+-1e-6")
    _, narrowed = build_pipeline_state(
        dataset, fit_parameters=("F0", "A1"), nharm=2, user_priors=[spec]
    )

    index = plain.parameter_names.index("A1")
    assert narrowed.factors[index] < plain.factors[index]


def test_the_starting_walkers_fit_inside_a_narrow_prior(dataset):
    """The initial ensemble is one local sigma wide, and must land in the support.

    A walker starting outside its own prior sees ``-inf`` and never moves, so
    this is the property that decides whether a tight ``--prior`` produces a
    chain at all.
    """
    spec = parse_prior_spec("A1:uniform:+-1e-6")
    _, setup = build_pipeline_state(
        dataset, fit_parameters=("F0", "A1"), nharm=2, user_priors=[spec]
    )

    index = setup.parameter_names.index("A1")
    logp = setup.logprior_funcs[index]
    spread = setup.factors[index] * TARGET_LOCAL_SIGMA
    centre = setup.baseline_values[index]

    assert np.isfinite(logp(centre - spread))
    assert np.isfinite(logp(centre + spread))
