r"""``ell1info``: what each observation contributes to the shared orbital fit.

This is the command-line face of :mod:`ell1fit.information` and
:mod:`ell1fit.observed_information`. It takes the files and parfiles an
``ell1fit`` run would take, with the same flags, and instead of fitting
anything it reports the curvature of the likelihood those inputs define:

* the 1-sigma uncertainty each shared orbital parameter would come back with;
* how that changes when the eccentricity is held at zero instead of fitted;
* which observations the information comes from, and how much of each one's
  raw information survives its own spin fit;
* the eccentricity upper limit the data can support.

None of it needs a sampler, so a question that would cost a week of MCMC --
"would dropping the short segments matter?", "how much is the eccentricity
costing me on ``A1``?" -- costs a minute instead.

The eccentricity limit without a chain
--------------------------------------

``EPS1`` and ``EPS2`` are the two Cartesian components of a vector whose length
is the eccentricity, so a bound on :math:`e` is a statement about a plane, not
about either component. Given their covariance, the bound follows by drawing
from that two-dimensional Gaussian centred on zero and taking a percentile of
:math:`\sqrt{\epsilon_1^2 + \epsilon_2^2}` -- the same construction
:mod:`ell1fit.eccentricity` applies to real posterior samples, and for the same
reason: the two marginals must never be added in quadrature, because that
throws away their correlation and ignores that a length cannot come out
negative.

When the two components are equally well measured and uncorrelated the answer
is analytic, :data:`RAYLEIGH_THREE_SIGMA` times their common sigma. They
usually are not: a single observation samples one direction in the plane far
better than the other, so the limit is set mostly by the worse one and the
sampling is worth doing properly.

What this is not
----------------

A curvature calculation at one point, under the model and the priors it is
handed. It cannot see a systematic -- if the same photons give a different
``A1`` when blocked two different ways, both answers have small error bars and
this tool reports both of them faithfully. For that, refit.
"""

import argparse

import numpy as np

from .information import InformationBudget
from .observed_information import DEFAULT_STEP, segment_information, split_shared_and_local


__all__ = [
    "RAYLEIGH_THREE_SIGMA",
    "format_budget",
    "main",
    "predicted_eccentricity_limit",
]


#: Radius holding 99.73% of a symmetric two-dimensional Gaussian's mass, in
#: units of one component's standard deviation. The one-dimensional
#: three-sigma point is 3; the radius of a two-dimensional noise cloud is not
#: one of its components, and comes out further at 3.439.
RAYLEIGH_THREE_SIGMA = float(np.sqrt(-2.0 * np.log(1.0 - 0.9973)))

#: Enough draws that the 99.73rd percentile is stable to a few parts in a
#: thousand, which is finer than any of these limits deserve to be quoted.
N_ECCENTRICITY_SAMPLES = 400_000


def predicted_eccentricity_limit(budget, level=0.9973, n_samples=None, seed=None):
    """Upper limit on the eccentricity implied by an information budget.

    Parameters
    ----------
    budget : InformationBudget
        Must cover both ``EPS1`` and ``EPS2``.
    level : float, optional
        Credible level of the limit. The default is the mass a Gaussian
        carries within three standard deviations.
    n_samples : int or None, optional
        Draws used for the percentile.
    seed : int or None, optional
        Seed for the draws, for reproducible output.

    Returns
    -------
    float
    """
    names = list(budget.parameter_names)
    if not {"EPS1", "EPS2"} <= set(names):
        raise ValueError(
            f"An eccentricity limit needs both EPS1 and EPS2 in the budget; it covers {names}"
        )
    free, covariance = budget.covariance()
    index = [free.index("EPS1"), free.index("EPS2")]
    block = covariance[np.ix_(index, index)]

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.zeros(2), block, size=n_samples or N_ECCENTRICITY_SAMPLES)
    return float(np.percentile(np.hypot(draws[:, 0], draws[:, 1]), 100.0 * level))


def _sigma_table(budget):
    """Predicted uncertainties, with and without the eccentricity free."""
    lines = ["Predicted 1-sigma uncertainties", "-------------------------------"]
    free = budget.uncertainties()
    for name, value in free.items():
        lines.append(f"  {name:<8} {value:.4g}")

    eccentricity = [n for n in ("EPS1", "EPS2") if n in budget.parameter_names]
    if eccentricity and len(eccentricity) < len(budget.parameter_names):
        held = budget.uncertainties(fixed=eccentricity)
        lines.append("")
        lines.append(f"  With {', '.join(eccentricity)} fixed at zero:")
        for name, value in held.items():
            cost = free[name] / value
            lines.append(f"    {name:<8} {value:.4g}   ({cost:.2f}x better)")
    return lines


def _segment_table(budget, parameter):
    """Per-segment shares, dilutions and leave-one-out uncertainties."""
    index = list(budget.parameter_names).index(parameter)
    baseline = budget.uncertainties()[parameter]

    lines = [
        f"Where the information on {parameter} comes from",
        "-" * (len(parameter) + 34),
        f"  {'segment':<44} {'share':>7} {'surviving':>10} {'sigma if dropped':>18}",
    ]
    for name, share in budget.ranking(parameter):
        segment = next(s for s in budget.segments if s.name == name)
        dropped = budget.without(name).uncertainties()[parameter]
        label = name if len(name) <= 44 else "..." + name[-41:]
        lines.append(
            f"  {label:<44} {100 * share:6.2f}% {segment.dilution[index]:10.4f}"
            f" {dropped:12.4g} ({dropped / baseline:.2f}x)"
        )

    for fraction in (0.9, 0.99):
        count = budget.n_segments_for(parameter, fraction)
        lines.append(
            f"  Segments carrying {100 * fraction:.0f}% of the {parameter} information: "
            f"{count} of {len(budget.segments)}"
        )
    return lines


def _correlation_table(budget):
    names = list(budget.parameter_names)
    correlation = budget.correlation()
    lines = ["Correlation of the shared parameters", "------------------------------------"]
    lines.append("  " + " " * 8 + "".join(f"{name:>9}" for name in names))
    for i, name in enumerate(names):
        lines.append(
            f"  {name:<8}" + "".join(f"{correlation[i, j]:9.3f}" for j in range(len(names)))
        )
    return lines


def format_budget(budget, parameter=None, seed=None):
    """Render an information budget as the text ``ell1info`` prints.

    Parameters
    ----------
    budget : InformationBudget
    parameter : str or None, optional
        Which shared parameter the per-segment table is about. Defaults to the
        first one.
    seed : int or None, optional
        Seed for the eccentricity limit's draws.
    """
    parameter = parameter or list(budget.parameter_names)[0]
    lines = [
        f"Information budget over {len(budget.segments)} segments "
        f"for {', '.join(budget.parameter_names)}",
        "",
    ]
    lines += _sigma_table(budget)
    lines.append("")

    if {"EPS1", "EPS2"} <= set(budget.parameter_names):
        limit = predicted_eccentricity_limit(budget, seed=seed)
        lines += [
            "Eccentricity",
            "------------",
            f"  Implied 3-sigma (99.73%) upper limit on e: {limit:.3g}",
            "  From the joint EPS1-EPS2 covariance, not the two marginals in quadrature.",
            "",
        ]

    if len(budget.parameter_names) > 1:
        lines += _correlation_table(budget)
        lines.append("")
    lines += _segment_table(budget, parameter)
    return "\n".join(lines)


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Report how much each observation constrains the shared orbital "
            "parameters of a multi-file ell1fit run, without running the fit."
        )
    )
    parser.add_argument("files", help="Input event files", type=str, nargs="+")
    parser.add_argument("-p", "--parfiles", type=str, nargs="+", required=True)
    parser.add_argument(
        "-P",
        "--parameters",
        type=str,
        default="F0,F1,A1,EPS1,EPS2",
        help="Comma-separated parameters the fit would free",
    )
    parser.add_argument(
        "--shared",
        type=str,
        default=None,
        help="Comma-separated subset of the shared parameters to report on",
    )
    parser.add_argument("-N", "--nharm", type=int, default=1)
    parser.add_argument("-e", "--energy-range", type=float, nargs=2, default=None)
    parser.add_argument("--use-weight", action="store_true", default=False)
    parser.add_argument("--use-pi", action="store_true", default=False)
    parser.add_argument("--ignore-uncertainties", action="store_true", default=False)
    parser.add_argument(
        "--no-priors",
        action="store_true",
        default=False,
        help=(
            "Differentiate the likelihood alone. The difference from the default is "
            "the part of the orbital constraint that comes from the spin priors "
            "rather than from the photons."
        ),
    )
    parser.add_argument("--report", type=str, default=None, help="Parameter for the segment table")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(args=None):
    """Entry point for ``ell1info``."""
    from .pipeline import prepare_fit_state

    args = _parser().parse_args(args)
    state = prepare_fit_state(
        args.files,
        args.parfiles,
        nharm=args.nharm,
        energy_range=args.energy_range,
        fit_parameters=args.parameters.split(","),
        use_weight=args.use_weight,
        use_pi=args.use_pi,
        ignore_uncertainties=args.ignore_uncertainties,
    )
    shared = args.shared.split(",") if args.shared else None
    segments = segment_information(
        state.observations,
        state.setup,
        shared=shared,
        include_priors=not args.no_priors,
        step=args.step,
    )
    names = shared or split_shared_and_local(state.setup.parameter_names)[0]
    budget = InformationBudget(names, segments)
    print(format_budget(budget, parameter=args.report, seed=args.seed))
    return budget
