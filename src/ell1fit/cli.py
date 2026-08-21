"""Command-line interface for the ``ell1fit`` script.

Argument parsing and nothing else: every option here maps onto a parameter of
:func:`ell1fit.pipeline.ell1fit`, which is where the work happens and where the
behaviour is documented.
"""

import argparse

from .likelihoods import pletsch_clarke_likelihood
from .likelihoods import rayleigh_as_likelihood
from .logging import configure_logging
from .pipeline import ell1fit


__all__ = [
    "main",
]


def main(args=None):
    """Main function called by the `ell1fit` script"""
    configure_logging()

    description = "Fit an ELL1 model and frequency derivatives to an X-ray pulsar observation."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "-p",
        "--parfile",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Input parameter files, one per event file. Must contain a simple ELL1 binary model, "
            "with no orbital derivatives, and a number of spin derivatives (F0, F1, ...). "
            "All other models will be ignored."
        ),
    )
    parser.add_argument("-o", "--outroot", type=str, default=None, help="Root of output file names")
    parser.add_argument(
        "-N",
        "--nharm",
        type=int,
        help="Number of harmonics to describe the pulse profile",
        default=1,
    )
    parser.add_argument(
        "--deorb-tolerance",
        type=float,
        help="Tolerance of deorbit operation, in seconds",
        default=1e-8,
    )
    parser.add_argument(
        "-E",
        "--erange",
        nargs=2,
        type=float,
        help="Energy range",
        default=None,
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        help="Maximum number of MCMC steps",
        default=100_000,
    )
    parser.add_argument(
        "-P",
        "--parameters",
        type=str,
        help="Comma-separated list of parameters to fit",
        default="F0,F1",
    )
    parser.add_argument(
        "--likelihood",
        type=str,
        help="Can be PC (Pletsch & Clarke, default) or Rayleigh",
        default="PC",
    )
    parser.add_argument(
        "--minimize-first",
        action="store_true",
        default=False,
        help="Minimize first, then MCMC (don't trust the solution in the par file)",
    )
    parser.add_argument(
        "--use-weight",
        action="store_true",
        default=False,
        help="Use pulse energy dependence of profile as weight",
    )
    parser.add_argument(
        "--use-pi",
        action="store_true",
        default=False,
        help=(
            "Base pulsed-fraction weighting (--use-weight) on PI channels instead of "
            "calibrated energy. No effect without --use-weight."
        ),
    )
    parser.add_argument(
        "--template-iterations",
        type=int,
        default=1,
        help=(
            "Maximum passes of iterative template refinement. Each pass refolds "
            "with the current best solution and rebuilds the pulse template, so "
            "the template is not the one smeared by errors in the input parfile. "
            "1 (default) disables refinement."
        ),
    )
    parser.add_argument("--ignore-uncertainties", action="store_true", default=False)

    args = parser.parse_args(args)
    files = args.files
    parfiles = args.parfile

    like = pletsch_clarke_likelihood
    if args.likelihood.lower() == "rayleigh":
        like = rayleigh_as_likelihood

    ell1fit(
        files,
        parfiles,
        nsteps=args.nsteps,
        nharm=args.nharm,
        tolerance=args.deorb_tolerance,
        energy_range=args.erange,
        fit_parameters=args.parameters.split(","),
        minimize_first=args.minimize_first,
        general_outroot=args.outroot,
        likelihood_func=like,
        use_weight=args.use_weight,
        use_pi=args.use_pi,
        ignore_uncertainties=args.ignore_uncertainties,
        template_iterations=args.template_iterations,
    )


if __name__ == "__main__":  # pragma: no cover
    # Without this, "python -m ell1fit.cli ..." silently does nothing at all.
    main()
