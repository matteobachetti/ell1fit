#!/usr/bin/env python
"""Profile-likelihood forecast for a global parameter, without running a chain.

Why this is a tool and not a test
---------------------------------
It answers a planning question -- "can this dataset reach an interesting
number?" -- and the answer depends on the dataset, not on the code. There is
nothing to assert. Its *agreement with the MCMC* is what the suite checks, on
the simulated fixture in ``ell1fit/tests/test_recovery.py``.

Why profile and not Fisher
--------------------------
The obvious forecast is the inverse Hessian at the best fit. On this posterior
it does not work. The conditional width (:math:`H_{ii}` alone) is stable to
better than a percent across finite-difference steps spanning a decade and a
half; the *marginal* width, which is the one a limit needs, moves by a factor
of eight over the same range. ``A1`` and ``A1DOT`` are correlated at ~0.8 and
the degeneracy between them is curved, so the Schur complement that produces
the marginal is a near cancellation whose value depends on how far out the
curvature was sampled.

Profiling makes no quadratic assumption and inverts no matrix: fix the
parameter, re-optimize everything else, read the width where the profile drops
by 0.5. On the two-epoch A1DOT fixture that reproduces the MCMC's own sigma to
1.3%, at a few dozen local optimizations instead of a full chain.

Usage
-----
::

    python tools/profile_likelihood.py ep1.nc ep2.nc ... -p ep1.par ep2.par ... \
        -P F0,F1,TASC,A1,A1DOT --parameter A1DOT --span 8e-10 --ngrid 21 \
        --use-weight --template-iterations 3

``--span`` is in the parameter's own physical units, measured from the solution
the setup lands on. Widen it until the profile falls by well over 2 at both
ends; a profile that stays flat means the other parameters are absorbing the
one being scanned, and no width can be read off it.

See ``docs/ell1fit/orbital_derivatives.rst`` for the full method and for the
M82 X-2 numbers this produced.
"""

import argparse
import dataclasses
import logging
import time

import numpy as np
from scipy.optimize import minimize

from ell1fit.events import _load_events_for_all_files
from ell1fit.fitting import _bounds_in_local_coordinates
from ell1fit.likelihoods import pletsch_clarke_likelihood
from ell1fit.models import _build_parameters_from_models, _load_and_validate_models
from ell1fit.outputs import _get_outroots, _make_outroot_getter
from ell1fit.pipeline import (
    _build_profiles_and_weights,
    _prepare_fit_setup,
    _prepare_templates_and_phase_priors,
)
from ell1fit.posterior import _build_posterior_functions, _trace_phase_0_likelihood
from ell1fit.refinement import refine_templates_and_solution
from ell1fit.scaling import precondition_factors
from ell1fit.setup_types import ObservationSet

SEC_PER_YEAR = 365.25 * 86400.0

#: Drop in log-posterior defining a one-sigma half-width and a 95% one-sided
#: bound, for one parameter of interest.
DELTA_1SIGMA = 0.5
DELTA_95 = 1.92


def build_setup(
    files,
    parfiles,
    requested,
    nharm=1,
    use_weight=True,
    template_iterations=1,
    ignore_uncertainties=True,
    outroot=None,
    tolerance=1e-8,
):
    """Reproduce the pipeline's setup stage and stop before the sampler.

    Every step below is in the same order as :func:`ell1fit.pipeline.ell1fit`.
    The order matters: the ``Phase_i`` trace has to run before preconditioning,
    and refinement after it, or the local coordinate system is centred on the
    wrong point and the profile is scanned in the wrong units.
    """
    n_files = len(files)
    nbin = max(32, nharm * 8)
    requested = sorted(requested)
    model, pepoch, ref_model = _load_and_validate_models(parfiles)
    get_outroot = _make_outroot_getter(
        files,
        requested,
        None,
        nharm,
        pletsch_clarke_likelihood,
        use_weight,
        use_pi=False,
        general_outroot=outroot,
    )
    times, obs_length, energies, expo = _load_events_for_all_files(
        files, None, pepoch, get_outroot, use_pi=False
    )
    observations = ObservationSet(
        files=files,
        models=model,
        ref_model=ref_model,
        pepoch=pepoch,
        times_from_pepoch=times,
        energies=energies,
        exposures=expo,
        observation_length=obs_length,
    )
    parameters_with_unc, parameters = _build_parameters_from_models(
        model, ref_model, obs_length, ignore_uncertainties=ignore_uncertainties
    )
    profile, profile_weight, weights = _build_profiles_and_weights(
        times, parameters, energies, n_files, get_outroot, use_weight, nbin, tolerance
    )
    template_func, _, parameters, parameters_with_unc = _prepare_templates_and_phase_priors(
        profile,
        profile_weight,
        use_weight,
        nharm,
        get_outroot,
        files,
        weights,
        nbin,
        parameters,
        parameters_with_unc,
    )
    setup = _prepare_fit_setup(
        parameters,
        requested,
        pletsch_clarke_likelihood,
        parameters_with_unc,
        obs_length,
        model,
        template_funcs=template_func,
        weights=weights if use_weight else None,
        tolerance=tolerance,
    )
    _trace_phase_0_likelihood(observations, setup, outroot=_get_outroots(get_outroot, n_files)[-1])
    setup = setup.with_baseline_from(setup.parameters)
    setup = dataclasses.replace(
        setup,
        factors=precondition_factors(
            _build_posterior_functions(observations, setup)[2],
            setup.factors,
            setup.n_parameters,
        ),
    )
    if template_iterations > 1:
        setup, _ = refine_templates_and_solution(
            observations, setup, nbin=nbin, nharm=nharm, max_iterations=template_iterations
        )
    return observations, setup, ref_model


def profile_over(observations, setup, name, local_values):
    """Maximize the log-posterior over every parameter except ``name``.

    Each grid point warm-starts from the previous one's solution, which is what
    keeps the scan smooth: a cold start occasionally settles in a different
    local optimum and puts a step in the profile.
    """
    _, _, func = _build_posterior_functions(observations, setup)
    n = setup.n_parameters
    idx = setup.parameter_names.index(name)
    bounds = _bounds_in_local_coordinates(
        setup.baseline_values, setup.factors, setup.logprior_funcs
    )
    free = [k for k in range(n) if k != idx]
    free_bounds = [bounds[k] for k in free]

    logp = []
    warm = np.zeros(n - 1)
    for value in local_values:

        def negative(sub, value=value):
            full = np.empty(n)
            full[idx] = value
            full[free] = sub
            return -func(full)

        result = minimize(negative, warm, bounds=free_bounds)
        warm = result.x
        logp.append(-result.fun)
    return np.asarray(logp)


def width_from_profile(values, logp, delta=DELTA_1SIGMA):
    """Return ``(peak, half_width, low, high)`` in local units.

    ``nan`` for anything the grid does not bracket -- widen ``--span``.
    """
    peak = int(np.argmax(logp))
    target = logp[peak] - delta

    def crossing(order):
        for a, b in zip(order[:-1], order[1:]):
            if (logp[a] - target) * (logp[b] - target) <= 0:
                frac = (logp[a] - target) / (logp[a] - logp[b])
                return values[a] + frac * (values[b] - values[a])
        return np.nan

    low = crossing(list(range(peak, -1, -1)))
    high = crossing(list(range(peak, len(values))))
    return values[peak], (high - low) / 2, low, high


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", help="event files")
    parser.add_argument("-p", "--parfiles", nargs="+", required=True)
    parser.add_argument("-P", "--fit-parameters", default="F0,F1,TASC,A1,A1DOT")
    parser.add_argument("--parameter", default="A1DOT", help="the one to profile")
    parser.add_argument("--span", type=float, required=True, help="half-range, physical units")
    parser.add_argument("--ngrid", type=int, default=21)
    parser.add_argument("-N", "--nharm", type=int, default=1)
    parser.add_argument("--use-weight", action="store_true")
    parser.add_argument("--template-iterations", type=int, default=3)
    parser.add_argument("--keep-uncertainties", action="store_true")
    parser.add_argument("-o", "--outroot", default=None)
    args = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO)
    start = time.time()
    observations, setup, ref_model = build_setup(
        args.files,
        args.parfiles,
        args.fit_parameters.split(","),
        nharm=args.nharm,
        use_weight=args.use_weight,
        template_iterations=args.template_iterations,
        ignore_uncertainties=not args.keep_uncertainties,
        outroot=args.outroot,
    )
    logging.info(
        "Setup built in %.0f s, %d free parameters", time.time() - start, setup.n_parameters
    )

    idx = setup.parameter_names.index(args.parameter)
    factor, base = setup.factors[idx], setup.baseline_values[idx]
    grid = np.linspace(-args.span, args.span, args.ngrid) / factor

    start = time.time()
    logp = profile_over(observations, setup, args.parameter, grid)
    logging.info("Profile scanned in %.0f s", time.time() - start)

    print(f"\n--- profile likelihood in {args.parameter} ---")
    for local, value in zip(grid, logp):
        print(f"  {base + local * factor:+.5e}  {value - logp.max():+.4f}")

    peak, half, _, _ = width_from_profile(grid, logp)
    _, _, low95, high95 = width_from_profile(grid, logp, delta=DELTA_95)
    print(f"\npeak      = {base + peak * factor:+.4e}")
    print(f"1-sigma   = {half * factor:.4e}")
    print(f"95% range = [{base + low95 * factor:+.4e}, {base + high95 * factor:+.4e}]")

    if args.parameter == "A1DOT" and getattr(ref_model, "PBDOT", None) is not None:
        # (2/3)(A1/PB)PBDOT: what conservative mass transfer predicts from the
        # orbital-period derivative the same parfile carries.
        pb_seconds = float(ref_model.PB.value) * 86400.0
        expected = (
            (2.0 / 3.0) * (float(ref_model.A1.value) / pb_seconds) * float(ref_model.PBDOT.value)
        )
        print(
            f"\nKepler expectation = {expected:.4e} lt-s/s = {expected * SEC_PER_YEAR:.4e} lt-s/yr"
        )
        print(f"expectation / sigma = {abs(expected) / (half * factor):.3f}")


if __name__ == "__main__":
    main()
