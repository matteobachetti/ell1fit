#!/usr/bin/env python
"""Effective-samples-per-second harness for the posterior-exploration work.

Why this is a tool and not a test
---------------------------------
Every claim this measures is statistical, and most of them are small. An
effective sample size is itself an estimate with roughly 10-20% scatter, and
wall-clock time depends on the machine, its thermal state and whatever else is
running. Pinning any of that in the test suite would either assert a difference
that is noise or fail on a busy CI runner -- both of which have already happened
in this repository. See ``tools/refactor_net.py`` for the same argument applied
to bitwise reproducibility, and ``ell1fit/tests/`` for what does belong in a
suite.

So this lives here: run it before a sampler change, run it after, and compare.

Requires the ``bench`` extra (``pip install -e .[bench]``) for ArviZ, which
supplies the diagnostics; the package itself never imports it.

Usage
-----
::

    python tools/sampler_bench.py list
    python tools/sampler_bench.py run --problem P1 --sampler emcee -o before.json
    ... change the sampler ...
    python tools/sampler_bench.py run --problem P1 --sampler emcee -o after.json
    python tools/sampler_bench.py compare before.json after.json

What it measures, and why each piece is there
---------------------------------------------
**Effective samples per second** is the headline::

    ess_per_second = min_over_parameters(bulk ESS) / total wall seconds

Three deliberate choices in that one line:

- **min over parameters, not mean.** A sampler that races along ``F0`` while
  sticking on the ``A1``-``Phase`` ridge must not be able to average its way to
  a win. The bottleneck direction is the one that decides how long a real run
  takes.
- **one estimator for every sampler.** emcee's ``get_autocorr_time`` and NUTS's
  built-in effective sample size are different estimators; comparing a number
  from one against a number from the other measures the estimators as much as
  the samplers. Everything here is reduced to a ``(chain, draw, parameter)``
  array and handed to the same rank-normalized ArviZ diagnostics.
- **total wall seconds includes setup and warm-up.** Compilation, tuning and
  burn-in are time the user waits. Excluding them would flatter any sampler that
  front-loads its cost -- which is exactly what a JIT-compiled NUTS does.

**ESS per step and steps per second are reported separately**, because their
product is the headline and the factors say *why* it moved: initialization and
move strategy change how much information a step carries, a batched evaluator
changes how long a step takes. Reporting only the product makes the mechanism
unfalsifiable. ``ess_per_posterior_call`` is the same efficiency with the
parallelism divided out, which is what lets a 32-walker ensemble be compared
against a 4-chain NUTS run at all.

**Convergence is a gate, not a metric.** An unconverged chain reports
gloriously high ESS per second. Any configuration whose R-hat exceeds
``RHAT_GATE`` is marked ``converged: false`` and its rate should be discarded
rather than interpreted.

**Every configuration runs several seeds.** The spread across seeds is reported
alongside the median, and :func:`compare` refuses to call a difference a win
when it is smaller than that spread. This is the whole reason the harness
exists: without it, "faster" is unfalsifiable.

**Speed is not the only axis.** A sampler that explores the ridge badly reports
credible intervals that are too narrow, and it will look fast doing it. So each
run also records posterior quantiles with their Monte Carlo standard errors, the
marginal standard deviations, and the parameter correlation matrix.
:func:`compare` checks agreement in MCSE units -- ``|q_a - q_b| /
sqrt(mcse_a^2 + mcse_b^2)``, which should be of order one -- and separately
checks the ratio of marginal widths and the correlation on the ridge, neither of
which a quantile comparison alone would catch.

The benchmark problems
----------------------
The answer depends on the size of the data. A single posterior call is already
threaded by numba, and how much of the machine that leaves idle is what decides
whether parallelising *across* walkers can win anything: measured on 10 cores,
going from 1 thread to 8 speeds one phase computation up by 1.4x at 5000 events
per file but 4.4x at 200000. A change benchmarked only at fixture scale can
therefore be a win there and a loss in production.

``P1``
    Fixture scale: 2 epochs, 5000 events each, ``F0`` and ``A1`` free. Fast
    enough to iterate on, and the regime where the cores sit idle.
``P2``
    Production scale: the same fit with 200000 events per epoch. This is where a
    real run spends its time.
``P3``
    The hard posterior: 3 epochs, eccentricity free, ten parameters. The
    ``A1``-``Phase`` ridge and the ``EPS`` directions are what a better move
    strategy or a gradient-based sampler is supposed to help with, and this is
    the problem that an evidence calculation would eventually run on.
``P1W``
    ``P1`` with ``--use-weight``. The other three are unweighted, which makes
    them insensitive to the weighting machinery by construction -- useful as a
    regression control, but it means none of them represents a real weighted
    run. Weighting changes the posterior the sampler has to explore: every event
    enters :func:`~ell1fit.likelihoods.pletsch_clarke_likelihood` scaled by a
    number in ``[0, 1]``, which sharpens the peak without changing where it is.
    Deliberately paired with ``P1`` -- same seed, same events, same free
    parameters, so the two differ in exactly one thing and ``compare`` reads as
    the effect of weighting rather than of anything else. The generator's
    pulsed fraction rises with energy
    (:func:`~ell1fit.tests.datagen.pulsed_fraction_at`), so there is a real
    trend for the weights to find; on flat-spectrum data they would be fitting
    noise and the comparison would measure nothing.
"""

import argparse
import contextlib
import dataclasses
import io
import json
import logging
import multiprocessing
import os
import shutil
import statistics
import sys
import tempfile
import time

import numpy as np


#: R-hat above this marks a run unconverged, and its rate uninterpretable.
RHAT_GATE = 1.01

#: Fraction of each chain discarded as warm-up before the diagnostics run.
#: A fixed fraction rather than an autocorrelation-derived burn-in, because it
#: has to mean the same thing for every sampler; the discarded time is still
#: charged to the run.
DISCARD_FRACTION = 0.5

#: Quantiles summarised for the credible-interval agreement check.
QUANTILES = (0.16, 0.5, 0.84)

#: Seed for the synthetic data. Fixed, so every sampler faces the same target
#: distribution; only the sampler's own RNG varies between repetitions.
DATA_SEED = 20260822

#: Flag an agreement failure above this many combined standard errors.
AGREEMENT_SIGMA_GATE = 3.0


@dataclasses.dataclass
class ProblemSpec:
    """The definition of one benchmark posterior."""

    name: str
    doc: str
    n_events: int
    epoch_offsets: tuple
    fit_parameters: tuple
    nharm: int = 2
    default_steps: int = 2000
    use_weight: bool = False


PROBLEMS = {
    "P1": ProblemSpec(
        name="P1",
        doc="fixture scale: 2 epochs x 5k events, F0 and A1 free",
        n_events=5_000,
        epoch_offsets=(0.0, 37.0),
        fit_parameters=("F0", "A1"),
        default_steps=4000,
    ),
    "P2": ProblemSpec(
        name="P2",
        doc="production scale: 2 epochs x 200k events, F0 and A1 free",
        n_events=200_000,
        epoch_offsets=(0.0, 37.0),
        fit_parameters=("F0", "A1"),
        default_steps=1000,
    ),
    "P1W": ProblemSpec(
        name="P1W",
        doc="P1's data and parameters, with energy weighting switched on",
        n_events=5_000,
        epoch_offsets=(0.0, 37.0),
        fit_parameters=("F0", "A1"),
        default_steps=4000,
        use_weight=True,
    ),
    "P3": ProblemSpec(
        name="P3",
        doc="the hard posterior: 3 epochs x 50k events, eccentricity free",
        n_events=50_000,
        epoch_offsets=(0.0, 37.0, 91.0),
        fit_parameters=("A1", "EPS1", "EPS2", "F0", "TASC"),
        default_steps=4000,
    ),
}


@dataclasses.dataclass
class Problem:
    """A built benchmark posterior, ready to sample."""

    spec: ProblemSpec
    logpost: object
    start: np.ndarray
    parameter_names: list
    factors: list
    baseline_values: list

    @property
    def ndim(self):
        """Number of free parameters."""
        return len(self.parameter_names)


@dataclasses.dataclass
class SamplerRun:
    """What a sampler adapter hands back.

    ``chains`` has shape ``(n_chains, n_draws, n_parameters)`` and is *not*
    trimmed: the harness discards its own warm-up so that every sampler is
    treated the same way.
    """

    chains: np.ndarray
    setup_seconds: float
    sample_seconds: float
    extra: dict = dataclasses.field(default_factory=dict)


def machine_configuration():
    """Record what decides the wall-clock numbers, so two files can be compared.

    The thread count is not a detail: a single posterior call is already
    threaded by numba, and how much of the machine that leaves idle depends on
    the event count. Measured here on 10 cores, one phase computation gains 1.4x
    from 8 threads at 5000 events per file and 4.4x at 200000. Comparing a rate
    taken at one thread count against one taken at another measures the machine.
    """
    configuration = {
        "cpu_count": os.cpu_count(),
        "platform": sys.platform,
    }
    try:
        import numba

        configuration["numba_threads"] = int(numba.get_num_threads())
        configuration["numba_version"] = numba.__version__
    except ImportError:
        configuration["numba_threads"] = None
    return configuration


def _cache_dir(spec):
    """Directory holding the generated event and parameter files for a problem."""
    return os.path.join(
        tempfile.gettempdir(),
        "ell1fit_sampler_bench",
        f"{spec.name}_{spec.n_events}_{len(spec.epoch_offsets)}_{DATA_SEED}",
    )


def _ensure_dataset(spec):
    """Generate the problem's synthetic dataset, reusing it if already present.

    The generator is deterministic given ``DATA_SEED``, so a cached dataset is
    the same dataset. Regenerating 200k events per epoch on every repetition
    would otherwise dominate the measurement's own runtime.
    """
    from ell1fit.tests.datagen import make_multi_epoch_dataset

    directory = _cache_dir(spec)
    marker = os.path.join(directory, "COMPLETE")
    if os.path.exists(marker):
        with open(marker) as handle:
            return json.load(handle)

    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory, exist_ok=True)
    dataset = make_multi_epoch_dataset(
        directory,
        epoch_offsets=spec.epoch_offsets,
        n_events=spec.n_events,
        seed=DATA_SEED,
    )
    payload = {
        "event_files": list(dataset["event_files"]),
        "par_files": list(dataset["par_files"]),
    }
    with open(marker, "w") as handle:
        json.dump(payload, handle)
    return payload


def _centre_phases(observations, setup):
    """Move each ``Phase_i`` to its best-fit value, without writing figures.

    This mirrors :func:`ell1fit.pipeline._trace_phase_0_likelihood`, which does
    the same scan but also saves a diagnostic plot per parameter. A benchmark
    should not litter, and the plots cost real time on the larger problems.
    """
    from ell1fit.posterior import trace_likelihood_over_parameter

    parameters = setup.parameters
    for parameter in [p for p in setup.parameter_names if p.startswith("Phase_")]:
        index = setup.parameter_names.index(parameter)
        trace = trace_likelihood_over_parameter(
            observations,
            setup,
            parameter_name=parameter,
            parameter_values=np.linspace(
                parameters[parameter] - setup.factors[index],
                parameters[parameter] + setup.factors[index],
                100,
            ),
        )
        values = list(trace.keys())
        parameters[parameter] = values[int(np.nanargmax(list(trace.values())))]
    return parameters


def build_problem(spec, verbose=False):
    """Build one benchmark posterior, at the point where the MCMC would start.

    This follows :func:`ell1fit.pipeline.ell1fit` stage by stage -- load, fold,
    template, priors and scaling, precondition, point estimate -- and stops
    where that function would call the sampler. It deliberately calls the
    pipeline's own stage helpers rather than reimplementing them, so the
    benchmark measures the posterior the package actually builds. When one of
    them moves or changes signature, this function has to be updated with it;
    that is the same bargain ``refactor_net.py`` makes with its import table.
    """
    import dataclasses as _dataclasses

    from ell1fit.events import _load_events_for_all_files
    from ell1fit.fitting import point_estimate_fit
    from ell1fit.likelihoods import pletsch_clarke_likelihood
    from ell1fit.models import _build_parameters_from_models, _load_and_validate_models
    from ell1fit.outputs import _make_outroot_getter
    from ell1fit.pipeline import (
        _build_profiles_and_weights,
        _prepare_fit_setup,
        _prepare_templates_and_phase_priors,
    )
    from ell1fit.posterior import _build_posterior_functions
    from ell1fit.scaling import precondition_factors
    from ell1fit.setup_types import ObservationSet

    dataset = _ensure_dataset(spec)
    files = dataset["event_files"]
    parfiles = dataset["par_files"]

    workdir = tempfile.mkdtemp(prefix="sampler_bench_build_")
    try:
        model, pepoch, ref_model = _load_and_validate_models(parfiles)
        nbin = max(32, spec.nharm * 8)
        requested = sorted(spec.fit_parameters)
        get_outroot = _make_outroot_getter(
            files,
            requested,
            None,
            spec.nharm,
            pletsch_clarke_likelihood,
            spec.use_weight,
            use_pi=False,
            general_outroot=os.path.join(workdir, "bench"),
        )

        times, observation_length, energies, expo = _load_events_for_all_files(
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
            observation_length=observation_length,
        )

        parameters_with_unc, parameters = _build_parameters_from_models(
            model, ref_model, observation_length, ignore_uncertainties=False
        )
        profile, profile_weight, weights = _build_profiles_and_weights(
            times, parameters, energies, len(files), get_outroot, spec.use_weight, nbin, 1e-8
        )
        (
            template_func,
            _,
            parameters,
            parameters_with_unc,
        ) = _prepare_templates_and_phase_priors(
            profile,
            profile_weight,
            spec.use_weight,
            spec.nharm,
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
            observation_length,
            model,
            template_funcs=template_func,
            weights=weights if spec.use_weight else None,
            tolerance=1e-8,
        )

        parameters = _centre_phases(observations, setup)
        setup = setup.with_baseline_from(parameters)
        setup = _dataclasses.replace(
            setup,
            factors=precondition_factors(
                _build_posterior_functions(observations, setup)[2],
                setup.factors,
                setup.n_parameters,
            ),
        )

        # Start every sampler from the same place: the MAP found by the bounded
        # optimizer, which is what ``--minimize-first`` gives a real run. Any
        # remaining optimizer variance would otherwise show up as sampler
        # variance.
        start, _, _ = point_estimate_fit(observations, setup)
        _, _, logpost = _build_posterior_functions(observations, setup)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if verbose:
        logging.info(
            "%s: %d parameters (%s), %d files",
            spec.name,
            setup.n_parameters,
            ", ".join(setup.parameter_names),
            len(files),
        )

    return Problem(
        spec=spec,
        logpost=logpost,
        start=np.asarray(start, dtype=float),
        parameter_names=list(setup.parameter_names),
        factors=[float(f) for f in setup.factors],
        baseline_values=[float(v) for v in setup.baseline_values],
    )


class _CountingPosterior:
    """Wrap a log-posterior so the harness can count its evaluations.

    ``ess_per_posterior_call`` is the sampler's statistical efficiency with
    parallelism divided out, which is the only way to compare a 32-walker
    ensemble against a 4-chain NUTS run on equal terms.
    """

    def __init__(self, func):
        self._func = func
        self.calls = 0

    def __call__(self, position):
        self.calls += 1
        return self._func(position)


#: One built posterior per worker process. Filled by :func:`_init_worker`.
_WORKER_PROBLEM = None


def _init_worker(spec):
    """Rebuild the posterior inside a worker process.

    The obvious thing -- send the parent's ``func_to_maximize`` to the pool --
    cannot be done: it is a closure over the observations and the fit setup, so
    it is unpicklable, and macOS spawns rather than forks. Forking instead is
    not the escape it looks like, because numba's thread pool has already been
    started in the parent by the time a sampler runs, and forking a process with
    live worker threads is unsupported.

    So each worker builds the problem itself from the spec. It is deterministic
    -- the dataset is cached and seeded, and the parent has already built it
    once, so the workers read rather than generate -- and the pooled and
    unpooled chains come out identical for the same seed, which is the check
    that this reproduces the parent's posterior rather than something adjacent
    to it.

    The build costs a few seconds per worker and they run concurrently; it lands
    in ``setup_seconds`` and therefore in the headline rate, which is the honest
    place for it.
    """
    global _WORKER_PROBLEM

    import numba

    # Each worker gets one numba thread. Ten workers each spawning ten threads
    # would oversubscribe the machine by 10x, and the point of the pool is to
    # use the cores that a single call's threading leaves idle at fixture scale.
    numba.set_num_threads(1)
    _WORKER_PROBLEM = build_problem(spec)


def _worker_logpost(position):
    """Evaluate the worker's own copy of the posterior."""
    return _WORKER_PROBLEM.logpost(position)


def _moves_stretch():
    """emcee's default: the affine-invariant stretch move alone."""
    return None


def _moves_de():
    """Differential evolution: propose along the difference of two walkers."""
    import emcee

    return [(emcee.moves.DEMove(), 1.0)]


def _moves_de_snooker():
    """The recipe emcee's own documentation recommends for correlated targets."""
    import emcee

    return [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]


# Named so a result file records *which* proposal produced it. The stretch move
# is affine invariant, which is why no amount of per-parameter rescaling changed
# its efficiency; a different move is a different proposal distribution, and is
# not neutralised the same way.
MOVES = {
    "stretch": _moves_stretch,
    "de": _moves_de,
    "de-snooker": _moves_de_snooker,
}


def run_emcee(problem, seed, steps, moves="stretch", jitter=1e-6, nwalkers=None, workers=0):
    """Run a bare ``emcee`` ensemble: no backend, no plots, no convergence loop.

    This is the object the sampler work iterates on. What a user actually pays
    for is measured by :func:`run_emcee_production`.
    """
    import emcee

    rng = np.random.default_rng(seed)
    ndim = problem.ndim
    if nwalkers is None:
        nwalkers = max(32, 2 * ndim)

    started = time.perf_counter()
    position = problem.start + rng.normal(0.0, jitter, size=(nwalkers, ndim))

    pool = None
    logpost = problem.logpost
    if workers:
        pool = multiprocessing.get_context("spawn").Pool(
            processes=workers, initializer=_init_worker, initargs=(problem.spec,)
        )
        # The parent's counting wrapper cannot see calls made in a worker, so
        # the function handed to the pool is the workers' own; the call count is
        # reconstructed below from what the moves provably do.
        logpost = _worker_logpost

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, logpost, moves=MOVES[moves](), pool=pool
    )
    # ``rng`` above seeds only the initial ball. The proposals come from
    # somewhere else entirely: ``EnsembleSampler.__init__`` copies the *global*
    # legacy numpy RNG into a private ``RandomState``, and nothing here sets
    # that. Without this line ``--seed`` does not control the sampling, every
    # process draws a different proposal stream, and no measurement can be
    # reproduced -- measured before it was added, one seed gave ESS 861, 846 and
    # 1096 on three invocations of the same command. It is invisible to an
    # in-process check, because the sampler mutates its copy and leaves the
    # global state untouched, so two same-seed runs in one process do agree.
    # A different bit generator from ``rng`` (MT19937 against PCG64), so seeding
    # both from ``seed`` does not couple the ball to the proposals.
    sampler.random_state = np.random.RandomState(seed).get_state()
    setup_seconds = time.perf_counter() - started

    started = time.perf_counter()
    try:
        sampler.run_mcmc(position, steps, progress=False)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    sample_seconds = time.perf_counter() - started

    return SamplerRun(
        # emcee stores (draw, chain, parameter); the diagnostics want chains first.
        chains=np.transpose(sampler.get_chain(), (1, 0, 2)),
        setup_seconds=setup_seconds,
        sample_seconds=sample_seconds,
        extra={
            "nwalkers": nwalkers,
            "moves": moves,
            "workers": workers,
            # One evaluation per walker per step, plus one pass over the
            # initial state. Confirmed against the counter, which reports
            # exactly nwalkers * (steps + 1) on every problem and both move
            # sets -- 128032 for 32 walkers and 4000 steps. Reconstructing it
            # rather than approximating matters because
            # ``ess_per_posterior_call`` is how a 32-walker ensemble will be
            # compared against a NUTS run.
            "posterior_calls": nwalkers * (steps + 1) if workers else None,
            "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        },
    )


def run_emcee_production(problem, seed, steps, **_):
    """Run the package's own :func:`ell1fit.mcmc_utils.safe_run_sampler`.

    Same sampler as :func:`run_emcee`, plus everything the pipeline wraps around
    it: an HDF5 backend written every step, an autocorrelation check every 100
    steps, and a full corner plot every 1000. Benchmarked separately so the cost
    of the wrapper can be told apart from the cost of the sampling.
    """
    import emcee

    from ell1fit.mcmc_utils import safe_run_sampler

    workdir = tempfile.mkdtemp(prefix="sampler_bench_prod_")
    try:
        outroot = os.path.join(workdir, "prod")
        # safe_run_sampler jitters its walkers with the global numpy RNG.
        np.random.seed(seed)

        started = time.perf_counter()
        # Its progress bar and logging would otherwise interleave with ours.
        with contextlib.redirect_stderr(io.StringIO()):
            safe_run_sampler(
                problem.logpost,
                problem.start,
                max_n=steps,
                outroot=outroot,
                labels=["d" + name for name in problem.parameter_names],
            )
        sample_seconds = time.perf_counter() - started

        backend = emcee.backends.HDFBackend(outroot + ".h5", read_only=True)
        chains = np.transpose(backend.get_chain(), (1, 0, 2))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return SamplerRun(
        chains=chains,
        setup_seconds=0.0,
        sample_seconds=sample_seconds,
        extra={
            "nwalkers": chains.shape[0],
            "iterations": chains.shape[1],
            # Not ``args.moves``: this adapter runs the package's own sampler,
            # which chooses its proposal itself. Recording the CLI value here
            # would label the result with a flag that had no effect on it.
            "moves": "package (ell1fit.mcmc_utils.default_moves)",
        },
    )


SAMPLERS = {
    "emcee": run_emcee,
    "emcee-production": run_emcee_production,
}


def summarize_chains(chains, discard_fraction=DISCARD_FRACTION):
    """Reduce a chain array to convergence diagnostics and posterior summaries.

    Parameters
    ----------
    chains : np.ndarray
        Shape ``(n_chains, n_draws, n_parameters)``, including warm-up.

    Returns
    -------
    dict
        Per-parameter diagnostics, the correlation matrix of the retained
        samples, and the worst R-hat across parameters.
    """
    import arviz

    n_chains, n_draws, ndim = chains.shape
    retained = chains[:, int(discard_fraction * n_draws) :, :]

    per_parameter = []
    for i in range(ndim):
        draws = np.ascontiguousarray(retained[:, :, i])
        entry = {
            "ess_bulk": float(arviz.ess(draws, method="bulk")),
            "ess_sd": float(arviz.ess(draws, method="sd")),
            "rhat": float(arviz.rhat(draws)),
            "mean": float(np.mean(draws)),
            "sd": float(np.std(draws, ddof=1)),
            "mcse_mean": float(arviz.mcse(draws, method="mean")),
            "mcse_sd": float(arviz.mcse(draws, method="sd")),
        }
        for quantile in QUANTILES:
            entry[f"q{quantile:g}"] = float(np.quantile(draws, quantile))
            entry[f"mcse_q{quantile:g}"] = float(
                arviz.mcse(draws, method="quantile", prob=quantile)
            )
        per_parameter.append(entry)

    flattened = retained.reshape(-1, ndim)
    correlation = np.corrcoef(flattened, rowvar=False) if ndim > 1 else np.ones((1, 1))

    return {
        "n_chains": int(n_chains),
        "n_draws": int(n_draws),
        "n_retained_draws": int(retained.shape[1]),
        "parameters": per_parameter,
        "correlation": np.asarray(correlation).tolist(),
        "rhat_max": max(entry["rhat"] for entry in per_parameter),
        "ess_min": min(entry["ess_bulk"] for entry in per_parameter),
    }


def run_one(problem, sampler_name, seed, steps, **sampler_kwargs):
    """Run one sampler once and reduce it to metrics."""
    counter = _CountingPosterior(problem.logpost)
    counted = dataclasses.replace(problem, logpost=counter)

    run = SAMPLERS[sampler_name](counted, seed=seed, steps=steps, **sampler_kwargs)
    summary = summarize_chains(run.chains)

    total_seconds = run.setup_seconds + run.sample_seconds
    n_draws = summary["n_draws"]
    calls = run.extra.get("posterior_calls") or counter.calls

    summary.update(
        {
            "seed": seed,
            "setup_seconds": run.setup_seconds,
            "sample_seconds": run.sample_seconds,
            "total_seconds": total_seconds,
            "posterior_calls": calls,
            "ess_per_second": summary["ess_min"] / total_seconds,
            "ess_per_step": summary["ess_min"] / n_draws,
            "steps_per_second": n_draws / total_seconds,
            "ess_per_posterior_call": summary["ess_min"] / max(calls, 1),
            "microseconds_per_posterior_call": 1e6 * total_seconds / max(calls, 1),
            "converged": summary["rhat_max"] < RHAT_GATE,
            "extra": run.extra,
        }
    )
    return summary


def do_run(args):
    """Run one configuration over several seeds and write a result file."""
    spec = PROBLEMS[args.problem]
    steps = args.steps if args.steps is not None else spec.default_steps

    print(f"Building {spec.name}: {spec.doc}")
    started = time.perf_counter()
    problem = build_problem(spec, verbose=True)
    build_seconds = time.perf_counter() - started
    print(
        f"  {problem.ndim} parameters: {', '.join(problem.parameter_names)}"
        f"  (built in {build_seconds:.1f} s)"
    )

    # ``emcee-production`` runs the pipeline's own wrapper and has no move
    # choice to make; it absorbs the keyword and ignores it.
    sampler_kwargs = {"moves": args.moves, "workers": args.workers}

    repetitions = []
    for index in range(args.seeds):
        seed = args.seed0 + index
        print(f"  {args.sampler} seed={seed} steps={steps} ...", end="", flush=True)
        result = run_one(problem, args.sampler, seed=seed, steps=steps, **sampler_kwargs)
        flag = "" if result["converged"] else "  UNCONVERGED"
        print(
            f" {result['total_seconds']:7.1f} s"
            f"  ESS_min={result['ess_min']:8.1f}"
            f"  ESS/s={result['ess_per_second']:8.2f}"
            f"  rhat={result['rhat_max']:.4f}{flag}"
        )
        repetitions.append(result)

    rates = [entry["ess_per_second"] for entry in repetitions]
    payload = {
        "problem": spec.name,
        "problem_doc": spec.doc,
        "sampler": args.sampler,
        # What the run actually used, which is not always what was asked for.
        "moves": repetitions[0]["extra"].get("moves", args.moves),
        "workers": args.workers,
        "steps": steps,
        "parameter_names": problem.parameter_names,
        "factors": problem.factors,
        "baseline_values": problem.baseline_values,
        "build_seconds": build_seconds,
        "python": sys.version.split()[0],
        "machine": machine_configuration(),
        "repetitions": repetitions,
        "ess_per_second_median": statistics.median(rates),
        "ess_per_second_min": min(rates),
        "ess_per_second_max": max(rates),
        "all_converged": all(entry["converged"] for entry in repetitions),
    }
    _report_run(payload)

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote {args.out}")
    return payload


def _spread(payload):
    """Half-range of the per-seed rates, as a fraction of the median."""
    median = payload["ess_per_second_median"]
    if median <= 0:
        return float("inf")
    return 0.5 * (payload["ess_per_second_max"] - payload["ess_per_second_min"]) / median


def _median_field(payload, field):
    """Median of one per-repetition metric."""
    return statistics.median(entry[field] for entry in payload["repetitions"])


def _report_run(payload):
    """Print the headline rate, its factors, and the seed-to-seed spread."""
    moves = payload.get("moves", "stretch")
    workers = payload.get("workers", 0)
    pool = f" / {workers} workers" if workers else ""
    print(
        f"\n{payload['problem']} / {payload['sampler']} / {moves}{pool}, "
        f"{payload['steps']} steps"
    )
    print(
        f"  ESS/s               {payload['ess_per_second_median']:10.2f}"
        f"   (seeds: {payload['ess_per_second_min']:.2f} - "
        f"{payload['ess_per_second_max']:.2f}, +-{100 * _spread(payload):.0f}%)"
    )
    print(f"  ESS/step            {_median_field(payload, 'ess_per_step'):10.4f}")
    print(f"  steps/s             {_median_field(payload, 'steps_per_second'):10.2f}")
    print(f"  ESS/posterior call  {_median_field(payload, 'ess_per_posterior_call'):10.6f}")
    print(
        f"  us/posterior call   {_median_field(payload, 'microseconds_per_posterior_call'):10.1f}"
    )
    print(f"  worst R-hat         {_median_field(payload, 'rhat_max'):10.4f}")
    if not payload["all_converged"]:
        print("  NOT CONVERGED: the rate above is not interpretable.")


def _pooled_parameter(payload, index):
    """Pool one parameter's summaries across repetitions.

    The repetitions are independent runs against the same target, so their
    quantile estimates average and their Monte Carlo errors combine as
    ``sqrt(sum of squares) / n``. Using every seed rather than just the first
    makes the agreement check correspondingly sharper -- which matters, because
    its job is to catch a sampler whose intervals are subtly too narrow.
    """
    entries = [entry["parameters"][index] for entry in payload["repetitions"]]
    n = len(entries)

    # Convert out of local coordinates. A change to the *scaling* -- which is
    # exactly what some of the sampler work is -- redefines the local unit, and
    # two runs summarised in their own local units would look wildly
    # inconsistent for no physical reason. physical = local * factor + baseline,
    # so widths and errors scale and only the quantiles take the offset.
    factor = payload["factors"][index]
    baseline = payload["baseline_values"][index]

    pooled = {}
    for key in ("sd", *[f"q{quantile:g}" for quantile in QUANTILES]):
        offset = baseline if key.startswith("q") else 0.0
        pooled[key] = float(np.mean([entry[key] for entry in entries]) * factor + offset)
        pooled[f"mcse_{key}"] = float(
            abs(factor) * np.sqrt(np.sum([entry[f"mcse_{key}"] ** 2 for entry in entries])) / n
        )
    return pooled


def _pooled_correlation(payload):
    """Mean correlation matrix across repetitions."""
    return np.mean([entry["correlation"] for entry in payload["repetitions"]], axis=0)


def _agreement_rows(before, after):
    """Compare posterior summaries parameter by parameter, in MCSE units."""
    rows = []
    for index, name in enumerate(before["parameter_names"]):
        left = _pooled_parameter(before, index)
        right = _pooled_parameter(after, index)
        for key in (*[f"q{quantile:g}" for quantile in QUANTILES], "sd"):
            combined = np.hypot(left[f"mcse_{key}"], right[f"mcse_{key}"])
            difference = right[key] - left[key]
            rows.append(
                {
                    "parameter": name,
                    "quantity": key,
                    "before": left[key],
                    "after": right[key],
                    "sigmas": abs(difference) / combined if combined > 0 else float("inf"),
                }
            )
    return rows


def do_compare(args):
    """Compare two result files: first whether they agree, then which is faster."""
    with open(args.before) as handle:
        before = json.load(handle)
    with open(args.after) as handle:
        after = json.load(handle)

    if before["problem"] != after["problem"]:
        raise SystemExit(
            f"Refusing to compare different problems: "
            f"{before['problem']} against {after['problem']}"
        )

    def label(payload):
        moves = payload.get("moves", "stretch")
        workers = payload.get("workers", 0)
        return f"{payload['sampler']}/{moves}" + (f"/{workers}w" if workers else "")

    print(f"{label(before)} -> {label(after)} on {before['problem']}\n")

    if before["parameter_names"] != after["parameter_names"]:
        raise SystemExit("Refusing to compare runs with different fitted parameters")

    if before.get("machine") != after.get("machine"):
        print(
            "WARNING: these runs were taken on different machine configurations; "
            "the speed comparison below is not meaningful.\n"
            f"  before: {before.get('machine')}\n  after:  {after.get('machine')}\n"
        )

    print("Credible intervals in physical units (difference in combined MCSE units)")
    rows = _agreement_rows(before, after)
    for row in sorted(rows, key=lambda entry: -entry["sigmas"])[: args.top]:
        flag = "  <== DISAGREES" if row["sigmas"] > AGREEMENT_SIGMA_GATE else ""
        print(
            f"  {row['parameter']:>10s} {row['quantity']:>4s}"
            f"  {row['before']:+12.5e} -> {row['after']:+12.5e}"
            f"  {row['sigmas']:6.2f} sigma{flag}"
        )
    worst = max(row["sigmas"] for row in rows)

    # The ridge: a sampler that fails to traverse it reports a weaker
    # correlation and narrower intervals, and looks fast doing so.
    names = before["parameter_names"]
    correlation_before = _pooled_correlation(before)
    correlation_after = _pooled_correlation(after)
    if len(names) > 1:
        triangle = np.triu_indices(len(names), k=1)
        deltas = np.abs(correlation_after - correlation_before)[triangle]
        worst_pair = int(np.argmax(deltas))
        i, j = triangle[0][worst_pair], triangle[1][worst_pair]
        print(
            f"\nLargest correlation change: {names[i]}-{names[j]}"
            f"  {correlation_before[i, j]:+.3f} -> {correlation_after[i, j]:+.3f}"
            f"  (delta {deltas[worst_pair]:.3f})"
        )

    print("\nSpeed")
    for payload in (before, after):
        status = "" if payload["all_converged"] else "  (UNCONVERGED)"
        print(
            f"  {label(payload):>22s}  {payload['ess_per_second_median']:9.2f} ESS/s"
            f"  +-{100 * _spread(payload):.0f}%{status}"
        )

    ratio = after["ess_per_second_median"] / before["ess_per_second_median"]
    combined_spread = _spread(before) + _spread(after)
    print(f"\n  speedup {ratio:.2f}x, seed-to-seed spread +-{100 * combined_spread:.0f}%")

    if not (before["all_converged"] and after["all_converged"]):
        print("  VERDICT: one of the runs did not converge; the rates mean nothing.")
    elif worst > AGREEMENT_SIGMA_GATE:
        print(
            f"  VERDICT: the posteriors disagree at {worst:.1f} sigma. "
            "A faster wrong answer is not a win."
        )
    elif abs(ratio - 1.0) <= combined_spread:
        print("  VERDICT: no detected difference -- the change is inside the seed spread.")
    else:
        direction = "faster" if ratio > 1 else "SLOWER"
        print(f"  VERDICT: {direction}, {ratio:.2f}x, beyond the seed spread.")


def do_list(_args):
    """Print the available problems and samplers."""
    print("Problems")
    for spec in PROBLEMS.values():
        print(
            f"  {spec.name}  {spec.doc}\n"
            f"        {len(spec.epoch_offsets)} epochs, {spec.n_events} events each, "
            f"-P {','.join(sorted(spec.fit_parameters))}, "
            f"{'weighted' if spec.use_weight else 'unweighted'}, "
            f"default {spec.default_steps} steps"
        )
    print("\nSamplers")
    for name in SAMPLERS:
        print(f"  {name}")
    print("\nMoves (emcee only)")
    for name, factory in MOVES.items():
        print(f"  {name:<12s}{factory.__doc__.splitlines()[0]}")


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="Show problems and samplers")
    listing.set_defaults(func=do_list)

    run = subparsers.add_parser("run", help="Benchmark one sampler on one problem")
    run.add_argument("--problem", choices=sorted(PROBLEMS), default="P1")
    run.add_argument("--sampler", choices=sorted(SAMPLERS), default="emcee")
    run.add_argument(
        "--moves",
        choices=sorted(MOVES),
        default="stretch",
        help="emcee proposal (ignored by emcee-production)",
    )
    run.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Evaluate walkers in a process pool of this size (0 = in-process)",
    )
    run.add_argument("--steps", type=int, default=None, help="Chain length per walker")
    run.add_argument("--seeds", type=int, default=3, help="Independent repetitions")
    run.add_argument("--seed0", type=int, default=1000, help="First sampler seed")
    run.add_argument("-o", "--out", default=None, help="Result file to write")
    run.set_defaults(func=do_run)

    compare = subparsers.add_parser("compare", help="Compare two result files")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--top", type=int, default=8, help="Worst rows to print")
    compare.set_defaults(func=do_compare)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    main()
