Precision and performance
=========================

Both are stated as measurements rather than claims. Every number below was
produced on the synthetic datasets from :mod:`ell1fit.tests.datagen`, and the
methods are described so they can be repeated.

Numerical precision
-------------------

The pipeline computes in float64 throughout. The question that matters is
whether that is enough, and the answer is obtained by running the same
computation at 80-bit extended precision and differencing.

.. list-table:: Phase error, float64 against an 80-bit reference
   :header-rows: 1
   :widths: 30 25 25

   * - Baseline
     - Phase error (cycles)
     - Cycles spanned
   * - 100 ks (one observation)
     - :math:`1.6\times10^{-10}`
     - :math:`7.5\times10^{5}`
   * - 116 d (a campaign)
     - :math:`2.0\times10^{-8}`
     - :math:`7.5\times10^{7}`
   * - 1 yr (long baseline)
     - :math:`4.3\times10^{-8}`
     - :math:`2.3\times10^{8}`

A fit achieves phase precision of order :math:`10^{-3}` cycles, so there are
roughly **five orders of magnitude of headroom**. The residual is dominated by
the spin-phase polynomial :math:`F_0 t`, not by the orbital inversion, which is
why arrival times are kept relative to each file's own ``PEPOCH``: a distant
reference epoch inflates :math:`t` and eats directly into that margin.

The deorbiting iteration solves its own defining equation to
:math:`1.5\times10^{-11}` s against a requested tolerance of :math:`10^{-8}` s.

.. note::

   On platforms without extended precision — Apple Silicon, for instance —
   PINT warns that it runs at reduced precision. That affects PINT's own MJD
   arithmetic, not the pipeline's phase computation, which is float64 by
   construction on every platform.

Speed
-----

The cost of a fit is dominated by evaluating the likelihood, once per MCMC step
per walker. One evaluation comprises: remove the orbital delay, convert to
phase, evaluate the template per event, and sum the logs.

.. list-table:: One full likelihood evaluation
   :header-rows: 1
   :widths: 20 20 20 15

   * - Events
     - Before
     - After
     - Speed-up
   * - 10,000
     - 1.07 ms
     - 0.37 ms
     - 2.9×
   * - 200,000
     - 16.6 ms
     - 3.0 ms
     - 5.5×
   * - 2,000,000
     - 156 ms
     - 26 ms
     - 5.9×

Where the time goes now, so future work can be aimed properly:

.. list-table:: Cost breakdown after optimization
   :header-rows: 1
   :widths: 40 30

   * - Stage
     - Share
   * - Orbital deorbiting
     - ~50%
   * - Log-sum over events
     - ~38%
   * - Template evaluation
     - ~11%

The deorbiting step is now the bottleneck, and it is limited by ``sin``/``cos``
throughput rather than by the number of iterations.

What made it faster
^^^^^^^^^^^^^^^^^^^

**Evaluating the template on its uniform grid.** The sample grid a template is
interpolated on is uniformly spaced, so locating the right polynomial piece is
arithmetic rather than a binary search. Since ``interp1d(kind="cubic")`` is
``make_interp_spline(k=3)`` underneath, and each uniform interval lies inside a
single polynomial piece, the exact per-interval cubic is recoverable from the
spline's derivatives. This gave 38–40× on what was ~80% of an evaluation. The
two agree to :math:`10^{-14}` relative — the difference between evaluating the
same polynomial in Taylor form and in the B-spline basis.

**Clamping inside the interpolation loop**, so the logarithm and the sum are
left to numpy. This removes two full-length passes and their temporaries, and is
also *more accurate*: ``np.sum`` accumulates pairwise, with error growing like
:math:`\log N` rather than the :math:`N` of a running total.

**Threading above 50,000 events.** Bitwise identical to the serial path, since
each iteration writes one independent element and there is no reduction to
reorder. Below that threshold thread-launch overhead dominates and the parallel
kernel is markedly slower, hence the cutoff.

What was tried and rejected
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recorded so they are not attempted again:

- **Fusing the log-sum into the compiled kernel with compensated summation:
  0.63×, i.e. slower.** A scalar loop calling ``log`` once per event loses badly
  to numpy's vectorised ``log``.
- **Newton iteration for the deorbiting: 1.1× only.** The fixed-point iteration
  already converges in about four passes, because the contraction factor
  :math:`A_1\omega \approx 6\times10^{-4}` gains three decimal digits per pass.
  The step is transcendental-bound, not iteration-bound.

Optimizer reliability
---------------------

Local coordinates are only useful if every direction has a comparable scale.
Before preconditioning they did not, differing by a factor of a thousand:

.. list-table:: Local step corresponding to one standard deviation
   :header-rows: 1
   :widths: 25 20 30

   * - Parameter
     - Factor
     - Local 1σ
   * - ``A1``
     - 1000
     - :math:`1.7\times10^{-6}`
   * - ``F0_0``, ``F0_1``
     - 0.01
     - :math:`6.6\times10^{-6}`
   * - ``Phase_0``, ``Phase_1``
     - 1 (default)
     - :math:`1.8\times10^{-3}`

Measuring the curvature and rescaling accordingly (see
:func:`ell1fit.scaling.precondition_factors`) changed the optimizer from
reaching the global optimum in **7 of 12** starts to **12 of 12**, and collapsed
a 3.2-nat spread in the achieved log-posterior to zero. Over 15 independent
realizations the log-posterior improved in 12 (mean +0.87 nats, maximum +5.7),
and the RMS error on ``A1`` fell from :math:`7.2\times10^{-3}` to
:math:`6.5\times10^{-3}`.

Sampler efficiency
------------------

Optimizer reliability decides whether a fit lands on the right solution; how
long the error bars take is a separate question, and it is measured separately.
The figure of merit is **effective samples per second**, taking the *minimum*
over parameters rather than the mean, so that a sampler stuck on one direction
cannot average its way to a win. ``tools/sampler_bench.py`` produces it, over
three fixed problems: ``P1`` (2×5,000 events, 5 parameters), ``P2`` (2×200,000,
5), and ``P3`` (3×50,000, 10, with eccentricity free — a strongly correlated
ridge). R-hat is reported as a gate, not as a score.

Two changes have been measured, each against a controlled baseline on the same
commit — reusing an older number would credit a change with someone else's win:

.. list-table:: Effective samples per second, ``P1``
   :header-rows: 1
   :widths: 34 16 16 16

   * - Change
     - Before
     - After
     - Factor

   * - Differential-evolution moves
     - 8.39
     - 31.98
     - 3.8×
   * - Whole chains in separate processes
     - 37.3
     - 69.5
     - 1.9×

Together with the threading-threshold fix these amount to roughly an order of
magnitude on ``P1`` and on ``P3``, and about 4× on ``P2``, relative to where
this work started.

**Differential-evolution moves**, which propose along the vector between two
walkers instead of stretching toward one, are the pipeline's default, supplied
by :func:`ell1fit.mcmc_utils.default_moves`; passing
``moves=emcee.moves.StretchMove()`` restores emcee's own. On ``P1`` this is 3.8×, and all of it is in
effective samples *per step* (0.158 → 0.618) with the cost per step unchanged —
the proposals are better, not cheaper. This matters because the ensemble is
affine invariant: rescaling the parameters cannot help it, only a better
proposal shape can.

That 3.8× is measured against the 0.8/0.2 ``DEMove``/``DESnookerMove`` mix that
emcee recommends for correlated targets, and it understates the current
default, because **the snooker move has since been dropped**. It was doing
nothing: it builds its proposal direction as ``delta / sqrt(|delta|)``, which is
not a unit vector, so its step length goes as the *square* of the coordinate
scale — and at the ``1e-6`` convention then in force it displaced a walker
3 × 10\ :sup:`-11` against an ensemble spread of 9 × 10\ :sup:`-6`, accepting
almost every proposal precisely because it never went anywhere.

Fixing the scale was not enough. On ``P1`` at 16000 steps over three seeds, all
converged (worst :math:`\hat{R}` ≤ 1.0034):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Move set
     - one σ = ``1e-6``
     - one σ = 1
   * - ``DEMove`` + ``DESnookerMove`` (0.8/0.2)
     - 0.717
     - 0.861
   * - ``DEMove`` alone
     - 0.919
     - 0.911

The two ``DEMove`` figures agreeing across a million-fold change of units is
the affine-invariance property acting as a control: only the snooker row is
supposed to move, and only it does. Rescaling recovers most of the dead move,
but a fifth of the proposals spent on it is still worse than not proposing at
all, so :func:`~ell1fit.mcmc_utils.default_moves` now returns ``DEMove`` alone.
The general recommendation is sound; it does not hold on these posteriors.

A matched stretch-move baseline at 16000 steps is not quoted here because it
did not converge (:math:`\hat{R}` = 1.009), which is itself the point — the
comparison above is between two configurations that both did.

**Running whole chains in separate processes**, rather than threading inside a
single likelihood call, is worth a further 1.9× on ``P1``. Each pooled run is
bitwise identical to its unpooled twin, so this is pure hardware and the
statistics are unchanged by construction. Match the worker count to
``nwalkers / 2`` — emcee updates the ensemble in two halves, so 8 workers beat
10 (six slots would idle in the second half) and 16 are far worse. Worker
start-up is several seconds and lands inside the measured rate, so short runs
pay for it. This is measured in the benchmark harness only; a pipeline run
builds its posterior from user files, folding and template derivation, so
workers would have to repeat the front half of the pipeline.

Gradient-based sampling
^^^^^^^^^^^^^^^^^^^^^^^

An ensemble sampler asks only for the value of the log-posterior. NUTS also
asks which way is uphill, and nothing in the compiled path yields a gradient, so
:mod:`ell1fit.nuts_sampling` rebuilds the log-posterior from JAX primitives.
Both ``ell1fit --sampler nuts`` and ``tools/sampler_bench.py --sampler nuts``
use the same rebuild; the CLI needs ``pip install ell1fit[nuts]`` for it,
since the base install does not pull in JAX or numpyro.

.. list-table:: Ensemble against NUTS, four chains
   :header-rows: 1
   :widths: 14 18 18 18 16

   * - Problem
     - Sampler
     - ESS/s
     - ESS/step
     - Worst R-hat

   * - ``P1``
     - emcee + DE, 8 processes
     - 69.5
     - 0.689
     - 1.012
   * - ``P1``
     - NUTS
     - **202.2**
     - 2.191
     - 1.0007
   * - ``P3``
     - emcee + DE, 8 processes
     - 7.68
     - 0.354
     - 1.0066
   * - ``P3``
     - NUTS
     - 2.47
     - 0.811
     - 1.0026

**The two results should not be averaged.** On ``P1`` NUTS is 6.3× over what
the pipeline delivers today, and it is the only configuration that reaches
R-hat 1.001 there at all. On ``P3`` it is 0.32×, i.e. three times slower: the
proposal is 2.6× better per posterior call, but on the ridge a gradient costs
9.9 ms against a value call's 3.1 ms, and the arithmetic goes the wrong way.
Gradient sampling is therefore a capability that pays where the data are small
relative to the parameter count, not a general speed-up. Neither problem
produced a post-warm-up divergence.

One configuration detail is not optional: JAX exposes a single CPU device by
default, so four chains queue on roughly one core. Giving XLA one device per
chain is worth 3.7–4.3×, with the chains bitwise unchanged; the figures above
assume it.

Since a second implementation of the model can silently disagree with the
first, three checks hold them together, and they divide the work deliberately:
the JAX log-posterior is compared against the compiled one over a ball of one
standard deviation (worst :math:`5.6\times10^{-7}` on ``P1``); the gradient is
checked by differencing *the JAX function*, not the compiled one, whose own
rounding would swamp the comparison; and the Hessians of the two are compared at
the maximum. The residual :math:`10^{-6}` is last-bit rounding in the
``fastmath`` deorbiting kernels, identified as such because it does not move
when their tolerance is tightened by six orders of magnitude.

That third check also settles a question the first two cannot. When two
samplers disagree about the *width* of a posterior — here 4.6% on ``TASC``,
which the harness flagged at 4σ — running either one longer converges on the
answer only slowly. The Hessian is deterministic: the two implementations agree
on the curvature to :math:`2\times10^{-6}`, which rules out a model difference,
and measured against those widths the ensemble is narrow on 10 parameters out of
10 while NUTS scatters evenly about them. The ensemble is still slightly
under-dispersed after 16,000 steps at R-hat 1.0066.

Sampler ideas tried and rejected
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Seeding NUTS with the preconditioning factors as its diagonal mass matrix:
  no effect, on either problem** (``P1`` 41.3 against 43.7 for the identity;
  ``P3`` 2.45 against 2.43). The expectation was that this would pay precisely
  because NUTS, unlike the ensemble, is *not* affine invariant. Warm-up
  rediscovers the diagonal from the identity within its first thousand steps
  regardless. ``--mass {curvature,identity}`` is kept so the claim stays
  testable.
- **Batching the likelihood over all walkers into one compiled call.** Costed
  before being written, against a process pool: the ceiling was 2.4× on ``P1``
  and 1.13× on ``P2``, below the pool everywhere, and it leaves the per-call
  Python untouched.
- **Vectorising the NUTS chains** (``--chain-method vectorized``) is *slower*
  than running them one after another, 37.8 against 41.3 on ``P1``, because
  chains mapped together all pay for the deepest trajectory.


Model comparison
----------------

Effective samples per second is the wrong figure of merit for one class of
question. Asking whether a binary is eccentric *at all* is a comparison between
two models, and that needs the evidence :math:`\log Z` — the likelihood
integrated over the prior — which neither the ensemble sampler nor NUTS
produces. Nested sampling does, and ``--sampler nested`` (dynesty) is in the
harness for that reason rather than for speed.

Two preconditions had to be established first, neither of them automatic.

**The prior has to be proper.** An unbounded flat prior has no evidence at all,
and :mod:`ell1fit.priors` returns ``0`` inside its bounds rather than
:math:`-\log w`, so the package's priors are a mixture of normalised and
unnormalised factors — harmless for MCMC, fatal for an integral. Every prior
``P3`` uses turns out to be proper. The omitted normalisation comes to exactly
:math:`2\log 2`: the two eccentricity uniforms and nothing else, which is a
useful confirmation that nothing else was left open.

**The unit-cube transform, not the log-prior, is what** :math:`\log Z`
**integrates against.** :mod:`ell1fit.prior_transform` builds it by inverting
each prior's own CDF, so :math:`\log Z` is already the evidence under a
normalised prior and no correction is ever applied afterwards. It is checked
three ways: a
constant likelihood must integrate to :math:`\log Z = 0`, measured
:math:`-0.0037 \pm 0.0075`; each transform must be proportional to
``exp(logprior)`` as the package itself computes it; and a likelihood made
Gaussian in exactly one *physical* parameter must reproduce independent
one-dimensional quadrature, which it does for all ten parameters within 1.93σ.
Only that third check can catch a wrong local-coordinate scale factor, to which
the first is blind.

Whether the Bayes factor is calibrated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Bayes factor that prefers eccentricity on eccentric data demonstrates
nothing by itself; the test is whether it also prefers a circle on circular
data. Four problems make that control explicit — data generated with and
without eccentricity, each fitted both with ``EPS1``/``EPS2`` free (10
parameters) and with them fixed at zero (8). All four are 3×5,000 events, a
scale at which the injected eccentricity is an 11.9σ effect and a call is an
order of magnitude cheaper than on ``P3``.

.. list-table:: Evidence at 4,000 live points, mean over seeds
   :header-rows: 1
   :widths: 12 20 20 26 22

   * - Problem
     - Data
     - Model
     - :math:`\log Z` sampled
     - Laplace

   * - ``E1``
     - eccentric
     - eccentric (10)
     - +71.97 ± 1.37
     - +73.24
   * - ``E0``
     - eccentric
     - circular (8)
     - +55.37 ± 0.02
     - +55.38
   * - ``C1``
     - circular
     - eccentric (10)
     - +83.68 ± 0.94
     - +85.26
   * - ``C0``
     - circular
     - circular (8)
     - +100.91 ± 0.00
     - +100.90

That is :math:`\ln B = +16.6 \pm 1.4` on eccentric data and
:math:`-17.2 \pm 0.9` on circular data — decisive in both directions on
Jeffreys' scale, and close to symmetric. The control holds. The quoted spread
is the standard error over five seeds for the 10-parameter problems and two for
the 8-parameter ones, which is the honest error bar for reasons given below.

The Laplace approximation, computed from the JAX Hessian at the optimum,
agrees to 0.01 nats on both 8-parameter problems and sits 1.3 and 1.6 nats
above the 10-parameter ones. That is roughly one standard error on ``E1`` and
somewhat beyond it on ``C1``; the eccentric model's posterior is a curved
ridge, which is exactly where a Gaussian approximation should be expected to
overstate the volume. It is quoted as an independent adjudicator, not as a
correction.

Nested sampling can fail silently
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At 200 live points ``E1`` returned ``log Z = -32.0 +- 0.30``: a confident
number, quoted to two decimals, and wrong by a hundred nats. The live points
had never found the peak — the best likelihood any of them saw was 112 nats
below one the optimizer reaches in seconds — and dynesty gave no indication of
it. Its own error estimate is the sampling error of the integral it *did*
compute, and says nothing about the mode it never entered.

The harness therefore compares the sampler's best likelihood against the
optimizer's and reports the shortfall, refusing to call a run converged when it
exceeds one nat. The peak occupies about :math:`e^{-24}` of the prior volume,
so finding it is a matter of the live points stumbling into it:

- 1,000 live points missed it in two attempts out of three.
- ``sample="rslice"`` did not help; it spent three times the calls and still
  missed. (``sample="auto"`` is not a separate option to try — dynesty selects
  ``rwalk`` at this dimension, reproducing it bit for bit.)
- 4,000 live points found it in all five seeds, with a shortfall of 0.03 nats.

The guard is necessary but not sufficient. The one 1,000-point run that *did*
reach the peak still came out about 5 nats low, because finding a mode late is
not the same as sampling it. Only the seed spread catches that.

**dynesty's error bars understate the scatter on the 10-parameter problems by
more than an order of magnitude.** Five seeds of ``E1`` gave 67.5 to 76.1, a
3.1-nat standard deviation, against a quoted ±0.10; ``C1`` behaves the same
way. On the 8-parameter problems the quoted error is honest — the seeds agree
to 0.02 where dynesty claims 0.06. The ``bayes`` subcommand therefore builds
its uncertainty from the seed scatter and ignores what dynesty reports, and
several seeds per model are not optional.

The cost is what one would expect of an integral rather than a sample: at 4,000
live points on eight workers, roughly 900 s per seed for the 10-parameter
problems and 200 s for the 8-parameter ones.

Reproducing these numbers
-------------------------

The precision comparison needs an interpreter with genuine 80-bit
``longdouble`` — an x86 build. The speed and optimizer measurements run
anywhere. All of them use :mod:`ell1fit.tests.datagen` to generate data from a
known solution, and ``tools/refactor_net.py`` records a bitwise snapshot of the
pipeline's outputs, which is how "this change altered nothing" is verified.

The sampler measurements come from ``tools/sampler_bench.py``, which needs
``pip install -e .[bench]``. ``list`` names the problems, ``run --problem P1
--sampler emcee -o before.json`` measures one, and ``compare before.json
after.json`` reports the ratio together with the seed spread, declining to call
a difference that falls inside it. Note that ``refactor_net.py`` cannot referee
this work: its MCMC entries carry a few tenths of a standard deviation of chain
noise at the step counts it uses, so only its deterministic entries mean
anything here.

The evidence runs use the same harness: ``run --problem E1 --sampler nested
--nlive 4000 --workers 8 --seeds 5 -o E1.json``, then ``bayes E1.json
E0.json`` for the Bayes factor. ``bayes`` quotes the seed scatter, warns when
any run failed the shortfall check, and reads ``Jeffreys`` off the result.

Both ``--sampler nuts`` and ``--sampler nested`` are also available on the
``ell1fit`` CLI itself, needing ``pip install ell1fit[nuts]`` or
``ell1fit[nested]`` respectively. A Bayes factor from the CLI needs no
orchestration beyond running it twice, with and without the eccentricity
parameters in ``-P``, and subtracting the two runs' ``log_evidence`` fields --
the harness's ``bayes`` subcommand exists to add the seed-scatter uncertainty
on top, not because the subtraction itself needs machinery. The CLI's
``--sampler nested`` also takes ``--workers``, spreading likelihood
evaluations across processes the same way the harness does, except a worker
here is handed the already-built model rather than rebuilding it from a
synthetic spec -- no event-file I/O or template refinement repeated per
worker. Measured on a small two-parameter fixture, ``--workers`` was a net
*slowdown* (23 s against 17 s single-process at ``nlive=500``): the
per-likelihood-call cost has to clear the pool's own IPC overhead before
spreading it across processes pays for itself, the same trade-off the
emcee pool above makes explicit. Reach for it on a fit expensive enough
that a single likelihood call is milliseconds, not microseconds -- a large
event count or several free parameters -- not by default.
