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
``tools/jax_posterior.py`` rebuilds the log-posterior from JAX primitives for
``tools/sampler_bench.py --sampler nuts``. **This is benchmark-side only; the
pipeline samples with emcee.**

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
