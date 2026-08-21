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

Reproducing these numbers
-------------------------

The precision comparison needs an interpreter with genuine 80-bit
``longdouble`` — an x86 build. The speed and optimizer measurements run
anywhere. All of them use :mod:`ell1fit.tests.datagen` to generate data from a
known solution, and ``tools/refactor_net.py`` records a bitwise snapshot of the
pipeline's outputs, which is how "this change altered nothing" is verified.
