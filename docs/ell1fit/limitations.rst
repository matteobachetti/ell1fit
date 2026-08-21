Known limitations
=================

Things worth knowing before trusting a result. Where a concern was tested, the
measurement is given — including the cases where the concern turned out to be
unfounded.

Model scope
-----------

- **ELL1 only.** The model must define ``TASC``, not ``T0``; a model with ``T0``
  is rejected at load. ELL1 is appropriate for nearly circular orbits, which is
  what it exists for, but it is not a general Keplerian solver.
- **No orbital derivative is fitted.** ``PBDOT`` is honoured as an *input*:
  PINT applies it when each model's binary epoch is aligned to its ``PEPOCH``,
  so a parfile value does reach the computation. But the phase model holds
  ``PB`` constant, so the likelihood is flat in ``PBDOT`` and ``-P PBDOT`` is
  rejected with an error rather than quietly returning its own prior as a
  measurement. ``XDOT``, ``OMDOT`` and friends are not used at all.
- **Small eccentricities only, and the limit is measured.** The Roemer delay is
  expanded to second order in :math:`e`, so what remains scales as
  :math:`e^3`. The residual phase error is

  .. math::

     \sigma_\Phi \approx 0.236\, e^3\, x\, F_0 \quad \text{cycles},

  independent of :math:`\omega`. ``ell1fit`` compares that against the
  precision the folded profiles imply and **warns** when it reaches a third of
  it — for a typical 22 lt-s, 7.5 Hz system at 1e-3 cycles, that is
  :math:`e \approx 0.03`; for a redback-like 10 lt-s, 200 Hz system,
  :math:`e \approx 0.013`. Nothing is rejected: the residual lives in the
  third harmonic of the orbit, orthogonal to every direction ELL1 can move in,
  so exceeding the limit costs sensitivity rather than biasing the recovered
  eccentricity — at :math:`e = 0.01` the bias on :math:`e` is 2.7e-5
  *relative*. Above it, a full Keplerian model (BT, DD) is the right tool.

  The check runs on the **input** parfile's eccentricity. A fit that starts
  circular and wanders somewhere eccentric will not trigger it.
- **One binary, shared across all files.** Spin parameters are per file; orbital
  parameters are not. They are, however, *propagated* to each file's epoch when
  the parfile sets an orbital derivative, so a shared solution stays valid
  across a long baseline. Ignoring that propagation costs up to
  ``PBDOT * baseline**2`` in phase — measured at 3e-2 cycles for
  ``PBDOT = 1e-10`` over ten years, against the ~1e-3 cycles a fit resolves.

The Rayleigh statistic
----------------------

``--likelihood Rayleigh`` uses :math:`Z_1^2` instead of the profile likelihood.
Two real restrictions follow, and the pipeline now warns about both rather than
discarding the options silently:

- **Only the fundamental harmonic is used.** The pulse template is not consulted
  at all, so ``-N``/``nharm`` has no effect on the fit. For a sharply peaked
  pulse this discards genuine information, making it strictly less sensitive
  than ``--likelihood PC``.
- **Per-event weights are ignored**, so ``--use-weight`` does nothing.

.. note::

   A third concern was tested and did **not** hold up. Because :math:`Z_1^2` is
   not on a log-density scale, one expects the log-prior added to it to be
   weighted wrongly, and the textbook weak-signal relation
   :math:`\log L \approx Z_1^2/2` suggests credible intervals a factor
   :math:`\sqrt{2}` too narrow. Measured over 30 realizations, the ratio of true
   scatter to quoted uncertainty was **1.07 for Rayleigh and 0.89 for
   Pletsch–Clarke** — both consistent with 1. The :math:`Z_1^2/2` result applies
   to a likelihood profiled over the pulse amplitude, whereas this pipeline
   holds the template fixed. Rayleigh is less sensitive, but it is not
   miscalibrated.

Fitting behaviour
-----------------

- **The point estimate is a starting point, not an answer.** ``--minimize-first``
  runs a local optimization, and the likelihood surface has genuine multiple
  optima — parameters trade off against one another, and pulse phase is
  periodic. The MCMC exploration is the result; the minimisation only seeds it.
  Preconditioning made this markedly more reliable (7 of 12 starts reaching the
  global optimum, versus 12 of 12), but "markedly more reliable" is not
  "guaranteed". Over 15 realizations, 3 still ended below where the unscaled
  version happened to land.
- **Convergence is assessed, not assured.** ``safe_run_sampler`` checks the
  integrated autocorrelation time and warns when acceptance is poor, but a chain
  can stop at ``--nsteps`` without having converged. The warnings are worth
  reading.

Iterative template refinement
------------------------------

``--template-iterations`` defaults to 1, which disables refinement entirely and
reproduces the behaviour of not having the feature.

Refining a template against the data it was fitted to carries a risk in the
opposite direction from the bias it corrects: the template can begin absorbing
noise, which then pulls the solution. This is mild at low ``nharm`` and real at
high ``nharm``, so the harmonic count is held fixed while iterating.

.. warning::

   **On the current code, refinement does not appear to help, and may slightly
   hurt.** This is worth stating plainly because the feature was built to remove
   a bias that measurement now attributes to something else.

   An early measurement over 40 realizations, with a deliberately mis-set
   ``A1``, found a 3.4σ bias in the one-pass fit which three passes reduced by a
   factor of 5.4. Both the optimizer's parameter conditioning and the refinement
   loop's convergence threshold were subsequently found to be wrong, and fixed.
   Repeating the identical measurement afterwards inverted the result:

   .. list-table::
      :header-rows: 1
      :widths: 30 25 25 20

      * -
        - bias, 1 pass
        - bias, 3 passes
        - RMS ratio
      * - before those fixes
        - +4.1e-03 (t = 3.4)
        - +7.6e-04 (t = 0.8)
        - 0.74 (better)
      * - **on the current code**
        - −1.8e-04 (t = −0.2)
        - −2.3e-03 (t = −2.8)
        - **1.17 (worse)**

   The one-pass bias has essentially vanished: it was largely an artifact of an
   optimizer that was not reliably finding the global optimum, rather than of a
   smeared template. With the fit already at the optimum, further passes mostly
   let the template absorb noise — the risk described above.

   This is a single configuration (one offset size, one parameter, ``nharm=2``)
   and the effect is about 2σ, so it does not establish that refinement is
   useless. It may still help when the starting ephemeris is badly wrong, which
   an 0.02 lt-s offset may no longer represent. It does mean the default of 1 is
   the right default on present evidence, and that anyone enabling refinement
   should verify it helps on their own data rather than assuming.

Numerical
---------

- **float64 throughout.** :doc:`performance` quantifies the headroom: about five
  orders of magnitude between the phase error and the precision a fit achieves.
  That margin depends on arrival times being referenced to each file's own
  ``PEPOCH``.
- **The deorbiting iteration has a hard cap.** Reaching it means the parameters
  are outside the invertible region; such positions are screened out before the
  iteration runs.

Platform
--------

- **Extended precision is not required**, and the code does not use it. On
  platforms lacking it, PINT warns that *it* runs at reduced precision; that
  affects PINT's MJD handling, not the phase computation here.
- **The threading threshold is fixed** at 50,000 events. It was measured on one
  machine, and the optimum will differ with core count.

Testing
-------

- **The shipped example data is single-epoch in disguise.** ``events0.par`` and
  ``events1.par`` are byte-identical, so the CLI tests that pass two files
  exercise no multi-epoch behaviour. Multi-epoch coverage comes from the
  synthetic generator in :mod:`ell1fit.tests.datagen`.
- **Figures are not verified.** Nothing in the test suite or in
  ``tools/refactor_net.py`` inspects a plot, so a diagnostic that silently stops
  being informative will not be caught automatically. One such regression has
  already happened.
