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
- **No orbital derivatives beyond** ``PBDOT``. ``XDOT``, ``OMDOT`` and friends
  are not fitted.
- **One binary, shared across all files.** Spin parameters are per file; orbital
  parameters are not.

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

Measured over 40 realizations with a deliberately mis-set ``A1``, refinement
reduced the bias by a factor of 5.4 — from a 3.4σ systematic to something
consistent with zero. But only about half of *individual* realizations landed
closer to the truth than they started: this is a shift of the distribution, not
a per-fit guarantee.

.. note::

   That measurement predates fixes to both the optimizer conditioning and the
   refinement convergence threshold, either of which could change the size of
   the effect. Treat the factor of 5.4 as indicative rather than current.

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
