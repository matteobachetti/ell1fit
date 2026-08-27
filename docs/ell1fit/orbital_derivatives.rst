Orbital-size drift: fitting and bounding ``A1DOT``
==================================================

``A1DOT`` (:math:`\dot{x}`, light-seconds per second) is the rate at which the
projected semi-major axis of the orbit changes. It is the orbital-size
counterpart of ``PBDOT``, and in a mass-transferring binary the two are not
independent: for a circular orbit whose total mass and angular momentum evolve
under conservative mass transfer,

.. math::

   \frac{\dot{x}}{x} \approx \frac{2}{3}\,\frac{\dot{P_b}}{P_b},

so a measured ``PBDOT`` predicts an ``A1DOT``. Confirming that prediction — or
bounding it away — tests whether the observed orbital decay really is the
orbit shrinking, rather than something in the companion (a quadrupole cycle,
say) that moves ``PB`` without moving the orbital size in step.

This page covers how ``ell1fit`` fits ``A1DOT``, how to turn a fit into an
upper limit, what will silently spoil that limit, and how to forecast in
advance whether a given dataset can reach an interesting number at all.

How ``A1DOT`` enters the model
------------------------------

The pipeline holds one binary shared across all files, referenced to a single
epoch, and gives each file a fixed offset carrying that solution to its own
``PEPOCH`` (see :doc:`pipeline`). ``PBDOT``, ``EPS1DOT`` and ``EPS2DOT`` are
handled entirely that way: PINT propagates them once at load, and the result is
a constant added before phases are computed.

``A1DOT`` cannot be handled that way, because a constant cannot be fitted.
Folding the drift into ``A1_offset_i`` freezes it at whatever the input parfile
said, and the likelihood is then flat in ``A1DOT`` — the fit would return its
prior and call it a measurement. So ``A1DOT`` is deliberately excluded from the
offsets (:data:`ell1fit.models.OFFSET_DERIVATIVES`), and the per-file amplitude
is instead rebuilt from it on every likelihood evaluation:

.. math::

   x_i = x + x_{\mathrm{offset},i} + \dot{x}\,\Delta t_i .

:math:`\Delta t_i` — ``binary_dt_i`` in the parameter dictionary — is the
**lever arm**: the time from the shared model's binary epoch to file *i*'s,
in seconds. Two properties of it are deliberate.

**PINT defines it, not** ``ell1fit``. It is the ``dt_integer_orbits`` that
``change_binary_epoch`` uses internally,

.. math::

   \Delta t = n P_b + \tfrac{1}{2} n^2 P_b \dot{P_b},

with :math:`n` the whole number of orbits between the two epochs. Using PINT's
own definition is what makes a fitted ``A1DOT`` written to a parfile read back
as the same model: if ``ell1fit`` computed its own lever arm, a round trip
through PINT would quietly change the answer.

**It is computed with** ``A1DOT`` **switched off.**
:func:`ell1fit.models._orbital_epoch_offsets` deep-copies the reference model
and zeroes ``A1DOT`` before asking PINT for the offsets, so the drift is
applied exactly once — by the phase model, where it is fitted — and not a
second time inside ``A1_offset_i``. Both the Numba and the JAX implementations
of the posterior carry the same expression, and a test compares them
(``test_jax_posterior_follows_a1dot_like_the_numba_one``).

Fitting it
----------

``A1DOT`` is requested like any other parameter::

    ell1fit epoch1.nc epoch2.nc ... -p epoch1.par epoch2.par ... \
        -P F0,F1,TASC,A1,A1DOT --use-weight --ignore-uncertainties \
        --minimize-first --template-iterations 3

Three things happen behind that command that are specific to a derivative.

**A single epoch is refused.** With one file every ``binary_dt_i`` is zero, the
likelihood is exactly flat in ``A1DOT``, and the MCMC would return the prior
with a straight face. :func:`ell1fit.pipeline._reject_unmeasurable_derivatives`
raises instead. This is the same reasoning behind rejecting ``-P PBDOT``, which
the phase model genuinely cannot see.

**Unknown parameter names are refused too.** ``-P`` used to drop silently
anything the model did not contain, so before this work ``-P A1DOT`` ran a
perfectly ordinary fit that simply did not fit ``A1DOT``, and reported success.
:func:`ell1fit.pipeline._collect_parameter_names` now raises and lists what is
available.

**The step scale comes from the lever arm.** ``get_factors`` needs a length
scale per parameter. For ``A1DOT`` it is :math:`\sigma_{A1} / T_{\rm span}`,
with :math:`T_{\rm span}` the spread of the files' ``PEPOCH`` — the only scale
in :func:`ell1fit.scaling.estimate_uncertainties_from_model` that depends on
how the observations are *spaced* rather than how long they are.

.. note::

   Fixing that revealed a bug affecting every parameter, not just this one. The
   candidate uncertainties were compared with ``np.argmin``, and ``NaN`` wins
   ``argmin``: a parfile that quoted no uncertainty beat a perfectly good model
   estimate, and the parameter silently fell back to the default scale. The
   candidates are now filtered to the finite positive ones before choosing.

The prior
---------

With no uncertainty in the parfile, ``A1DOT`` gets a flat prior symmetric about
the parfile value, of half-width

.. math::

   \left|\dot{x}\right|_{\max} = \frac{2\pi x}{P_b},

the projected orbital velocity in units of :math:`c`. It is a physical bound —
the orbit cannot change size faster than the star moves along it — and it is
about :math:`10^7` times wider than any credible drift. That width is the
point: **an upper limit is only worth quoting if the prior bound is not what
sets it.** Keeping it finite also keeps the prior proper, so nested sampling
can integrate against it, at the cost of an Occam factor that a Bayes factor on
``A1DOT`` would feel.

Setting an upper limit
----------------------

Fit ``A1DOT`` as a global free parameter in the multi-file fit, with the flat
symmetric prior above, and quote **the 95th percentile of** :math:`|\dot{x}|`
**over the posterior samples**. That is it. Three alternatives are worse:

- *Do not* use a Bayes factor. A Bayes factor answers "is a drift preferred?",
  which is a different question, and its answer depends on the prior width —
  which here is a physical bound chosen to be uninformative, precisely the
  situation in which the Occam factor is arbitrary.
- *Do not* build the limit from per-epoch ``A1`` measurements. Within one epoch
  ``A1`` is weakly constrained and strongly correlated with ``TASC`` and
  ``F0``; the coherent fit uses the phase connection between epochs, which the
  epoch-by-epoch route throws away.
- *Do not* quote the interval from a fit that did not refine its templates.
  See the next section: that limit is too tight, in the one direction an upper
  limit must never err.

What will spoil the limit
-------------------------

**Template smearing biases the drift toward zero.** Each file's pulse template
is built by folding *that file's own events* with the current solution. An
uncorrected ``A1DOT`` smears the far epoch, so that epoch's template comes out
broadened by exactly the error being fitted — and a broadened template fits the
smeared events better than the sharp truth does. The likelihood is pulled
toward :math:`\dot{x} = 0`.

Measured on the checked-in two-epoch fixture (700-day baseline, injected
:math:`\dot{x} = 6\times10^{-10}`), over nine seeds: a single template pass
returns a mean :math:`4.80\times10^{-10}`, mean pull :math:`-0.50 \pm 0.34`;
three passes return :math:`5.80\times10^{-10}`, mean pull :math:`-0.09 \pm
0.40`, with error bars 20% smaller. On the fixture's own seed the single-pass
answer is :math:`1.7\times10^{-10}`, three sigma low. So the effect is a
systematic ~20% underestimate that occasionally reaches three sigma, and
``--template-iterations 3`` removes it.

A drift measurement biased toward zero is **an upper limit that is too tight**.
Always refine.

**The cycle-count ambiguity sets a ceiling on what a local optimizer can
find.** The drift shows up as a sinusoid in orbital phase of amplitude
:math:`\dot{x}\,T_{\rm span}\,F_0` cycles at the far epoch. Push that past half
a cycle and the far epoch is off by whole rotations: the likelihood becomes
multimodal and a minimizer starting from :math:`\dot{x}=0` cannot walk to the
truth. On the fixture, :math:`6\times10^{-10}` is 0.27 cycles and is found;
:math:`10^{-8}` is 4.5 cycles and is not, even though folding the events *at*
the injected value recovers the pulse at full strength. This is the pipeline's
usual cycle-count ambiguity, not something specific to ``A1DOT`` — but it means
a null result must be checked against this scale before it is called a limit.

**PINT rescales large values on their way into a parfile.** PINT reads
``PBDOT 7.2`` as :math:`7.2\times10^{-12}`, and implements that convention in
the *assignment*: a bare float above ``scale_threshold`` (1e-7) is multiplied
by ``scale_factor`` (1e-12) on the way in. ``A1DOT`` carries the same
convention. Since the flat prior above reaches :math:`\sim 6\times10^{-4}`, a
fit that strayed that far would have written a parfile disagreeing with its own
result table by twelve orders of magnitude, silently.
:func:`ell1fit.create_parfile.update_model` now assigns a units-carrying
Quantity, which takes the branch that does not rescale, and warns when the
value exceeds the threshold — above it the *format* is ambiguous and the
parfile will not round-trip no matter what is written.

**Use one self-consistent processing batch.** ``PBDOT`` enters the lever arm,
and independently produced batches of parfiles for the same source have been
seen to disagree about it at the 0.2% level. Mixing them makes
:math:`\Delta t_i` inconsistent between files.

Forecasting the sensitivity
---------------------------

Before spending a fit, it is worth knowing what precision the data can reach.
The obvious tool is the Fisher matrix — take the Hessian of the log-posterior
at the best fit, invert it, read the marginal variance off the diagonal. **On
this problem that does not work**, and the failure is worth recording because
it is not obvious.

The *conditional* width (from :math:`H_{ii}` alone) is perfectly stable: on the
fixture it comes out :math:`5.79\times10^{-11}` for any finite-difference step
between :math:`10^{-7}` and :math:`3\times10^{-6}` local units. The *marginal*
width, which is what a limit needs, is not: over the same range of steps it
moves from :math:`1.03\times10^{-9}` to :math:`1.28\times10^{-10}`, a factor of
eight. ``A1`` and ``A1DOT`` are correlated at 0.79 and the degeneracy between
them is *curved*, so the Schur complement that produces the marginal is a near
cancellation whose value depends on how far out the curvature was sampled. A
quadratic form is simply the wrong description of that direction.

**The profile likelihood is the right tool.** Fix ``A1DOT`` on a grid,
re-optimize every other parameter at each point, and read the half-width where
the profile has dropped by 0.5 (or 1.92 for a 95% one-sided bound). It makes no
quadratic assumption and inverts no matrix. On the fixture, against the MCMC
the pipeline actually runs:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30

   * - quantity
     - MCMC (1000 steps)
     - profile likelihood
   * - :math:`\dot{x}`
     - :math:`+6.22\times10^{-10}`
     - :math:`+6.12\times10^{-10}`
   * - :math:`\sigma_{\dot{x}}`
     - :math:`1.26\times10^{-10}`
     - :math:`1.24\times10^{-10}`

— agreement to 1.3% on the width, against an injected
:math:`6.00\times10^{-10}`. The profile costs a few dozen local optimizations
instead of a full chain.

.. warning::

   Match the priors when comparing the two. The fixture's parfiles quote
   ``A1`` to :math:`10^{-2}` lt-s, and that prior carries a large part of the
   constraint; profiling with ``ignore_uncertainties=True`` against an MCMC run
   without it gives a profile that is flat to within 0.14 in log-posterior
   across eight sigma — not a bug, just a different problem. With flat priors
   on ``A1`` the two-epoch fixture constrains ``A1DOT`` only through the
   photons, and two epochs give ``(A1, A1DOT)`` almost exactly as many degrees
   of freedom as there are epoch amplitudes to fit.

Recipe
~~~~~~

1. Build the fit setup exactly as the pipeline does — load, fold, weight,
   build templates, trace ``Phase_i``, precondition, **refine** — and stop
   before ``optimize_solution``.
2. Pick a grid in ``A1DOT`` spanning a few times the scale
   ``get_factors`` chose for it.
3. At each grid point, fix ``A1DOT`` and maximize the log-posterior over
   everything else, warm-starting from the previous point.
4. Read the half-width at :math:`\Delta \log p = 0.5`. For the 95% one-sided
   bound, read the crossing at :math:`\Delta \log p = 1.92`.

The M82 X-2 dataset
-------------------

Fifteen NuSTAR epochs, :math:`2.67\times10^{6}` events, spanning MJD 56683 to
60659 (10.9 yr), fitting ``F0_i``, ``F1_i`` and ``Phase_i`` per epoch plus a
global ``A1``, ``TASC`` and ``A1DOT`` — 48 free parameters, flat priors
throughout, three template passes. The profile is smooth and very nearly
parabolic over :math:`\pm 8\times10^{-10}` lt-s/s:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - quantity
     - value
   * - profile peak
     - :math:`-1.0\times10^{-10}` lt-s/s (:math:`-3.2\times10^{-3}` lt-s/yr)
   * - :math:`\sigma_{\dot{x}}`
     - :math:`2.0\times10^{-10}` lt-s/s (:math:`6.3\times10^{-3}` lt-s/yr)
   * - 95% interval
     - :math:`[-5.2, +2.6]\times10^{-10}` lt-s/s
   * - Kepler expectation
     - :math:`-3.9\times10^{-12}` lt-s/s (:math:`-1.2\times10^{-4}` lt-s/yr)

The peak sits half a sigma from zero, so this is a null result, and the number
to quote is the bound: :math:`|\dot{x}| < 5.2\times10^{-10}` lt-s/s at 95%.

**The Kepler expectation is a factor of 52 below that.** The current dataset
cannot test it, and it is worth being precise about why, because the obvious
remedy is the wrong one.

*It is not mainly the baseline.* Eight of the fifteen epochs sit inside a
single month in 2014; the rest are scattered over the following twelve years.
Weighting all epochs equally, the root-mean-square spread of the lever arms —
which is what :math:`\sigma_{\dot{x}}` actually scales with — is 3.7 yr, not
10.9. Extending the monitoring helps, but slowly and with a hard ceiling: one
more epoch today improves :math:`\sigma` by 1.2 times, one in 2031 by 1.4, one
in 2036 by 1.7.

*It is the photons.* With the lever arm doing its best, closing a factor of 52
needs roughly :math:`(52/1.7)^2 \approx 900` times the effective pulsed counts
:math:`N\,f_p^2`. M82 X-2's pulsed fraction over these epochs runs from 1% to
30% and the source is at 3.5 Mpc; no realistic amount of further NuSTAR time
gets there.

So the honest use of ``A1DOT`` on this source is as a bound, not a
confirmation. What the bound *does* exclude is a pathological orbital-size
drift: it puts the orbit-shrinking timescale :math:`x/\dot{x}` above 1.3 kyr,
against the 180 kyr the measured ``PBDOT`` implies.

.. note::

   The Fisher matrix, run on the same setup, reports
   :math:`\sigma_{\dot{x}} = 1.1\times10^{-10}` — 1.8 times *tighter* than the
   profile. On the two-epoch fixture it errs the other way, by a factor of two.
   Neither direction is reproducible, which is the point: on this posterior the
   inverse-Hessian marginal is not a usable forecast. Profile.
