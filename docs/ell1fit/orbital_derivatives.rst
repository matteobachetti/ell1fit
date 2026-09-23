Orbital derivatives: period drift and orbital-size drift
========================================================

Two different derivatives say the orbit is evolving, and ``ell1fit`` reaches
them by two different routes. ``PBDOT`` and ``PBDDOT`` — the orbital period's
first and second time derivatives — are measured *across* epochs by
``ell1decay``, from how the fitted ascending-node time ``TASC`` of each epoch
drifts away from a constant-period ephemeris. ``A1DOT`` — the drift of the
projected orbital size — is instead a free parameter *inside* a single
coherent multi-file ``ell1fit`` run.

Both are usually non-detections, and both are then quoted the same way: as the
95th percentile of the parameter's magnitude over the posterior samples. This
page covers the period derivatives first, since they are the ones most often
asked for, and ``A1DOT`` second.

Period drift: ``PBDOT`` and ``PBDDOT``
--------------------------------------

If the orbital period drifts as
:math:`P(t) = P_b + \dot{P_b}\,t + \tfrac{1}{2}\ddot{P_b}\,t^2 + \ldots`, then
the ascending node arrives progressively earlier or later than a fixed-period
ephemeris predicts, by

.. math::

   \Delta T_{\rm asc}(t) = \frac{\dot{P_b}\,t^2}{2 P_b}
                         + \frac{\ddot{P_b}\,t^3}{6 P_b} + \ldots

``ell1decay`` takes one ``ell1fit`` result file per epoch, forms
:math:`\Delta T_{\rm asc}` for each, and fits that curve. The per-epoch
``TASC`` uncertainties are generally asymmetric, so the likelihood is a
split-normal one, picking the negative or positive error bar per point
according to which side of the model the point falls on.

Three models, two Bayes factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three nested models are always fit, never one in isolation:

.. list-table::
   :header-rows: 1
   :widths: 12 40 48

   * - Model
     - Terms in :math:`\Delta T_{\rm asc}(t)`
     - What it represents
   * - ``MLIN``
     - constant + linear
     - No period derivative at all. The constant absorbs a reference ``TASC``
       that was not exactly the data's own mean; the linear term absorbs a
       plain ``PB`` *miscalibration*, which is not a ``PB`` derivative.
   * - ``M0``
     - ``MLIN`` plus a quadratic term
     - Adds ``PBDOT``.
   * - ``M1``
     - ``M0`` plus a cubic term
     - Adds ``PBDDOT``.

Each model's evidence :math:`\log Z` comes from nested sampling, repeated over
several seeds because the scatter across seeds is a fairer error on
:math:`\log Z` than any single run's own quoted one. Comparing neighbours in
that ladder gives one Bayes factor per derivative:

- ``bayes_factor_pbdot`` — ``M0`` over ``MLIN``: does the data need ``PBDOT``?
- ``bayes_factor_pbddot`` — ``M1`` over ``M0``: does it need ``PBDDOT``?

Each comparison differs by exactly one parameter, which is what makes it a
question about that one derivative rather than about the shape of the curve in
general.

Measurement or upper limit
~~~~~~~~~~~~~~~~~~~~~~~~~~

A derivative is reported as a **measurement** when its own Bayes factor
reaches :math:`\ln \mathrm{BF} \ge 1`, and as an **upper limit** otherwise.
That threshold is not a new invention: :math:`2\ln\mathrm{BF} = 2` is exactly
where the Kass & Raftery (1995) grading stops calling the evidence
inconclusive. Change it with ``--detection-ln-bf`` if a paper wants a
different bar.

The limit itself is **the 95th percentile of the parameter's magnitude** over
the posterior samples, the same convention used for ``A1DOT`` below and for
the eccentricity. It says what an upper limit should say — that fraction of
the posterior mass lies below the quoted magnitude — and it is deliberately
*not* the more extreme end of the two-sided interval, which for a posterior
sitting off-centre is a larger and different statement. Use
``--upper-limit-level`` for a level other than 95%.

A **three-sigma limit is always reported as well**, at 99.73% — the mass a
Gaussian carries within three standard deviations — since that is how a
non-detection is often quoted. It is unaffected by ``--upper-limit-level``,
which moves only the headline number.

It comes with a caveat the 95% limit does not have. A 99.73% quantile is
determined by the outermost 0.27% of the chain, and the chain is built by
``resample_equal``, which draws *with replacement* — so the number of
genuinely independent points out in that tail is smaller than
``PBDOT_nsamples`` suggests. Bootstrapping the chain of a nine-epoch fit
measures what that costs:

.. list-table::
   :header-rows: 1
   :widths: 14 20 22 22 22

   * - ``--nlive``
     - Samples
     - Beyond the 3 sigma limit
     - Scatter, 95% limit
     - Scatter, 3 sigma limit
   * - 200
     - 885
     - ~2
     - 2.5%
     - 7.8%
   * - 500 (default)
     - 2151
     - ~6
     - 2.1%
     - 1.6%
   * - 2000
     - 8526
     - ~23
     - 1.2%
     - 2.6%

At the default ``--nlive 500`` the three-sigma limit is good to a few percent,
which is fine for a bound quoted to two significant figures. At ``--nlive
200`` it rests on about two samples and is not worth quoting. Raising
``--nlive`` past the default buys more samples but not obviously more
accuracy — the central value itself moved by about 3% between these three
runs, comparable to the scatter within any one of them, so **do not quote a
three-sigma limit to more than two significant figures**. The 95% limit is
stable throughout and needs no such care.

Because a magnitude limit throws away the sign, **two-sided credible intervals
are always reported alongside it**, whether or not the parameter was detected,
at the credible levels a Gaussian one, two and three sigma carry (68.27%,
95.45% and 99.73%) rather than the rounded 16/84 and 2.5/97.5. A reader who
needs one number for a table reads the limit; a reader who needs to know
whether the drift leans positive or negative reads the interval.

One number in the output is a diagnostic and nothing more:
``PBDOT_significance_sigma`` is the distance of zero from the posterior in
units of its own standard deviation — a Gaussian approximation, quoted only so
the numbers are comparable with what ``ell1ecc`` prints. Nothing switches on
it. If it disagrees sharply with the Bayes factor, that is a sign the
posterior is not the near-Gaussian shape the approximation assumes, and the
corner plot is worth a look.

What the limit rests on, and what it does not
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two halves of the answer have very different sensitivities to the prior
box, and it is worth being clear about which is which. Measured on a synthetic
nine-epoch dataset with no injected derivative, varying the prior half-width
on the quadratic and cubic coefficients over a factor of 100 (the default is
200 times the data's own residual spread):

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 19 19

   * - Prior half-width
     - :math:`\ln\mathrm{BF}` ``PBDOT``
     - :math:`\ln\mathrm{BF}` ``PBDDOT``
     - ``PBDOT`` limit
     - ``PBDDOT`` limit
   * - 20 × spread
     - -1.80
     - -0.63
     - 1.90e-10
     - 2.38e-10
   * - 200 × spread (default)
     - -4.06
     - -2.88
     - 1.84e-10
     - 2.23e-10
   * - 2000 × spread
     - -5.80
     - -6.15
     - 1.78e-10
     - 2.48e-10

**The limits are prior-insensitive.** They move by under about 10% over that
whole range, and not even monotonically — that is nested-sampling scatter, not
a prior dependence. The box is far wider than the data can constrain, so the
posterior is likelihood-dominated and its percentiles are a statement about
the data.

**The detection gate is not the same kind of statement.** Both Bayes factors
shift by roughly 2.3 per decade of prior width — which is :math:`\ln 10`,
exactly the Occam factor, since the box width is set from the data's own
residual spread and not from a physical bound. Widening the prior by a factor
of three moves :math:`\ln\mathrm{BF}` by more than the entire detection
threshold.

The ``A1DOT`` half of this page argues against using a Bayes factor to *set a
limit* for precisely this reason, and that argument still stands. It is not
contradicted here, because the Bayes factor decides only **whether to quote a
value or a limit**, never how large the limit is. In the measurement above the
verdict happens to be the same at every width — everything is far below the
threshold — but a genuinely marginal derivative sitting near
:math:`\ln\mathrm{BF} \approx 1` could be flipped by a factor-of-three
change in the box. If a result turns on that verdict, vary
``--detection-ln-bf`` and check whether the conclusion survives.

Can the limits be cross-checked against the 1-sigma error?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tempting, and usually wrong. For a Gaussian posterior *centred on zero* the
99.73% magnitude limit really is three times the one-sigma half-width (and the
95% one is 1.960 times it, not two — 95% is not 95.45%). Neither condition
generally holds here, for two independent reasons.

**The posterior is rarely centred on zero.** A non-detection means the
posterior is *consistent* with zero, not centred on it; in practice it sits
one to two sigma away. Once it does, :math:`|x|` is folded-normal and its
high quantiles run well above the interval half-width. Across eight
symmetric-error test posteriors the direct three-sigma limit came out 19% to
42% larger than three times the one-sigma half-width. The one case that
happened to land near zero (:math:`\mu/s = -0.20`) agreed to 0.8%.

**With asymmetric TASC errors the posterior is not Gaussian at all.** This is
the case that matters, since real ``ell1fit`` output has asymmetric error
bars — it is why the likelihood is split-normal in the first place. That
likelihood picks each point's sigma from the *sign* of its residual, so the
log-likelihood is piecewise quadratic, with a kink wherever a residual crosses
zero. With equal error bars the curvature is constant and the posterior is
exactly Gaussian; with unequal ones the curvature jumps in magnitude and sign.
Measured on five asymmetric-error datasets, the ratio
:math:`h_{3\sigma}/h_{1\sigma}` ranged from **2.32 to 4.01** against the
Gaussian 3.000, with excess kurtosis from −0.94 to +1.27 — and it is not
sampler noise: on one dataset the ratio held at 1.66/2.32 through
``--nlive`` 500, 2000 and 6000 (3k to 40k samples).

So the deviation is real, large, and specific to each dataset — there is no
correction factor to apply. This is precisely why the reported limits are
**empirical quantiles of the chain**, which assume nothing about the shape,
rather than anything derived from a one-sigma error. If a cross-check is
wanted, the honest one is to compare the quoted limit against the
folded-normal quantile implied by the chain's own mean and standard deviation:
on the symmetric-error runs, where the posterior really is Gaussian, that
agreed to a few percent every time, and a sharp disagreement is a useful
signal that the posterior is skewed or kinked.

Reading the output
~~~~~~~~~~~~~~~~~~

``{outroot}_results.json`` holds a block per model. Inside ``M0`` (for
``PBDOT``) and ``M1`` (for ``PBDDOT``) each derivative carries:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Key
     - Meaning
   * - ``PBDOT_50``
     - Posterior median.
   * - ``PBDOT_1sigma_lo`` / ``_hi``
     - Two-sided 68.27% credible interval.
   * - ``PBDOT_2sigma_lo`` / ``_hi``
     - Two-sided 95.45% credible interval.
   * - ``PBDOT_3sigma_lo`` / ``_hi``
     - Two-sided 99.73% credible interval.
   * - ``PBDOT_upper_limit``
     - Headline magnitude limit, or ``NaN`` when the parameter is detected.
   * - ``PBDOT_upper_limit_level``
     - Credible level of that limit (0.95 unless overridden).
   * - ``PBDOT_upper_limit_3sigma``
     - Magnitude limit at 99.73%, also ``NaN`` when detected. Estimated from
       the chain's tail — see the caveat above.
   * - ``PBDOT_upper_limit_3sigma_level``
     - 0.9973, always.
   * - ``PBDOT_detected``
     - Whether the Bayes factor cleared the threshold.
   * - ``PBDOT_ln_bf`` / ``_ln_bf_err``
     - The Bayes factor the decision was made on.
   * - ``PBDOT_significance_sigma``
     - Gaussian-approximation exclusion of zero. Diagnostic only.
   * - ``PBDOT_nsamples``
     - Posterior samples the summary was built from. Relevant to how well the
       three-sigma limit is resolved.
   * - ``PBDOT_summary``
     - The one-line form, ready to paste into a draft.

``PBDDOT`` is reported in :math:`\mathrm{yr}^{-1}`; ``PBDOT``, being a period
over a time, is dimensionless. The summary lines are also written to the log,
and read like this:

.. code-block:: text

   PBDOT = 2.996e-08 (+7.58e-11 -7.55e-11, 1 sigma); zero excluded at 384.3 sigma
   |PBDDOT| < 2.54e-10 1/yr (95% upper limit), < 3.29e-10 1/yr (99.73%, 3
   sigma); 2 sigma interval -1.53e-10 to 2.82e-10 1/yr; zero excluded only at
   0.62 sigma, so this is a limit and not a measurement

A typical invocation:

.. code-block:: console

   $ ell1decay epoch*.ecsv -o decay --upper-limit-level 0.95

Note that ``{outroot}.par`` records ``M0``'s **median** ``PBDOT`` even when
that parameter is only an upper limit. That is deliberate: an ephemeris needs
a number, and "this is a limit rather than a measurement" is a statement for
the paper, not for the parameter file. ``M1`` is never adopted into the
ephemeris regardless of its Bayes factor.

Orbital-size drift: bounding ``A1DOT``
--------------------------------------

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

This section covers how ``ell1fit`` fits ``A1DOT``, how to turn a fit into an
upper limit, what will silently spoil that limit, and how to forecast in
advance whether a given dataset can reach an interesting number at all.

How ``A1DOT`` enters the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~

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
~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
^^^^^^

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
~~~~~~~~~~~~~~~~~~~

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

Forecasting a future campaign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A profile likelihood costs a few dozen local optimizations — half an hour for
the dataset above — which is cheap for one answer and too expensive for the
question that follows it: *would more observations, or a bigger telescope,
change the verdict?* For screening candidate epoch lists there is a linear
model that costs nothing, provided it is calibrated once against a real profile
run.

.. warning::

   Do **not** read :math:`\sigma_{A1}` from a fit with ``A1DOT`` held at zero
   as the drift sensitivity. On the M82 X-2 set that fit reports
   :math:`\sigma_{A1} = 5.8` ms, but the drift the same data can resolve over
   its own 12.35-yr span is :math:`\sigma_{\dot{x}} \times T_{\rm span} = 68`
   ms — twelve times larger. The gap is the price of the ``A1``/``A1DOT``
   degeneracy plus the fact that the useful lever arm is 3.2 yr, not 12.4. One
   number is the precision on the *average* ``A1``; the other is the precision
   on the *difference* between the ends, and only the second one is the
   measurement.

The cheap model
^^^^^^^^^^^^^^^

Treat each epoch as an independent measurement of ``A1`` with weight
:math:`w_i`, and :math:`(A1, \dot{x})` as a straight-line fit through them:

.. math::

   \sigma_{\dot{x}} = \kappa \left[\sum_i w_i (t_i - \bar{t}_w)^2\right]^{-1/2},
   \qquad \bar{t}_w = \frac{\sum_i w_i t_i}{\sum_i w_i}.

Two things make it usable:

**The weights come out of the results table.** Phase precision goes as the
square root of the pulsed signal-to-noise, so :math:`w_i \propto Z^2_{1,i} - 2`
— the per-file ``Z21_i`` column the pipeline already writes, minus the two
degrees of freedom of its noise floor. Only the shape matters; normalise the
set so that :math:`(\sum_i w_i)^{-1/2}` equals the :math:`\sigma_{A1}` the
pipeline actually reported for that dataset.

**One constant absorbs everything the model ignores** — the curved
``A1``/``A1DOT`` degeneracy, the per-epoch ``F0``/``F1``/``Phase`` covariance,
the template smearing. Fit :math:`\kappa` by running the model on the same
epochs a profile likelihood has already been run on. On the M82 X-2 dataset
:math:`\kappa = 3.0`; the uncalibrated model is three times too optimistic,
which is the same disease as the Fisher matrix and the same reason not to
trust either raw. :math:`\kappa` is **not** universal — re-derive it against
one profile run before forecasting on a different source, and treat the output
as good to a factor of ~1.5.

Applied to the 14-epoch set that includes the 2026 epoch, the model forecasts
:math:`\sigma_{\dot{x}} = 1.7\times10^{-10}` lt-s/s, a shortfall of 45 against
the Kepler expectation — the 2026 epoch having bought a little over the 52 the
15-epoch profile run measured.

What the scalings say
^^^^^^^^^^^^^^^^^^^^^

.. math::

   \sigma_{\dot{x}} \propto \frac{1}{\sqrt{N_{\rm ep} A}\; T_{\rm rms}},

with :math:`A` the effective area and :math:`T_{\rm rms}` the weighted spread
of the epochs. Collecting area enters under a square root and the baseline
enters linearly, which sets the terms of trade: **a mission with ten times
NuSTAR's area buys a factor of 3.2, and nothing more.** Monitoring at a fixed
cadence makes :math:`N_{\rm ep} \propto T`, so time is worth
:math:`T^{3/2}` — closing the remaining factor of 14 on M82 X-2 needs six times
the present baseline, about seventy years. Two epochs a year from 2032 with a
10x instrument crosses one sigma around 2100.

Angular resolution is the underrated axis. :math:`Z^2_1` is the squared pulsed
counts over the *total* counts in the aperture, so resolving the target out of
its neighbours cuts the denominator directly. If M82 X-2 is a third of what
NuSTAR's aperture collects, separating it is worth another :math:`\sqrt{3}` —
comparable to a factor of three in area, for free.

The anchor floor
^^^^^^^^^^^^^^^^

No future instrument improves the *early* end of an existing lever arm. Once
the earliest block of data is fixed, its own ``A1`` precision bounds everything
downstream:

.. math::

   \sigma_{\dot{x}} \;\geq\; \kappa\, \frac{\sigma_{A1}^{\rm (anchor)}}{T_{\rm span}}.

The 2014 NuSTAR block — seven epochs inside one month — reaches
:math:`\sigma_{A1} = 6.5` ms together, and that number is now permanent:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - span from 2014
     - floor on :math:`\sigma_{\dot{x}}`
     - shortfall vs Kepler
   * - 12.4 yr (today)
     - :math:`5.0\times10^{-11}`
     - 13x
   * - 30 yr
     - :math:`2.1\times10^{-11}`
     - 5.3x
   * - 50 yr
     - :math:`1.2\times10^{-11}`
     - 3.2x
   * - 160 yr
     - :math:`3.9\times10^{-12}`
     - 1.0x

A perfect telescope launched tomorrow reaches one sigma on this source in the
twenty-second century. That is the honest ceiling, and it is why the section
above says to quote the bound rather than chase the detection.

Short observations barely constrain ``A1`` at all
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The obvious way to buy lever arm cheaply is an archival epoch from long before
the campaign started. It is worth checking, but the arithmetic is unforgiving,
and for a reason that is easy to miss: within one epoch the fit is free in
``Phase_i``, ``F0_i`` and ``F1_i``, which between them absorb the constant,
linear and quadratic parts of the orbital delay over that window. What is left
to constrain ``A1`` is the cubic and higher terms of
:math:`x \sin(2\pi t / P_b)`, so for :math:`\Delta T \ll P_b` the usable signal
falls as :math:`(\Delta T / P_b)^3` — verified numerically as a log-log slope
of 2.99.

Taking the rms of :math:`\sin(2\pi t/P_b)` orthogonalised against
:math:`\{1, t, t^2\}` over a window of length :math:`\Delta T`, averaged over
the starting orbital phase, and normalising to a NuSTAR pointing spanning a
full 2.53-d orbit:

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - window
     - :math:`\Delta T / P_b`
     - leverage on ``A1``
     - relative
   * - 2.5 d (full orbit)
     - 0.99
     - :math:`3.2\times10^{-1}`
     - 1
   * - 1.5 d
     - 0.59
     - :math:`9.1\times10^{-2}`
     - 1/4
   * - 100 ks
     - 0.46
     - :math:`4.4\times10^{-2}`
     - 1/7
   * - 50 ks
     - 0.23
     - :math:`5.8\times10^{-3}`
     - 1/55
   * - 30 ks
     - 0.14
     - :math:`1.3\times10^{-3}`
     - 1/252

A single 30-ks snapshot is 250 times worse at measuring ``A1`` than an
orbit-spanning pointing with the same photons. **An archival epoch is worth
having only if it spans a decent fraction of an orbit** — how long it is
matters far more than how many counts it collected, and no amount of effective
area compensates.

The gain, when the epoch does qualify, is real but bounded. Adding one epoch in
2001 to the M82 X-2 set — 13.4 yr before the 2014 block — gives:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - its share of the total ``A1`` weight
     - required single-epoch :math:`\sigma_{A1}`
     - gain
     - shortfall
   * - 0.03
     - 33 ms
     - 1.3x
     - 36x
   * - 0.1
     - 18 ms
     - 1.7x
     - 27x
   * - 1.0
     - 6 ms
     - 3.4x
     - 13x
   * - :math:`\infty`
     - 0
     - 4.6x
     - 10x

Even an infinitely good 2001 epoch stops at 4.6x, because the *other* end of
the lever arm then becomes the anchor. A 4.6x tighter published bound is a
result worth having; a detection is not on the table.

One last thing such an epoch must clear: the pulse has to be findable. M82
X-2's spin is not extrapolatable backwards — ``F0`` runs from 0.72876 Hz in
2014 to 0.71166 Hz in 2026, a mean :math:`\dot{F_0}` of
:math:`-4.3\times10^{-11}` Hz/s, but wandering by :math:`\sim 3\times10^{-3}`
Hz about any smooth trend, and spinning *up* between 2020 and 2021. Reaching
back thirteen years means a blind search over several times that scatter, with
an orbital acceleration search inside it, and the trials penalty comes straight
off the detection threshold.
