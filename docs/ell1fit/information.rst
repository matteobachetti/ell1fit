Information budget: ``ell1info``
================================

A multi-epoch fit returns one number per shared parameter and says nothing
about where that number came from. ``ell1info`` answers the questions the fit
cannot answer for itself:

* Which observations are actually measuring the orbit, and which are along for
  the ride?
* What would the uncertainties be if I dropped one, or added one?
* How much is freeing the eccentricity costing me on the other parameters?
* What eccentricity limit can this dataset support at all?
* How much of the constraint is coming from the priors rather than the photons?

It takes the files, parfiles and flags an ``ell1fit`` run would take, and
instead of fitting anything it reports the curvature of the likelihood those
inputs define. Nothing is sampled, so a question that would cost days of MCMC
costs a minute -- which is what makes it usable *before* committing to a run,
or to a proposal.

Quickstart
----------

.. code-block:: console

   $ ell1info epoch*_ev.nc -p epoch*_ev.par \
         -P "F0,F1,A1,EPS1,EPS2" --use-weight --report A1

The arguments mirror ``ell1fit``: the same event files, the same parfiles in
the same order, and the same ``-P``, ``--use-weight``, ``--use-pi``,
``-e/--energy-range`` and ``-N/--nharm`` you would pass to the fit. That is
deliberate -- the point is to describe *the fit you are about to run*, so any
flag that changes the likelihood has to be passed here too.

.. warning::

   List the files in the same order as the fit. ``ell1fit`` takes its global
   reference model from the **first** parfile, so the order is part of the
   problem definition, not a presentation detail.

Why the answer is not "the longest observations"
------------------------------------------------

The projected semi-major axis enters the arrival times as :math:`A_1 \sin\Phi`,
with :math:`\Phi` the orbital phase. Over a stretch of data much shorter than
one orbit, :math:`\sin\Phi` is very nearly a quadratic in time -- and a
quadratic in time is exactly what a segment's own ``Phase_i``, ``F0_i`` and
``F1_i`` are free to absorb. A short observation can therefore be rich in
photons and still say almost nothing about ``A1``, because its local spin fit
eats the signature before it reaches the residuals.

What survives is the cubic-and-higher structure, so *orbital coverage* matters
far more than exposure. ``ell1info`` reports that surviving fraction per
segment, under the heading ``surviving``: one means the segment's own spin
parameters take nothing, and a number near zero means the local fit can nearly
reproduce the orbital signature by itself.

The consequence is worth stating plainly, because it inverts the obvious
intuition: **the ranking by information share does not follow the ranking by
exposure**, and it is common for a segment that survives 0.7 to contribute less
than one that survives 0.1. Only the tool knows which.

The eccentricity behaves differently, and unequally
----------------------------------------------------

``EPS1`` and ``EPS2`` multiply :math:`\cos 2\Phi` and :math:`\sin 2\Phi`. A wave
of half the period departs from a quadratic much sooner, so more of the
eccentricity signature survives the same spin fit than of ``A1``'s.

But only in one direction at a time. Which of :math:`\cos 2\Phi` and
:math:`\sin 2\Phi` looks quadratic over the arc observed depends entirely on
where in orbital phase that arc sits, so a single observation measures one
combination of ``EPS1`` and ``EPS2`` far better than the other -- by factors of
several, whatever its length. Closing the gap needs segments at *different*
orbital phases. That is a scheduling statement, and no exposure table shows it.

This is also why the eccentricity limit must come from the joint
``EPS1``-``EPS2`` covariance and never from the two marginals in quadrature:
when the two directions are measured unequally, the bound is set mostly by the
worse one. See :doc:`eccentricity`.

Reading the output
------------------

A run prints four blocks. The listing below is illustrative -- the format, not
anyone's data:

.. code-block:: text

   Information budget over 6 segments for A1, EPS1, EPS2

   Predicted 1-sigma uncertainties
   -------------------------------
     A1       0.004192
     EPS1     0.0002714
     EPS2     0.0002605

     With EPS1, EPS2 fixed at zero:
       A1       0.004106   (1.02x better)

   Eccentricity
   ------------
     Implied 3-sigma (99.73%) upper limit on e: 0.000914
     From the joint EPS1-EPS2 covariance, not the two marginals in quadrature.

   Correlation of the shared parameters
   ------------------------------------
                  A1     EPS1     EPS2
     A1        1.000    0.118    0.067
     EPS1      0.118    1.000    0.204
     EPS2      0.067    0.204    1.000

   Where the information on A1 comes from
   ------------------------------------
     segment                                        share  surviving   sigma if dropped
     epoch03_ev.nc                                 41.02%     0.3311     0.005458 (1.30x)
     epoch01_ev.nc                                 22.85%     0.5027     0.004771 (1.14x)
     epoch05_ev.nc                                 17.30%     0.1204     0.004603 (1.10x)
     epoch02_ev.nc                                 11.44%     0.6680     0.004432 (1.06x)
     epoch04_ev.nc                                  7.38%     0.2158     0.004349 (1.04x)
     epoch06_ev.nc                                  0.01%     0.0009     0.004192 (1.00x)
     Segments carrying 90% of the A1 information: 4 of 6
     Segments carrying 99% of the A1 information: 5 of 6

``Predicted 1-sigma uncertainties``
    What each shared parameter would come back with, and -- separately -- what
    it would come back with if the eccentricity were held at zero instead of
    fitted. The ratio is the price of letting ``EPS1`` and ``EPS2`` float. When
    it is close to 1.00, fitting the eccentricity is free and there is no
    argument for fixing it.

``Eccentricity``
    The upper limit the data support, built from the joint covariance by
    drawing from it and taking a percentile of
    :math:`\sqrt{\epsilon_1^2 + \epsilon_2^2}` -- the construction
    :mod:`ell1fit.eccentricity` applies to real posterior samples.

``Correlation of the shared parameters``
    How much the shared parameters are trading against each other. Small
    entries mean the leave-one-out column below can be read parameter by
    parameter; large ones mean it cannot.

``Where the information comes from``
    Per segment: its share of the diagonal information, the fraction surviving
    its own spin fit, and the uncertainty that would result if it were dropped.
    The share and the leave-one-out disagree when the shared parameters are
    correlated, and then the leave-one-out is the one to quote.

Note that the predicted uncertainties are a *curvature* forecast. They are a
lower bound on what a chain will report, and on a well-behaved posterior they
land within some tens of percent of it. Treat a large disagreement as
information about the posterior, not as a defect in either number.

Command-line options
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - option
     - meaning
   * - ``-p``, ``--parfiles``
     - One parfile per event file, in the same order. Required.
   * - ``-P``, ``--parameters``
     - Comma-separated parameters the fit would free. Default
       ``F0,F1,A1,EPS1,EPS2``. Per-file names are derived from these exactly as
       the fit derives them.
   * - ``--shared``
     - Restrict the budget to a subset of the shared parameters. The ones left
       out are **held fixed**, not marginalised, so the uncertainties returned
       are conditional on them. Narrowing to ``"A1"`` therefore answers "how
       well would ``A1`` be measured if the eccentricity were known?", which is
       not the same question as the default.
   * - ``--report``
     - Which parameter the per-segment table is about. Defaults to the first
       shared parameter.
   * - ``--use-weight``, ``--use-pi``, ``-e``, ``-N``
     - As in ``ell1fit``. They change the likelihood, so they change the answer.
   * - ``--ignore-uncertainties``
     - Give every segment the wide default spin prior instead of whatever its
       parfile happens to carry. See below.
   * - ``--no-priors``
     - Differentiate the likelihood alone, with no prior curvature at all.
   * - ``--step``
     - Step size for the central differences, in units of the preconditioned
       scale. The default is 0.3; see :ref:`information-numerics`.
   * - ``--seed``
     - Seed for the eccentricity limit's draws, for reproducible output.

Four things it is good for
--------------------------

**Deciding whether to include a segment.** Read the ``sigma if dropped``
column. A segment whose removal leaves the uncertainty at ``1.00x`` is
contributing nothing, and dropping it is free -- which makes it a clean
robustness check rather than a loss. A segment at ``1.30x`` cannot be dropped
without changing the result, so if you suspect it, you have to fix it rather
than remove it.

**Pricing the priors.** Run once as-is and once with ``--ignore-uncertainties``.
The difference is what the spin priors are worth. This is worth measuring
rather than assuming, for a reason explained below.

**Pricing the eccentricity.** The ``With EPS1, EPS2 fixed at zero`` line
already reports it. If the cost is 1.00x, the common practice of fixing the
eccentricity to buy precision elsewhere buys nothing.

**Planning.** Because the calculation is per segment and additive, a candidate
future observation can be priced by building its segment and adding it to the
budget -- without any of the data existing yet. What dominates that answer is
usually where the observation sits in orbital phase, not how long it is.

Priors are not a detail
-----------------------

By default the priors' curvature is included, because that is what the
posterior has. It matters more than it looks: a *narrow* Gaussian on ``F0_i``
and ``F1_i`` stops those parameters absorbing the orbital signature, and the
information that would have been eaten survives into ``A1`` instead. Running
again with ``--no-priors`` prices exactly that -- the difference between the two
is the part of the orbital constraint coming from the spin priors rather than
from the photons.

This is worth checking rather than assuming, because
:func:`ell1fit.models._get_par_dict` only supplies its wide default width when
a parfile leaves the uncertainty unset. A parfile carrying the uncertainty from
an *earlier fit to the same photons* silently becomes a tight prior instead,
and the result is both a double-counting of that data and, if only some
parfiles carry uncertainties, a fit that treats its segments very unequally for
no physical reason. ``--ignore-uncertainties`` gives every segment the wide
default; running ``ell1info`` both ways shows what that choice is worth.

The same mechanism makes the answer depend on the **order** of the files, since
:func:`ell1fit.models._load_and_validate_models` takes the global reference
model from the first parfile. If two runs disagree, check they used the same
order before attributing the difference to anything physical.

How it works
------------

The trick is that one binary is shared by every observation while each carries
its own spins, and file 3's spin parameters appear in file 3's likelihood term
and nowhere else. Writing one segment's observed information over its shared
and local parameters as

.. math::

   J_i = \begin{pmatrix} A_i & B_i \\ B_i^{T} & D_i \end{pmatrix},

profiling the spins away leaves the Schur complement
:math:`S_i = A_i - B_i D_i^{-1} B_i^{T}`, and because no local block is shared,

.. math::

   I = \sum_i S_i

exactly. One small Hessian per observation replaces one enormous Hessian over
everything, and the answer is the same. This is also what makes the per-segment
accounting meaningful in the first place: the total is a *sum* over segments,
so "this segment's contribution" is a well-defined object rather than a
heuristic.

.. _information-numerics:

Numerics
~~~~~~~~

The Hessians are central differences of the very log posterior the fit
maximises, with steps taken from the preconditioned scales, so the weights,
template and deorbiting tolerance in play are the ones being differentiated.
Before differencing, the nuisance parameters are re-optimized at the centre, so
the expansion is around a point that is actually stationary in them.

The ``--step`` default of 0.3 preconditioned units is a compromise: too small
and the differences are dominated by round-off in the likelihood, too large and
they pick up the quartic term. If a result looks suspicious, rerun at 0.15 and
0.6 -- a well-conditioned problem moves by well under a percent.

A segment whose nuisance block is singular raises
:class:`numpy.linalg.LinAlgError` naming that segment. In practice this means
the segment cannot constrain its own ``F0_i``/``F1_i`` at all, which is worth
knowing before a fit rather than after.

Using it from Python
--------------------

The command line is a thin wrapper. The pieces are usable directly, which is
what you want for a scan over candidate observations:

.. code-block:: python

   from ell1fit.pipeline import prepare_fit_state
   from ell1fit.observed_information import segment_information
   from ell1fit.information import InformationBudget

   state = prepare_fit_state(
       files, parfiles, fit_parameters=["F0", "F1", "A1", "EPS1", "EPS2"],
       use_weight=True,
   )
   segments = segment_information(state.observations, state.setup)
   budget = InformationBudget(["A1", "EPS1", "EPS2"], segments)

   print(budget.uncertainties())            # {'A1': ..., 'EPS1': ..., ...}
   print(budget.ranking("A1"))              # [(name, share), ...] descending
   print(budget.without("epoch06_ev.nc").uncertainties())
   print(budget.n_segments_for("A1", 0.9))  # how many carry 90%

:class:`~ell1fit.information.InformationBudget` also exposes
:meth:`~ell1fit.information.InformationBudget.covariance`,
:meth:`~ell1fit.information.InformationBudget.correlation` and
:meth:`~ell1fit.information.InformationBudget.cumulative`, each accepting a
``fixed`` argument that holds named parameters at zero. Because
:class:`~ell1fit.information.SegmentInformation` objects are just matrices with
names attached, a hypothetical segment can be added to the list and the budget
recomputed, which is how a future observation gets priced.

What it cannot tell you
-----------------------

``ell1info`` computes curvature at one point, under the model and priors it is
handed. It is silent about whether that model is right.

* **Unmodelled noise inside a segment does not appear.** Timing noise makes the
  real residuals correlated; the Fisher calculation assumes they are not, and
  will happily report a small uncertainty for a segment whose phases wander.
* **Neither does a systematic.** If the same photons, blocked two different
  ways, give different answers with small error bars on each, ``ell1info``
  reports both of them faithfully and flags nothing.
* **It is a local quantity.** A posterior with a second mode, or one that is
  strongly non-Gaussian, is not described by its curvature at one point.

All three are measured by refitting, not by inverting a matrix. The natural
companion check is to fit residual delays against orbital phase directly: that
measures the noise (through the reduced chi-squared) and the systematic
(through re-blocking) which this calculation cannot see.
