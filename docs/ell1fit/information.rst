Information budget: ``ell1info``
================================

``ell1info`` answers a question a multi-file fit cannot answer for itself:
*which observations are actually measuring the orbit?* It takes the same files,
parfiles and flags an ``ell1fit`` run would take, and instead of fitting
anything it reports the curvature of the likelihood those inputs define::

    ell1info nu*_ev.nc blk*_ev.nc -p nu*_ev.par blk*_ev.par \
        -P "F0,F1,A1,EPS1,EPS2" --use-weight --report A1

Nothing is sampled, so a question that would cost days of MCMC costs a minute.

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
everything, and the answer is the same. The Hessians are central differences of
the very log likelihood the fit maximises, with steps taken from the
preconditioned scales, so the weights, template and deorbiting tolerance in
play are the ones being differentiated.

Reading the output
------------------

``Predicted 1-sigma uncertainties``
    What each shared parameter would come back with, and -- separately -- what
    it would come back with if the eccentricity were held at zero instead of
    fitted. The ratio is the price of letting ``EPS1`` and ``EPS2`` float.

``Eccentricity``
    The upper limit the data support, built from the joint ``EPS1``-``EPS2``
    covariance. It is never the two marginals added in quadrature: that throws
    away their correlation, and it ignores that a length cannot come out
    negative. See :doc:`eccentricity`.

``Where the information comes from``
    Per segment: its share of the diagonal information, the fraction surviving
    its own spin fit, and the uncertainty that would result if it were dropped.
    The share and the leave-one-out disagree when the shared parameters are
    correlated, and then the leave-one-out is the one to quote.

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

What it cannot tell you
-----------------------

``ell1info`` computes curvature at one point, under the model and priors it is
handed. It is silent about whether that model is right. Unmodelled torque noise
inside a segment does not appear, and neither does the systematic that shows up
when the same photons, blocked two different ways, give different answers with
small error bars on each. Both of those are measured by refitting, not by
inverting a matrix.
