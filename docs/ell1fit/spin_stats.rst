Spin statistics: ``ell1stat``
=============================

``ell1stat`` summarises a pulsar's spin history from a set of single-epoch
``ell1fit`` results -- the same files ``ell1decay`` takes -- plus, optionally,
tables of extra epochs that never had a full ``ell1fit`` run::

    ell1stat nu*_results.ecsv --extra literature.csv -o mysource \
        --min-period 50 --max-period 70 --rate-exclude 'chandra*'

Inputs
------
From each ``ell1fit`` result it reads ``PEPOCH`` and the *fitted* ``F0`` and
``F1`` -- the posterior medians and 16th/84th percentiles. The bare ``F0``/``F1``
columns of a result file are the fit's starting values and are not used. An
``F1`` that was held fixed is not a measurement and is left out of every
statistic.

It also records ``pf × ctrate`` as ``pulsed_rate``: the pulse semi-amplitude,
``(max - min) / 2`` of the profile, in counts/s. A constant background cancels
out of it, so it is a background-free proxy for the pulsed luminosity -- but
only within one instrument.

An ``--extra`` table (any format astropy can read; repeat the option for
several) needs ``MJD``, ``F0`` and ``F0_err`` columns, and may add ``F1``,
``F1_err``, ``pulsed_rate`` and ``label``. Points are grouped by ``label`` in
the plots; every ``ell1fit`` point is labelled ``ell1fit``.

The ``Start``/``Stop`` columns of result files are not used: ``ell1fit``
copies them from the input parfile's ``START``/``STOP`` when present, and these
are often stale.

What it computes
----------------
Secular trend
    A polynomial in time (``--degree``, default 1) fitted to ``F0``, referenced
    to ``--reference-epoch`` (default: the mean epoch). Its slope is the
    long-term ``F1``.

Local ``F1``
    The mean of the ``F1`` values measured within each epoch, their intrinsic
    scatter, the plain inverse-variance mean with its chi-squared, and the
    fraction of epochs spinning up.

Torque and luminosity
    The Spearman rank correlation between local ``F1`` and the pulsed rate
    ``R`` (model-free), and a fit of ``F1 = B + A (R / R_ref)^alpha``, with
    ``R_ref`` the median rate. ``B`` is a rate-independent term, e.g. a
    spin-down torque, which lets the relation cross zero, as it must for a
    source seen both spinning up and down. ``alpha`` is fitted both fixed at
    ``--torque-index`` (default 6/7, disc accretion onto a magnetised star)
    and free, with the ``Delta chi2 = 1`` interval from a chi-squared profile
    with the intrinsic scatter held at its fixed-index value. That interval is
    slightly narrow (61% coverage in simulations, rather than 68%). When it
    reaches an edge of the profiled range (0.05-5), it is reported as an upper
    or lower limit (``interval_open`` is ``"below"`` or ``"above"``), or as
    unconstrained (``"both"``). ``--rate-exclude`` applies here too.

The secular and local ``F1`` are compared (``secular_minus_local_sigma``). In an accreting source
they need not agree: local ``F1`` follows the torque at the time of each
observation, while the secular trend integrates every torque episode,
including the ones no observation caught.

Periodicity search
------------------
Given ``--min-period`` and ``--max-period`` (days), local ``F1`` (a proxy for
the accretion torque) and the pulsed rate (a proxy for the pulsed luminosity)
are each searched for a sinusoid within that range
(:mod:`ell1fit.spin_periodicity`):

* The periodogram fits a sinusoid plus a free constant at each trial period
  (the "floating-mean" Lomb-Scargle periodogram; ``--oversample`` frequency
  steps per inverse baseline). ``F1`` is weighted by its error and the
  intrinsic scatter in quadrature; the pulsed rate is unweighted.
* The false-alarm probability comes from keeping the observation times and
  shuffling the values among them (``--n-shuffle``): the fraction of shuffles
  whose highest peak in the range beats the real one. Analytic false-alarm
  formulas assume even sampling and white noise, and fail for a few
  campaigns years apart.
* The detectable amplitude is the sinusoid semi-amplitude that, injected at a
  random period in the range and a random phase into shuffled data, produces a
  peak above the ``--fap-level`` threshold 90% of the time. For sparse data
  this upper-limit-like number is often the more useful one. A particular
  sinusoid can still be detected below it, if its period and phase happen to
  suit the sampling.
* The spectral window (the periodogram of the sampling alone) is plotted
  under the data: a data peak where the window peaks is suspect.

* A joint search looks for one period shared by local ``F1`` and the pulsed
  rate, over the epochs where both are measured. Each quantity keeps its own
  mean, amplitude and phase, so a lag between torque and luminosity is allowed;
  the lag of the pulsed-rate maximum after the ``F1`` maximum is reported, in
  cycles. The joint power is the mean of the two powers. The shuffles move each
  epoch's pair of values together. A torque-luminosity relation makes the two
  quantities correlated epoch by epoch, and correlated quantities have
  coincident periodogram peaks whether or not anything is periodic;
  shuffling them independently would call every such coincidence significant.

* ``--leave-one-out`` repeats every search (single and joint) with each epoch
  left out in turn, logs the range of best periods and false-alarm
  probabilities, and writes them to ``{outroot}_leave_one_out.ecsv``. A
  candidate carried by one or two epochs moves or fades when they are dropped.

``--rate-exclude PATTERN`` (repeatable, shell-style, matched against file name
and label) leaves epochs out of everything that uses the pulsed rate -- for
example another instrument's, whose count rates are not comparable -- while
keeping their ``F0`` and ``F1``.

Caveats
    One sinusoid is fitted across the whole baseline, so the period must be
    stable over it: points ``T`` apart stay in phase only if the period is
    steady to about ``P**2 / T``. And the shuffling assumes the values are
    exchangeable between epochs. Epochs a few days apart within one campaign
    are usually correlated -- a smooth rise within a campaign mimics part of a
    sinusoid, and shuffling breaks that smoothness -- so the false-alarm
    probability is optimistic when the data are clustered. Check a candidate
    with ``--leave-one-out``.

Fitting with extra scatter
--------------------------
Accretion-torque noise moves ``F0`` far more than its measurement errors, so a
plain weighted fit gives an enormous chi-squared and error bars far too small.
Every fit adds one intrinsic scatter term in quadrature to the errors, set so
that chi-squared per degree of freedom is 1. When the scatter dominates, the
fit becomes effectively unweighted, and a single very precise epoch cannot pin
it. When the data already agree with their errors, nothing is added. The
correlation of torque noise between nearby epochs is not modelled, so these are
rough estimates with indicative error bars. Asymmetric errors are averaged.

Outputs
-------
``{outroot}_results.json``
    The numbers above.
``{outroot}_epochs.ecsv``
    The input epochs as read, with each epoch's residual from the secular trend.
``{outroot}_trend``
    ``F0`` against time with the secular fit, over the residuals. Each epoch's
    local ``F1`` is drawn as a short segment through its residual, with slope
    ``F1 - F1_secular``: a segment tilting up is an epoch spinning up faster
    than the long-term trend.
``{outroot}_f1``
    Local ``F1`` against time, with their mean, the intrinsic-scatter band and
    the secular ``F1``.
``{outroot}_torque``
    Local ``F1`` against pulsed rate, with the fixed- and free-index relations.
``{outroot}_periodogram``
    With a periodicity search: power against period for each quantity, with the
    shuffled-data detection threshold, the joint power, and the spectral windows.
``{outroot}_folded``
    With a periodicity search: each quantity folded at its own best period,
    with the best-fit sinusoid.
``{outroot}_folded_joint``
    With a periodicity search: both quantities folded at the joint period.
``{outroot}_leave_one_out.ecsv``
    With ``--leave-one-out``: for each search and each dropped epoch, the best
    period, its power and its false-alarm probability.
