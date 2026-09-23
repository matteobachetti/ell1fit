Spin statistics: ``ell1stat``
=============================

``ell1stat`` summarises a pulsar's spin history from a set of single-epoch
``ell1fit`` results -- the same files ``ell1decay`` takes -- plus, optionally,
tables of extra epochs that never had a full ``ell1fit`` run::

    ell1stat nu*_results.ecsv --extra literature.csv -o mysource

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

The two are compared (``secular_minus_local_sigma``). In an accreting source
they need not agree: local ``F1`` follows the torque at the time of each
observation, while the secular trend integrates every torque episode,
including the ones no observation caught.

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
