The pipeline, stage by stage
============================

This page describes what :func:`ell1fit.pipeline.ell1fit` actually does, in the
order it does it. Several of the orderings are load-bearing and are called out
where they occur.

.. contents::
   :local:
   :depth: 1

1. Loading models and events
----------------------------

:mod:`ell1fit.models` reads one PINT timing model per event file and checks it
is an ELL1 model with ``TASC`` defined. Each model is re-referenced to its own
``PEPOCH``; a separate reference model is re-referenced to the *mean* ``PEPOCH``
and supplies the orbital parameters shared across files.

:mod:`ell1fit.events` then loads the event files and expresses arrival times as
**seconds from each file's own** ``PEPOCH``. This is not a cosmetic choice.
Spin phase is :math:`F_0 t`, so a distant reference epoch inflates :math:`t` and
consumes the float64 significand that everything downstream depends on. Keeping
:math:`t` small keeps the phase error at :math:`1.6\times10^{-10}` cycles for a
100 ks observation — see :doc:`performance`.

Two representations of the parameters are built here and used throughout:

``parameters``
   ``{name: value}``, the working set that phases are computed from.
``parameters_with_unc``
   ``{name: [value, uncertainty]}``, used to build priors and scaling.

Spin parameters are **per file** (``F0_0``, ``F0_1``, …), because each file has
its own ``PEPOCH`` and a spin frequency is only valid at its epoch. Orbital
parameters are global: one binary, one solution. The phase offset ``Phase_i`` is
per file as well, a free nuisance parameter absorbing each observation's
arbitrary phase zero.

A global orbital solution is still only valid at the epoch it is referenced to.
When the parfile sets an orbital derivative — ``PBDOT``, ``EPS1DOT`` or
``EPS2DOT`` — each file therefore also carries a **fixed** correction
(``PB_offset_i``, ``TASC_offset_i``, ``A1_offset_i``, …) that carries the shared
solution to that file's own ``PEPOCH``. These are not fitted and do not vary
with the trial parameters; PINT computes them once at load. They are exactly
zero when no derivative is set. See
:func:`ell1fit.models._orbital_epoch_offsets`, which also explains why the
``TASC`` correction is the one that matters.

``A1DOT`` is the exception, and is excluded from those offsets on purpose: a
fixed offset cannot be fitted. Each file instead carries its **lever arm**
``binary_dt_i``, the time in seconds from the shared binary epoch to its own,
and the orbital amplitude is rebuilt as ``A1 + A1_offset_i + A1DOT *
binary_dt_i`` on every likelihood evaluation. See
:doc:`orbital_derivatives`.

.. note::

   ``TASC`` is only defined modulo ``PB``. Because the shared model is
   re-referenced to the mean ``PEPOCH``, the ``TASC`` reported by a fit can
   differ from the input parfile's by a whole number of orbits while describing
   the identical orbit. Any comparison of ``TASC`` values must be reduced modulo
   ``PB`` first.

2. Folding and event weights
----------------------------

Events are folded into pulse profiles using the current solution. With
``--use-weight``, :mod:`ell1fit.weighting` additionally estimates a per-event
weight from how the pulsed amplitude varies with energy. Each event's phase is
projected onto the harmonic model of that observation's own pulse profile,
which gives an unbiased per-event estimate of the local pulsed amplitude, and
those estimates are fit against :math:`\log E` with a penalised cubic spline
whose smoothing strength is chosen by generalised cross-validation. Nothing is
binned in energy: energy bands hold wildly different numbers of counts, so any
fixed binning either smears real structure where counts are plentiful or reports
noise as signal where they are scarce.

Weights are normalised into :math:`[0, 1]`, which the weighted likelihood
requires — 1 means "trust this event's phase fully", 0 means "treat it as
unmodulated background". Only the *shape* of the curve matters: the weighted
profile and its noise level both scale linearly with the weights, so the peak
normalisation is free to serve the :math:`[0, 1]` constraint alone.

3. Building templates
---------------------

:mod:`ell1fit.templates` turns each folded profile into a smooth template by
truncating its Fourier series at ``nharm`` harmonics, which both suppresses
counting noise and yields a continuous function evaluable at any phase.

.. important::

   This stage **must** precede the next one. The template determines each file's
   phase-zero offset, and that offset is the value the ``Phase_i`` prior is
   centred on. Running the priors first leaves the prior centred on a
   placeholder — a bug that existed and was fixed.

4. Assembling the fit
---------------------

The requested parameter names are expanded into the per-file set actually
fitted (``F0`` becomes ``F0_0``, ``F0_1``, …), priors are attached
(:mod:`ell1fit.priors`) and scaling factors computed (:mod:`ell1fit.scaling`).

Everything from here on works in **local coordinates**:

.. math::

   \mathrm{physical} = \mathrm{local} \times \mathrm{factor} + \mathrm{initial}

The reason is conditioning. The fitted quantities span wildly different
magnitudes — ``F0`` in Hz to fifteen significant digits alongside ``A1`` in
light-seconds — and an optimizer or MCMC walker that steps the same distance in
every direction only behaves sensibly if those directions have comparable
scale. The convention is that **one standard deviation is one local unit** for
every parameter (:data:`ell1fit.scaling.TARGET_LOCAL_SIGMA`).

5. Conditioning the scales
--------------------------

The factors from :func:`ell1fit.scaling.get_factors` are derived from whatever
uncertainty information is available, and for some parameters there is none:
``Phase_i`` has no recorded uncertainty and falls through to a default. The
result is directions whose local scales differ by a factor of a thousand, which
L-BFGS-B cannot cope with — it maintains a single Hessian approximation, and no
step size suits all directions at once.

:func:`ell1fit.scaling.precondition_factors` therefore measures each direction's
curvature from the posterior itself and rescales so that the convention above
actually holds. Measured effect: the optimizer goes from reaching the global
optimum in 7 of 12 starts to 12 of 12. See :doc:`performance`.

6. Iterative template refinement
--------------------------------

With ``--template-iterations N`` above 1, :mod:`ell1fit.refinement` alternates
between refitting the solution and rebuilding the template from a fold made with
that improved solution, until the fit stops moving.

Two safeguards are worth knowing about:

- **Convergence is judged in parameter space**, as the size of the point
  estimate's step in local coordinates. Because preconditioning has put every
  parameter on the same scale, one threshold is meaningful across ``F0``,
  ``PB``, ``A1`` and ``TASC`` simultaneously.
- **The best iterate is kept, not the last.** Refinement is not guaranteed to
  improve monotonically, so each pass is scored by the folded profile's
  :math:`Z_n^2` and the best retained.

There is a risk in the other direction, and it is why ``nharm`` is held fixed
while iterating: a template refined against the data it was fitted to can begin
absorbing noise, which then pulls the solution. :doc:`limitations` records what
was measured.

7. Optimization and sampling
-----------------------------

:func:`ell1fit.fitting.point_estimate_fit` runs a bounded local optimization —
bounded because several priors return :math:`-\infty` outside a hard window, and
an unconstrained optimizer cannot see that. :func:`ell1fit.fitting.optimize_solution`
optionally uses it to warm-start, then explores the posterior with ``emcee``,
checking convergence against the integrated autocorrelation time and
checkpointing to an HDF5 backend so a run can be resumed or extended.

Trial positions outside the physically invertible region are rejected as
impossible: if the implied projected orbital velocity reaches :math:`c` the
arrival-time map is no longer monotonic and pulse phases are undefined. See
:func:`ell1fit.phase_utils.orbit_is_invertible`.

8. Results
----------

:mod:`ell1fit.results_io` writes a combined result table plus one per input
file, and :mod:`ell1fit.create_parfile` folds the fitted values back into
updated parfiles, so a fit can serve as the ephemeris for the next one.

Diagnostic figures are written alongside: light curves, templates, likelihood
traces over each ``Phase_i``, a corner plot, and side-by-side phaseograms
comparing the starting solution against the fitted one.

.. note::

   The left-hand phaseogram panel shows the solution the run *started* from.
   It is captured before refinement runs, because refinement re-centres the
   baseline — deriving it afterwards produced a comparison of the refined
   solution against itself, which looked perfect no matter how the fit went.
