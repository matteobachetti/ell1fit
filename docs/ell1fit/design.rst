Why the code looks like this
============================

Some decisions in ``ell1fit`` look arbitrary until you know what goes wrong
without them. This page records those, so the reasoning does not have to be
rediscovered — several of them were, expensively.

Parameters are handled in local coordinates
-------------------------------------------

Every fitted parameter is represented as

.. math::

   \mathrm{physical} = \mathrm{local} \times \mathrm{factor} + \mathrm{initial}

and the optimizer and sampler both work entirely in the local variable.

The reason is conditioning. A joint fit may vary ``F0`` — a frequency in Hz
known to fifteen significant digits — alongside ``A1`` in light-seconds and a
pulse phase of order unity. An optimizer that steps the same distance in every
direction, or an MCMC that spreads its walkers by a fixed amount, behaves
sensibly only if those directions are comparably scaled. In local coordinates
the starting point is the origin and the convention is that **one standard
deviation is one local unit** for every parameter
(:data:`ell1fit.scaling.TARGET_LOCAL_SIGMA`). Corner plots are drawn in these
coordinates, so the choice is also what makes their axis labels read directly
as sigmas rather than carrying a shared multiplier on every panel.

That convention is easy to state and easy to get wrong, and the number itself
used to be ``1e-6``. The refinement loop was written with a convergence
threshold of ``0.1``, on the belief that the factors normalised each parameter
to order unity; against ``1e-6`` that was :math:`10^5\sigma`, so convergence was
declared on essentially every first pass. The threshold is now expressed as a
multiple of :data:`ell1fit.scaling.TARGET_LOCAL_SIGMA` so the units are visible
at the point of definition -- which is why it survived the constant changing.

Two things downstream were written in absolute terms and so were quietly tied
to that number. ``scipy.optimize``'s finite-difference probe is one; it is now
set explicitly, as :data:`ell1fit.scaling.OPTIMIZER_EPS`, in units of sigma.
``emcee.moves.DESnookerMove`` is the other, and is discussed under
:doc:`performance`.

The factors are measured, not assumed
--------------------------------------

:func:`ell1fit.scaling.get_factors` derives each parameter's scale from whatever
uncertainty is available — but for some parameters there is none. ``Phase_i``
has its uncertainty recorded as zero, no model-based estimate covers it, and it
falls through to a default of 1 while every other parameter gets a data-derived
value. The result was directions differing in scale by a factor of a thousand.

Rather than invent a formula for each parameter type,
:func:`ell1fit.scaling.precondition_factors` measures the curvature of the
actual posterior and rescales. This needs no per-parameter knowledge and adapts
to the data.

Times are relative to each file's own PEPOCH
--------------------------------------------

Spin phase is :math:`F_0 t`, so the magnitude of :math:`t` directly determines
how much of the float64 significand survives. Referencing every file to a common
epoch would inflate :math:`t` by the campaign length and consume precision that
:doc:`performance` shows is only comfortable because :math:`t` stays small.

This is also why the parameter dictionaries carry ``F0_0``, ``F0_1``, … and
``PEPOCH_0``, ``PEPOCH_1``, … rather than a single spin solution.

TASC is only defined modulo PB
-------------------------------

The shared timing model is re-referenced to the mean ``PEPOCH``, so the ``TASC``
a fit reports may differ from the input parfile's by a whole number of orbits
while describing the identical orbit. In one test dataset the difference was 17
orbits — about 43 days.

Any comparison of two ``TASC`` values must therefore reduce the difference
modulo ``PB`` first. Comparing them directly produces a spectacular but
meaningless discrepancy.

Superluminal orbits are rejected, not attempted
------------------------------------------------

Deorbiting solves :math:`t_{\rm emit} = t_{\rm obs} - A_1 \sin(\omega
t_{\rm emit})` by fixed-point iteration, which converges only while the map is a
contraction — that is, while :math:`A_1\omega`, the projected orbital velocity
in units of :math:`c`, stays below 1. Beyond it there is no fixed point and no
inverse to find.

An optimizer or sampler exploring freely *will* propose such positions, and a
Gaussian prior cannot stop it: its log-density there is enormously negative but
still finite, so a guard testing for :math:`-\infty` never fires. The posterior
therefore screens with :func:`ell1fit.phase_utils.orbit_is_invertible` and
returns :math:`-\infty` itself, and the iteration carries a hard cap so that no
input can make it run forever.

Templates are built before priors
----------------------------------

The template determines each file's phase-zero offset, and that offset is the
value the ``Phase_i`` prior is centred on. Assigning priors first leaves them
centred on a placeholder. This ordering is enforced by the structure of
:func:`ell1fit.pipeline.ell1fit` and noted at the call site.

The comparison phaseogram's reference is captured early
--------------------------------------------------------

The left-hand panel of the before/after phaseogram shows the solution the run
started from. It is captured *before* refinement, because refinement re-centres
the baseline: deriving the reference afterwards yields the refined solution, and
the comparison becomes that solution plotted against itself — a diagnostic that
looks perfect however badly the fit went.

EPS1 pairs with cos 2Phi, EPS2 with sin 2Phi
---------------------------------------------

ELL1 replaces the eccentricity and the longitude of periastron, which are
individually ill-defined for a nearly circular orbit, with the Laplace–Lagrange
pair

.. math::

   \epsilon_1 = e \sin\omega, \qquad \epsilon_2 = e \cos\omega,

and expands the Roemer delay. To first order in :math:`e`,

.. math::

   \Delta_R(t) = x \left[ \sin\Phi
                 + \frac{\epsilon_2}{2}\sin 2\Phi
                 - \frac{\epsilon_1}{2}\cos 2\Phi \right].

The parameterisation is due to Lange et al. (2001), *MNRAS* **326**, 274; the
implementation of record is tempo's `bnryell1.f
<https://github.com/nanograv/tempo/blob/master/src/bnryell1.f>`_, whose header
states the two definitions above and credits Wex (1998). PINT is a faithful
port of it. What both actually compute — and what this package computes — is one
order further, with the Wex–Zhu :math:`O(e^2)` block written out in
:doc:`motivation`. See `Second order costs nothing`_ for why that is free.

Note that the pairing is *asymmetric*: :math:`\epsilon_2` goes with
:math:`\sin 2\Phi` and :math:`\epsilon_1` with :math:`-\cos 2\Phi`. Exchanging
them is a first-order error, not a subtlety.

This package got it wrong, and the way it got it wrong is worth recording. The
old expression was

.. math::

   \Delta_R(t) = x \left[ \sin\Phi
                 + \frac{\epsilon_1}{2}\sin 2\Phi
                 + \frac{\epsilon_2}{2}\cos 2\Phi \right],

which is not a random transposition of two labels. ``bnryell1.f`` computes both
``dre``, the delay, and ``drep``, its derivative with respect to orbital phase.
The old eccentric terms are **exactly half of** ``drep``\ 's — they agree to
1.1e-16 — grafted onto :math:`\sin\Phi`. The halving is what made the result
look plausible, since ``dre``\ 's eccentric terms carry a factor
:math:`\tfrac{1}{2}` and ``drep``\ 's carry 1. Somebody read one line too far
down the file.

Two things then conspired to hide it for a long time.

**Only** :math:`\omega` **moves.** Under the exchange,
:math:`e = \sqrt{\epsilon_1^2 + \epsilon_2^2}` is invariant, so the eccentricity
*magnitude* — the quantity anyone actually reports — comes out right. What
rotates is :math:`\omega`, by exactly 90 degrees. Nothing looks obviously
broken.

**The generator shared the mistake.** :mod:`ell1fit.tests.datagen` is
deliberately independent of ``phase_utils``, so that agreement between them is
evidence rather than tautology — but its orbital delay had been written to match
the package rather than the published model. Every recovery test therefore
injected the wrong orbit, fitted it with the same wrong orbit, and passed. This
is the general hazard, and it is worth stating in the abstract: **a generator
that shares the model under test can only ever demonstrate self-consistency.**
It catches implementation errors and is blind to convention errors.

So the guard is a comparison against something neither implementation can
influence: an exact Keplerian orbit, with Kepler's equation solved numerically
and the delay evaluated as :math:`x\,(r/a)\sin(\omega + \nu)`. Because ELL1
truncates at first order, the residual against an exact orbit must fall as
:math:`e^2`; a mispairing leaves one that falls as :math:`e`. Measured:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - :math:`e`
     - correct pairing
     - exchanged pairing
   * - 1e-4
     - 9.0e-09
     - 7.1e-05
   * - 1e-3
     - 9.0e-07
     - 7.1e-04
   * - 1e-2
     - 9.0e-05
     - 7.1e-03

A hundredfold drop per decade against a tenfold one. ``test_ell1fit.py``
asserts both the bound and the scaling exponent, over several values of
:math:`\omega`, which is a statement no documentation can drift away from. With
the :math:`O(e^2)` block included the same test demands a *thousandfold* drop,
so it pins the pairing and the expansion order together.

Second order costs nothing
--------------------------

ELL1 is normally written to first order, and carrying the :math:`O(e^2)` block
sounds like paying for accuracy nobody needs on a nearly circular orbit. It is
the other way round: the second-order kernel is **faster** than the first-order
one it replaced.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - deorbiting 200,000 events
     - first order
     - with :math:`O(e^2)`
   * - :math:`e = 0`
     - 2.18 ms
     - 2.08 ms
   * - :math:`e = 10^{-2}`
     - 2.79 ms
     - 2.49 ms

The reason is that the deorbiting iteration is **transcendental-bound** — it is
limited by ``sin``/``cos`` throughput, not by arithmetic. Written naively the
second-order delay needs five transcendental calls per iteration
(:math:`\sin\Phi`, :math:`\sin 2\Phi`, :math:`\cos 2\Phi`, :math:`\sin 3\Phi`,
:math:`\cos 3\Phi`) against the first-order form's three. But collecting the
whole expression into six harmonic coefficients
(:func:`ell1fit.phase_utils._second_order_coefficients`) lets every multiple
angle come from :math:`\sin\Phi` and :math:`\cos\Phi` through exact identities,
so it needs **two**. The extra polynomial work costs less than the one trig call
it saves.

So there is no first-order option, and no flag. Carrying a second code path
would have bought a slower, less accurate model. At :math:`e = 0` the kernel is
bit-for-bit identical to the circular one, so nothing changes for the orbits
this code is usually pointed at.

The cost of the old convention, for scale: at :math:`e = 0.005` the delay was
wrong by 0.0035\ :math:`\,x`, which for ``A1`` = 22.2 lt-s and ``F0`` = 7.5 Hz
is 0.078 s, or 0.58 rotations.

.. warning::

   PINT's ``ELL1model.delayI`` docstring writes the delay as
   ``a1*(sin(Phi)+eps1/2*sin(2*Phi)+eps1/2*cos(2*Phi))``. It names ``eps1``
   twice, and it puts ``eps1`` on ``sin(2*Phi)`` where PINT's own computing path
   puts ``eps2``. It is a stale comment on the inverse-timing expansion and
   contradicts the code beneath it; do not read the model out of it.

Orbital derivatives are propagated per epoch, not fitted
--------------------------------------------------------

The fit uses one binary: a single ``PB``, ``TASC``, ``A1``, ``EPS1`` and
``EPS2``, shared by every file. But a solution is only valid at the epoch it is
referenced to, and an orbital derivative carries it away from that epoch. For a
long time the shared values were taken from one snapshot at the mean ``PEPOCH``
and applied to every file regardless of when it was observed.

The resulting phase error grows as :math:`\dot{P_b} \times \mathrm{baseline}^2`.
Measured against the exact orbit count
:math:`N(t) = x - \tfrac{1}{2}\dot{P_b}x^2`, with :math:`x = (t - T_{asc})/P_b`:

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - :math:`\dot{P_b}`
     - baseline
     - one mean-epoch snapshot
     - propagated per epoch
   * - 1e-11
     - 1 yr
     - 2.9e-05 cycles
     - 1.3e-09
   * - 1e-10
     - 10 yr
     - 3.0e-02 cycles
     - 7.5e-09
   * - 1e-09
     - 10 yr
     - 3.0e-01 cycles
     - 2.5e-07

A fit resolves about 1e-3 cycles, so the snapshot is wrong by more than the
measurement is worth over a multi-year baseline with a derivative in the range
redbacks and black widows actually show.

Each file therefore carries a **fixed** offset — ``PB_offset_i``,
``TASC_offset_i``, ``A1_offset_i``, ``EPS1_offset_i``, ``EPS2_offset_i`` — added
to the global value before phases are computed. Three things about them are
deliberate.

**They are constants, not parameters.** The offsets are second order
(:math:`\Delta P_b = \dot{P_b}\,\Delta t`), so moving ``PB`` by its own
uncertainty changes them negligibly. They can be computed once at load and never
revisited inside the likelihood.

**PINT computes them.** The propagation goes through the same
``change_binary_epoch`` that aligns the models in the first place. Deriving it
here would make agreement with PINT a tautology instead of a check, and would
silently drop the parameterizations (``FB0``/``FB1``) and the other derivatives
(``A1DOT``, ``EPS1DOT``, ``EPS2DOT``) that it already handles.

**The** ``TASC`` **offset is the one that matters, and it is not obvious.**
:func:`ell1fit.phase_utils._calculate_phases` brings ``TASC`` near each
``PEPOCH`` by wrapping it modulo ``PB``, which re-adds :math:`n P_b` computed
with the *trial* period — and that is right, because it is how a ``PB`` error
grows into a phase offset across epochs, which is what makes ``PB`` measurable
from multi-epoch data at all. What the wrap cannot express is the quadratic term
the exact model accumulates, :math:`n^2 P_b \dot{P_b}/2`. ``TASC_offset``
supplies exactly that residual.

Correcting the period *without* it is the tempting half of this fix, and it
accomplishes nothing: it flips the sign of the residual while leaving its
magnitude alone. Measured at :math:`\dot{P_b}` = 1e-10 over three years, the
period correction alone improves the error by **0.15%**, against a factor of
**300,000** when both are applied.

Parameter names share a namespace with result fields
-----------------------------------------------------

:func:`ell1fit.fitting.optimize_solution` merges the parameter dictionary into
the results dictionary before writing the output table. The two therefore share
one namespace, and a parameter whose name collides with a result field silently
overwrites it — no error, no warning, just a wrong number in a column.

This is not hypothetical. The epoch offsets above were first called ``dPB_i``,
``dA1_i``, and so on, which reads naturally. But posterior percentiles are
written as ``d<par>_<percentile>``, so ``dA1_1`` — intended as "the ``A1``
offset for file 1" — landed on top of **the first percentile of the fitted**
``A1``. ``tools/refactor_net.py`` caught it; nothing else would have.

Hence the ``<PAR>_offset_<i>`` spelling, which cannot collide, and a test that
asserts no parameter key ends in something a percentile field could produce. Any
new per-file quantity should be checked the same way.

Tests assert physics, not stored numbers
-----------------------------------------

The obvious way to protect a numerical pipeline is to pin its output and assert
equality forever after. This project deliberately does not, for two reasons.

Pinned numbers cannot distinguish *correct* from *consistently wrong*: a bias
that predates the reference file is locked in and reported as success. And the
MCMC path depends on the RNG stream, so pinned percentiles fire on every library
upgrade, training everyone to regenerate the reference without looking.

So the checked-in tests assert that the fit recovers a known injected solution
within its own quoted uncertainty. Bit-for-bit comparison is still the right
tool for verifying that a *refactor* changed nothing — but it belongs in
``tools/refactor_net.py``, run before and after on one machine, not in the test
suite.

One corollary is worth stating: **do not assert that one measurement beats
another unless the margin is real.** Two tests in this repository originally
demanded an improvement in a regime where both quantities were noise; both
passed locally and failed in CI. Either make the case hard enough that the
effect exists, or assert the absolute property that must hold.
