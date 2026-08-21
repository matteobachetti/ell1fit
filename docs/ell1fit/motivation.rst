Scientific motivation
=====================

What the problem is
-------------------

A pulsar in a binary system is a clock in orbit. Two effects shape when its
pulses arrive at a telescope:

1. **Spin evolution.** The star rotates at a frequency :math:`F_0` that changes
   slowly, described by derivatives :math:`F_1, F_2, \ldots`.
2. **Orbital motion.** As the pulsar moves around its companion, its distance
   from us changes, and the pulses are delayed or advanced by the light-travel
   time across that displacement — the Roemer delay.

Recovering the orbit means measuring how the second effect modulates the first.
For a nearly circular orbit the natural description is the **ELL1** model, which
avoids the numerical degeneracy that afflicts the classical Keplerian
parametrisation when the eccentricity approaches zero: with :math:`e \to 0` the
argument of periastron :math:`\omega` and the time of periastron :math:`T_0`
become individually meaningless while their combination stays well determined.
ELL1 sidesteps this by using the time of the ascending node :math:`T_{ASC}` and
the two Laplace–Lagrange parameters

.. math::

   \epsilon_1 = e \sin\omega, \qquad \epsilon_2 = e \cos\omega,

which stay finite and uncorrelated as the orbit circularises. The delay is then

.. math::

   \Delta_R(t) = x \left[ \sin\Phi
                 + \frac{\epsilon_2}{2}\sin 2\Phi
                 - \frac{\epsilon_1}{2}\cos 2\Phi \right],
   \qquad \Phi = \frac{2\pi (t - T_{ASC})}{P_B},

with :math:`x = A_1` the projected semi-major axis in light-seconds and
:math:`P_B` the orbital period.

Why X-ray data need a different method
--------------------------------------

Radio pulsar timing folds many pulses into a high signal-to-noise profile and
measures one *time of arrival* per observation, then fits a model to those
arrival times. X-ray observations of accreting or millisecond pulsars often
cannot: the source may be faint enough that an entire observation yields a few
thousand photons, so no individual segment produces a usable arrival time.

``ell1fit`` therefore fits the **individual photon arrival times** directly.
Rather than compressing the data into arrival times and then fitting, it
evaluates how well a candidate timing solution concentrates *every recorded
photon* into a sharp pulse, and searches the parameter space for the solution
that does that best. Nothing is averaged away before the fit.

How a solution is scored
------------------------

For a trial set of parameters :math:`\theta`, the pipeline removes the orbital
delay from each photon's arrival time, converts the result to a pulse phase, and
asks how probable the resulting phases are under a model of the pulse shape.
With a template :math:`\lambda(\phi)` normalised to unit integral, the
log-likelihood is a sum over photons,

.. math::

   \log L(\theta) = \sum_{j} \log \lambda\bigl(\phi_j(\theta)\bigr),

which is the form given by Pletsch & Clarke (2014) for photon-by-photon pulsar
timing. A correct :math:`\theta` piles the photons up under the peak of
:math:`\lambda`; a wrong one smears them across all phases and the sum drops.

Two refinements matter in practice, and both are supported:

**Energy weighting.** The pulsed fraction of an X-ray pulsar usually varies with
photon energy. Photons from a band where the pulse is strong carry more timing
information than photons from a band where it is weak, and weighting them
accordingly recovers signal that an unweighted fit discards. See
:mod:`ell1fit.weighting`.

**Self-consistent templates.** The template has to come from the data, by
folding it — but folding requires a timing solution, which is what we are trying
to measure. If the starting ephemeris is imperfect the fold is smeared, and
because orbital errors produce *structured* rather than random phase residuals,
the resulting template is **skewed** rather than merely broadened. A pure offset
would be absorbed by the free phase parameter; a skew is not, and biases the
fit. Iterating the fold and the fit together removes it — see
:doc:`pipeline` and :mod:`ell1fit.refinement`.

Why the answer is a posterior, not a point
------------------------------------------

The likelihood surface for this problem is not a simple peak. Pulse phase is
periodic, so a timing solution wrong by a whole number of rotations folds the
data just as sharply as the right one; and parameters trade off against each
other — an error in :math:`A_1` shifts the orbital delay in a way that a change
in the phase offset can partly absorb.

A single best-fit value with an error bar hides that structure. ``ell1fit``
explores the parameter space with MCMC and reports the posterior, so
correlations and multiple modes are visible rather than silently collapsed.
The optional deterministic minimisation exists only to find a good starting
point for that exploration.

References
----------

- Pletsch, H. J. & Clarke, C. J. (2014), *Optimal Semicoherent Searches for
  Continuous Gravitational Waves from Spinning Neutron Stars in Binary Systems*,
  ApJ 795, 75 — the photon-by-photon likelihood used here.
- Lange, C. et al. (2001), MNRAS 326, 274 — the ELL1 timing model.
- Foreman-Mackey, D. et al. (2013), PASP 125, 306 — ``emcee``, the sampler.
