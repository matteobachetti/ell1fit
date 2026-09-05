Eccentricity: combining the ``EPS1`` and ``EPS2`` posteriors
============================================================

ELL1 never samples the eccentricity. It samples

.. math::

   \epsilon_1 = e \sin\omega, \qquad \epsilon_2 = e \cos\omega,

because those two stay well behaved as :math:`e \to 0`, where the periastron
angle :math:`\omega` is undefined and a fit in :math:`(e, \omega)` would be
trying to measure the direction of a zero-length vector. The eccentricity is
the length of that vector, :math:`e = \sqrt{\epsilon_1^2 + \epsilon_2^2}`.

:mod:`ell1fit.eccentricity` turns a finished fit into a posterior on :math:`e`.

.. contents::
   :local:
   :depth: 1

The rule
--------

**Transform the samples, not the summaries.** Each step of the chain carries an
``EPS1`` and an ``EPS2`` that belong together; evaluating the length on every
step gives samples from the posterior of :math:`e`, correlations and all. This
is exact, not an approximation — pushing joint posterior samples through a
function always yields samples of that function's posterior.

**Never add the published error bars in quadrature.** ``dEPS1_50`` and
``dEPS2_50`` with their intervals are *marginal* summaries: combining them
throws away the correlation between the two, and the correlation is usually
large. The test suite carries a case in which each component sits a mere 1.5
sigma from zero while the joint posterior excludes a circular orbit at more
than ten — the quadrature recipe reports no eccentricity where there is a
solid one.

From a finished run
-------------------

Every run writes its posterior samples to ``<outroot>_samples.npz``, whichever
sampler produced them, alongside the summary table ``<outroot>_results.ecsv``.
Both are in the sampler's local coordinates: column ``i`` of the chain is
``dEPS1``, an offset from the starting value in units of that parameter's
preconditioned scale. The table records ``dEPS1_initial`` and ``dEPS1_factor``,
which is all that is needed to put it back into physical units, and
:func:`~ell1fit.eccentricity.load_eps_samples` does that::

    from ell1fit.eccentricity import eccentricity_summary_from_run

    summary = eccentricity_summary_from_run("campaign_A1_EPS1_EPS2_results.ecsv")
    print(summary["ECC_summary"])

Pass the results table that sits beside the samples: for a single-file run that
is the *event file's* output root, and for a multi-file run the combined ``-o``
root. The sample file names its columns, so nothing has to be guessed. A run
made before those files existed leaves only the emcee backend ``<outroot>.h5``,
which carries no names; that is still read, with each column identified by the
percentiles the table already records for it rather than by column order, so a
reordered or split table cannot silently mismatch the two — and a table and
chain from different fits raise instead of returning nonsense.

The orbit summary figure
------------------------

Whenever a run has an eccentricity to show — that is, whenever both ``EPS1``
and ``EPS2`` were fitted — it also writes ``<outroot>_orbit.jpg``, which puts
the orbit and its eccentricity side by side: on the left a corner plot of
whichever of ``A1``, ``PB``, ``TASC``, ``EPS1`` and ``EPS2`` the chain actually
explored, on the right the eccentricity posterior described above. The pipeline
writes it at the end of a fit and ``ell1ecc`` writes it from a finished run;
``--orbit-plot`` renames it.

This is not the same picture as the main ``<outroot>_corner.jpg``. That one is
drawn in the sampler's *local* coordinates — offsets from the starting
solution, in units of each parameter's preconditioned scale — which is the
right frame for asking whether the chain moved and a useless one for reading a
number off. Here each parameter is in the units it is quoted in.

Getting that readable takes some care, because an orbital posterior is a spike
in an awkward place: ``A1`` might be 22 light-seconds known to a millionth of
one, ``TASC`` an MJD known to a few microseconds. Plotted raw, every panel is a
vertical line at an uninformative tick label. So each axis is drawn as an
offset from the posterior **mean**, in a unit chosen so that one standard
deviation is a number between 1 and 1000 — hours or minutes for a loosely known
orbital period, microseconds for a sharply known epoch — with the subtracted
mean printed in the axis label to enough significant digits that adding it back
to a residual read off the axis lands where the posterior actually is.

The units a parameter is *stored* in are not the units it is *read* in, and the
figure keeps the two apart. ``PB`` lives in seconds inside the fit but every
parfile quotes it in days, so the label carries days and the residuals carry
milliseconds or seconds. ``TASC`` lives in days, as an MJD, but nobody quotes
an epoch uncertainty in days, so its residuals are in seconds. ``A1`` keeps
light-seconds with a metric prefix. ``EPS1`` and ``EPS2`` are dimensionless and
get a bare power of ten. :data:`ell1fit.orbit_plot.CONVENTIONS` records this,
one entry per parameter, and anything absent from it falls back to the
dimensionless treatment.

Measurement or upper limit
--------------------------

A length built from two noisy components cannot be negative, so it is biased
away from zero. Pure noise with no eccentricity at all still returns a positive
:math:`e`, Rayleigh distributed, peaking at the per-component uncertainty
:math:`\sigma` and with a median of :math:`1.18\,\sigma`. Quote that as a
measurement and the paper claims a detection of nothing.

So :func:`~ell1fit.eccentricity.eccentricity_summary` decides first, on the
*joint* posterior, whether the origin is excluded — the Mahalanobis distance of
:math:`(0,0)` from the sample cloud, which for a Gaussian posterior is exactly
the credible contour the origin lies on — and only then chooses what to print.
Below three sigma equivalent it reports **the 95th percentile of** :math:`e`,
the same upper-limit convention this package uses for ``A1DOT``
(:doc:`orbital_derivatives`).

The two branches, on samples with a known answer:

.. doctest::

    >>> import numpy as np
    >>> from ell1fit.eccentricity import eccentricity_summary
    >>> rng = np.random.default_rng(20260903)
    >>> sigma = 1e-4

A circular orbit observed with that precision. The limit lands on the Rayleigh
95th percentile, :math:`\sqrt{-2\ln 0.05}\,\sigma = 2.45\,\sigma`:

.. doctest::

    >>> noise = rng.normal(0, sigma, (2, 200000))
    >>> summary = eccentricity_summary(noise[0], noise[1])
    >>> summary["ECC_detected"]
    False
    >>> bool(np.isclose(summary["ECC_upper_limit"] / sigma, 2.4477, rtol=0.02))
    True

The same data, twenty sigma of eccentricity added:

.. doctest::

    >>> omega = np.radians(71.0)
    >>> eps1 = 20 * sigma * np.sin(omega) + noise[0]
    >>> eps2 = 20 * sigma * np.cos(omega) + noise[1]
    >>> summary = eccentricity_summary(eps1, eps2)
    >>> summary["ECC_detected"]
    True
    >>> bool(np.isclose(summary["ECC_50"] / sigma, 20.0, rtol=0.01))
    True
    >>> bool(np.isclose(summary["OM_deg_mean"], 71.0, atol=0.5))
    True

``ECC_summary`` holds the one-line form of whichever branch was taken.

The periastron angle
--------------------

:math:`\omega = \mathrm{atan2}(\epsilon_1, \epsilon_2)` is an angle, so it is
summarized with circular statistics: 359 and 1 degree average to zero, not to
180. ``OM_concentration`` is the length of the mean unit vector, running from 0
(samples spread uniformly around the circle) to 1 (all in one place). When the
eccentricity is only a limit, this number sits near zero and the reported angle
means nothing — as it should, since a circular orbit has no periastron.

What prior the answer is under
------------------------------

``ell1fit`` puts an independent flat prior on ``EPS1`` and on ``EPS2``. Flat
over the *plane* is not flat over the radius: a wider annulus holds more area,
so the implied prior on the eccentricity grows as :math:`p(e) \propto e`, and
small eccentricities are disfavoured before any data arrive. That is the
default, and it is the defensible one — the components, not the radius, are
what the model is parameterised in.

Passing ``flat_in_e_prior=True`` reweights the samples by :math:`1/e` to report
the answer under a prior flat in :math:`e` instead. It changes nothing for a
solid detection, and for pure noise it moves the limit from the Rayleigh
:math:`2.45\,\sigma` to the half-normal :math:`1.96\,\sigma` — a 20% tighter
number obtained purely by changing the prior, which is worth knowing before
quoting either. The weights are largest exactly where samples are sparsest, so
it is a cross-check rather than the headline result. The detection test itself
is never reweighted: whether the data exclude a circular orbit is not a
question the radial prior should be allowed to answer.
