r"""Eccentricity and periastron angle from the sampled ``EPS1``/``EPS2`` pair.

ELL1 does not sample the eccentricity. It samples the two Laplace--Lagrange
parameters

.. math::

   \epsilon_1 = e \sin\omega, \qquad \epsilon_2 = e \cos\omega,

which stay well behaved as :math:`e \to 0` -- the regime the model is written
for -- where :math:`\omega` itself is undefined. The eccentricity is their
length, :math:`e = \sqrt{\epsilon_1^2 + \epsilon_2^2}`.

The posterior on :math:`e` follows by evaluating that length on **every
posterior sample**: pushing joint samples through a function gives samples of
that function's posterior, with the ``EPS1``/``EPS2`` correlation carried along
for free. What must never be done is to combine the two published marginals --
``dEPS1_50`` and ``dEPS2_50`` with their error bars -- in quadrature. That
throws the correlation away, and it ignores the positivity bias below.

Why a median is not always the right thing to quote
---------------------------------------------------

A length built from two noisy components cannot come out negative, so it is
biased away from zero. A pure-noise pair with no eccentricity at all still
returns a positive :math:`e`, Rayleigh distributed, peaking at the
per-component uncertainty :math:`\sigma` and with a median of
:math:`1.18\,\sigma`. Quoting that as ``e = 1.18 sigma +- ...`` manufactures a
detection out of nothing.

:func:`eccentricity_summary` therefore asks first whether the *joint* posterior
excludes the origin -- see :func:`zero_eccentricity_exclusion` -- and quotes an
upper limit when it does not. The limit is the 95th percentile of :math:`e`
over the posterior samples, the same convention this package already uses for
``A1DOT``.

What prior this is under
------------------------

``ell1fit`` puts an independent flat prior on each of ``EPS1`` and ``EPS2``
(:func:`ell1fit.priors.assign_logpriors`). Flat over the *plane* is not flat
over the radius: a wider annulus holds more area, so the implied prior on the
eccentricity grows as :math:`p(e) \propto e`, disfavouring small eccentricities
before any data are seen. That is the default here, and for a solid detection
the likelihood swamps it either way.

Passing ``flat_in_e_prior=True`` reweights the samples by :math:`1/e` to undo
it and report the answer under a prior flat in :math:`e` instead. It only
matters when the eccentricity is marginal -- which is exactly when it moves the
upper limit, from :math:`2.45\,\sigma` (Rayleigh) to :math:`1.96\,\sigma`
(half-normal) in the pure-noise case. The reweighting is by construction
noisiest at small :math:`e`, where the weights are largest and the samples
sparsest, so treat it as a cross-check rather than the headline number.
"""

import logging
import os

import numpy as np
from astropy.table import Table
from scipy.special import ndtri_exp

from .mcmc_utils import SAMPLES_SUFFIX
from .plotting import plot_style_context


__all__ = [
    "default_chain_file",
    "eccentricity_and_omega",
    "eccentricity_summary",
    "eccentricity_summary_from_run",
    "eps_samples_from_chain",
    "load_eps_samples",
    "output_root",
    "plot_eccentricity_posterior",
    "zero_eccentricity_exclusion",
]


#: Percentiles reported for the eccentricity, matching those that
#: :func:`ell1fit.mcmc_utils.calculate_result_array_from_samples` stores for
#: every fitted parameter, so the two sets of numbers can sit in one table.
PERCENTILES = (1, 10, 16, 50, 84, 90, 99)

#: Credible level of the quoted upper limit. The 95th percentile of the
#: posterior samples, as for ``A1DOT``.
DEFAULT_UPPER_LIMIT_LEVEL = 0.95

#: Below this equivalent-Gaussian significance the eccentricity is reported as
#: an upper limit rather than a measurement.
DEFAULT_DETECTION_SIGMA = 3.0

#: Tolerance of :func:`_column_for_parameter`, in units of the parameter's own
#: recorded 16--84 width. A chain column belongs to a parameter when its
#: percentiles reproduce the recorded ones to within this fraction. Loose
#: enough to survive a chain that grew since the table was written, far tighter
#: than the gap between any two different parameters.
COLUMN_MATCH_TOLERANCE = 0.25


def eccentricity_and_omega(eps1, eps2):
    r"""Convert ``EPS1``/``EPS2`` samples into eccentricity and periastron angle.

    Parameters
    ----------
    eps1, eps2 : array-like
        Paired posterior samples of :math:`e\sin\omega` and :math:`e\cos\omega`.
        Pairing matters: entry ``i`` of both must come from the same step of the
        chain.

    Returns
    -------
    eccentricity : np.ndarray
        :math:`\sqrt{\epsilon_1^2 + \epsilon_2^2}`, one value per sample.
    omega_deg : np.ndarray
        :math:`\mathrm{atan2}(\epsilon_1, \epsilon_2)` in degrees, wrapped to
        ``[0, 360)``.

    Examples
    --------
    >>> ecc, om = eccentricity_and_omega([0.0, 3e-3], [2e-3, 0.0])
    >>> np.allclose(ecc, [2e-3, 3e-3])
    True
    >>> np.allclose(om, [0.0, 90.0])
    True
    """
    eps1 = np.asarray(eps1, dtype=float)
    eps2 = np.asarray(eps2, dtype=float)
    return np.hypot(eps1, eps2), np.degrees(np.arctan2(eps1, eps2)) % 360.0


def zero_eccentricity_exclusion(eps1, eps2):
    r"""Ask how firmly the joint posterior excludes a circular orbit.

    The question "is the eccentricity significant?" is a question about the
    *plane*, not about either component: ``EPS1`` and ``EPS2`` are usually
    correlated, and each one alone can straddle zero while the pair together
    stays well away from the origin. So the statistic is the Mahalanobis
    distance of the origin from the sample cloud,

    .. math::

       d^2 = \bar{\boldsymbol{\epsilon}}^{\mathsf{T}}\,
             \mathsf{C}^{-1}\,\bar{\boldsymbol{\epsilon}},

    with :math:`\mathsf{C}` the sample covariance. For a Gaussian posterior the
    credible region at level :math:`\alpha` is the ellipse
    :math:`d^2 \le -2\ln(1-\alpha)`, so the origin lies exactly on the
    :math:`1 - e^{-d^2/2}` contour: that number is returned as the credibility
    with which :math:`e = 0` is excluded. ``EPS1`` and ``EPS2`` enter the phase
    model almost linearly at small :math:`e`, which is what makes the Gaussian
    step a fair one; a visibly banana-shaped corner plot is a sign it is not.

    Returns
    -------
    credibility : float
        Credible level at which the origin is excluded, in ``[0, 1)``.
    sigma : float
        The same statement as an equivalent Gaussian significance -- the number
        of standard deviations of a one-dimensional Gaussian carrying the same
        two-sided tail probability. ``3.0`` means "excluded with the
        credibility of a three-sigma result", i.e. 99.73%.
    """
    samples = np.column_stack(
        [np.asarray(eps1, dtype=float).ravel(), np.asarray(eps2, dtype=float).ravel()]
    )
    if samples.shape[0] < 3:
        raise ValueError("At least three samples are needed to estimate a covariance.")

    mean = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False)

    try:
        distance_squared = float(mean @ np.linalg.solve(covariance, mean))
    except np.linalg.LinAlgError:  # a degenerate cloud says nothing about the origin
        return 0.0, 0.0

    if not np.isfinite(distance_squared) or distance_squared <= 0:
        return 0.0, 0.0

    # -expm1(-d^2/2) rather than 1 - exp(...) so that a marginal case keeps its
    # precision, and ndtri_exp works from the log tail so that a very strong
    # detection does not saturate at "inf sigma" when exp(-d^2/2) underflows.
    credibility = float(-np.expm1(-0.5 * distance_squared))
    sigma = float(-ndtri_exp(-0.5 * distance_squared - np.log(2.0)))
    return credibility, sigma


def _weighted_quantile(values, quantiles, weights):
    """Quantiles of a weighted sample, by linear interpolation of the weighted CDF.

    ``np.percentile`` has no weight argument; this is the standard replacement,
    with the CDF evaluated at the midpoint of each sample's weight interval so
    that unit weights reproduce ``np.percentile``'s behaviour closely.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]

    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return np.interp(np.asarray(quantiles, dtype=float), cumulative, values)


def _circular_summary(omega_deg, weights=None):
    """Mean direction, its 68% credible interval, and how concentrated it is.

    An angle cannot be averaged with :func:`numpy.mean`: 359 and 1 degree
    average to 180, the opposite direction. The mean of the unit vectors is
    used instead. Its length ``concentration`` runs from 0 (samples spread
    uniformly around the circle -- no constraint at all, which is what an
    undetected eccentricity looks like) to 1 (all samples on top of each
    other).
    """
    angles = np.radians(np.asarray(omega_deg, dtype=float))
    if weights is None:
        weights = np.ones_like(angles)
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)

    mean_cos = float(np.sum(weights * np.cos(angles)) / total)
    mean_sin = float(np.sum(weights * np.sin(angles)) / total)
    concentration = float(np.hypot(mean_cos, mean_sin))
    mean_angle = np.arctan2(mean_sin, mean_cos)

    # Deviations from the mean direction, wrapped onto (-180, 180], are an
    # ordinary linear quantity, so the interval can be taken on those and put
    # back around the mean.
    deviations = np.degrees((angles - mean_angle + np.pi) % (2 * np.pi) - np.pi)
    low, high = _weighted_quantile(deviations, [0.16, 0.84], weights)
    mean_deg = float(np.degrees(mean_angle) % 360.0)

    return {
        "OM_deg_mean": mean_deg,
        "OM_deg_16": float((mean_deg + low) % 360.0),
        "OM_deg_84": float((mean_deg + high) % 360.0),
        "OM_concentration": concentration,
    }


def eccentricity_summary(
    eps1,
    eps2,
    upper_limit_level=DEFAULT_UPPER_LIMIT_LEVEL,
    detection_sigma=DEFAULT_DETECTION_SIGMA,
    flat_in_e_prior=False,
):
    r"""Summarize the eccentricity posterior, as a value or as an upper limit.

    Parameters
    ----------
    eps1, eps2 : array-like
        Paired posterior samples, in physical units. See
        :func:`load_eps_samples` for getting them out of a finished run.
    upper_limit_level : float
        Credible level of the upper limit. Default 0.95.
    detection_sigma : float
        Equivalent-Gaussian significance (see
        :func:`zero_eccentricity_exclusion`) the exclusion of :math:`e = 0` has
        to reach before a median and error bars are quoted instead of an upper
        limit. Default 3.
    flat_in_e_prior : bool
        Reweight the samples by :math:`1/e` to report the answer under a prior
        flat in :math:`e`, undoing the :math:`p(e) \propto e` implied by the
        flat priors on ``EPS1`` and ``EPS2``. The detection test is always made
        on the samples as drawn -- it asks what the data say about the origin,
        which is not a question the radial prior should be allowed to move.

    Returns
    -------
    dict
        ``ECC_<percentile>`` for each of :data:`PERCENTILES`;
        ``ECC_upper_limit`` and ``ECC_upper_limit_level`` (the limit is ``nan``
        when the eccentricity is detected, since a limit is then not the thing
        to quote); ``ECC_detected``; ``ECC_zero_credibility`` and
        ``ECC_significance_sigma`` from
        :func:`zero_eccentricity_exclusion`; the ``OM_deg_*`` fields of
        :func:`_circular_summary`; ``ECC_nsamples``; ``ECC_prior``; and
        ``ECC_summary``, the one-line form to paste into a paper draft.
    """
    if np.shape(eps1) != np.shape(eps2):
        raise ValueError("eps1 and eps2 must be paired: same number of samples in each.")
    eccentricity, omega_deg = eccentricity_and_omega(eps1, eps2)

    credibility, sigma = zero_eccentricity_exclusion(eps1, eps2)
    detected = bool(sigma >= detection_sigma)

    if flat_in_e_prior:
        # 1/e diverges at the origin; clip at a value far below any sample the
        # posterior actually places there, so one unlucky draw cannot carry the
        # whole weighted CDF.
        floor = 1e-6 * np.median(eccentricity)
        weights = 1.0 / np.clip(eccentricity, floor, None)
        weights /= np.sum(weights)
        quantiles = _weighted_quantile(eccentricity, [p / 100 for p in PERCENTILES], weights)
        limit = float(_weighted_quantile(eccentricity, [upper_limit_level], weights)[0])
    else:
        weights = None
        quantiles = np.percentile(eccentricity, PERCENTILES)
        limit = float(np.percentile(eccentricity, 100 * upper_limit_level))

    results = {f"ECC_{p:g}": float(value) for p, value in zip(PERCENTILES, quantiles)}
    results.update(_circular_summary(omega_deg, weights))
    results["ECC_detected"] = detected
    results["ECC_zero_credibility"] = credibility
    results["ECC_significance_sigma"] = sigma
    results["ECC_upper_limit"] = np.nan if detected else limit
    results["ECC_upper_limit_level"] = float(upper_limit_level)
    results["ECC_nsamples"] = int(eccentricity.size)
    results["ECC_prior"] = "flat in e" if flat_in_e_prior else "flat in the EPS1-EPS2 plane"

    if detected:
        median = results["ECC_50"]
        results["ECC_summary"] = (
            f"e = {median:.3g} (+{results['ECC_84'] - median:.2g} "
            f"-{median - results['ECC_16']:.2g}, 68%), "
            f"omega = {results['OM_deg_mean']:.1f} deg; "
            f"e = 0 excluded at {sigma:.1f} sigma"
        )
    else:
        results["ECC_summary"] = (
            f"e < {limit:.3g} ({100 * upper_limit_level:g}% upper limit); "
            f"e = 0 excluded only at {sigma:.2g} sigma, "
            "so this is a limit and not a measurement"
        )

    return results


def _colnames(results_row):
    """Column names of an astropy ``Row``, or keys of a plain mapping."""
    if hasattr(results_row, "colnames"):
        return list(results_row.colnames)
    return list(results_row.keys())


def _column_for_parameter(flat_chain, results_row, par, tolerance=COLUMN_MATCH_TOLERANCE):
    """Find which column of the chain holds ``par``, by its recorded percentiles.

    The HDF5 chain stores no parameter names, and the result table's column
    order is not guaranteed to survive :func:`ell1fit.results_io.split_output_results`,
    which reorders per-file columns. Rather than trusting either, each column is
    identified by the fingerprint the fit already wrote down for it: its 16th,
    50th and 84th percentiles, in the same local coordinates the chain is in.
    Two different parameters agreeing to within a quarter of their own width is
    not a thing that happens, so the match is unambiguous -- and a failure to
    match is itself worth knowing about, since it means table and chain are not
    from the same fit.
    """
    keys = [f"d{par}_{p:g}" for p in (16, 50, 84)]
    available = _colnames(results_row)
    missing = [key for key in keys if key not in available]
    if missing:
        raise ValueError(
            f"{par} was not a fitted parameter in these results: {missing[0]} is missing. "
            "Refit including EPS1 and EPS2 among the fitted parameters."
        )

    target = np.array([float(results_row[key]) for key in keys])
    spread = target[2] - target[0]
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError(f"Recorded percentiles of {par} are degenerate: {target}.")

    found = np.percentile(flat_chain, [16, 50, 84], axis=0)
    mismatch = np.max(np.abs(found - target[:, None]), axis=0) / spread
    best = int(np.argmin(mismatch))
    if mismatch[best] > tolerance:
        raise ValueError(
            f"No column of the chain matches the recorded percentiles of {par} "
            f"(closest is column {best}, off by {mismatch[best]:.2g} of its own "
            "16-84 width). The chain and the result table are probably from "
            "different fits."
        )
    return best


def _column_from_labels(labels, par, flat_chain, results_row, tolerance=COLUMN_MATCH_TOLERANCE):
    """Index of ``par`` among named columns, with a percentile sanity check.

    Names are authoritative when the sampler wrote them down, so a disagreeing
    fingerprint only warns -- most often it means the chain was extended after
    the table was written, which is harmless.
    """
    candidates = [f"d{par}", par]
    for candidate in candidates:
        if candidate in labels:
            column = labels.index(candidate)
            break
    else:
        raise ValueError(
            f"{par} is not among the sampled parameters {labels}. "
            "Refit including EPS1 and EPS2 among the fitted parameters."
        )

    keys = [f"d{par}_{perc:g}" for perc in (16, 50, 84)]
    if all(key in _colnames(results_row) for key in keys):
        target = np.array([float(results_row[key]) for key in keys])
        spread = target[2] - target[0]
        found = np.percentile(flat_chain[:, column], [16, 50, 84])
        if spread > 0 and np.max(np.abs(found - target)) / spread > tolerance:
            logging.warning(
                f"Samples of {par} disagree with the percentiles recorded for it "
                f"({found} against {target}). Using the samples; check that the "
                "table and the sample file come from the same fit."
            )
    return column


def eps_samples_from_chain(results_row, flat_chain, labels=None):
    r"""Turn a flattened chain into physical ``EPS1``/``EPS2`` samples.

    The sampler works in local coordinates: column ``i`` of the chain holds
    ``dEPS1``, the offset from the starting value in units of that parameter's
    preconditioned scale. The fit records both numbers needed to undo that,
    ``dEPS1_initial`` and ``dEPS1_factor``, so that

    .. math::

       \epsilon_1 = \epsilon_1^{\mathrm{initial}}
                    + \mathrm{d}\epsilon_1 \times \mathrm{factor}.

    Parameters
    ----------
    results_row : astropy.table.Row or dict
        One row of a ``*_results.ecsv`` table.
    flat_chain : np.ndarray
        Flattened chain, shape ``(nsamples, ndim)``, in local coordinates.
    labels : list of str, optional
        Parameter name per column, as saved by
        :func:`ell1fit.mcmc_utils.save_flat_samples`. When absent -- an HDF5
        chain carries no names -- the columns are identified by their recorded
        percentiles instead.

    Returns
    -------
    eps1, eps2 : np.ndarray
        Physical posterior samples, paired sample by sample.
    """
    flat_chain = np.atleast_2d(np.asarray(flat_chain, dtype=float))
    samples = []
    for par in ("EPS1", "EPS2"):
        if labels is None:
            column = _column_for_parameter(flat_chain, results_row, par)
        else:
            column = _column_from_labels(list(labels), par, flat_chain, results_row)
        initial = float(results_row[f"d{par}_initial"])
        factor = float(results_row[f"d{par}_factor"])
        samples.append(initial + flat_chain[:, column] * factor)
    return samples[0], samples[1]


RESULTS_SUFFIX = "_results.ecsv"


def output_root(results_file):
    """The output root a ``*_results.ecsv`` file was built from."""
    if not str(results_file).endswith(RESULTS_SUFFIX):
        raise ValueError(
            f"Cannot guess the output root of {results_file!r}: it does not end in "
            f"{RESULTS_SUFFIX!r}. Pass the sample file explicitly."
        )
    return str(results_file)[: -len(RESULTS_SUFFIX)]


def default_chain_file(results_file):
    """Sample file that goes with a ``*_results.ecsv`` table.

    Everything a run writes is built from one output root, so the samples sit
    next to the table. Two files can hold them: ``<root>_samples.npz``, which
    every sampler writes and which carries the parameter names, and
    ``<root>.h5``, the emcee backend, which is all that older runs left behind.
    The first is preferred, the second is the fallback.
    """
    root = output_root(results_file)
    samples_file = root + SAMPLES_SUFFIX
    if os.path.exists(samples_file):
        return samples_file
    return root + ".h5"


def load_eps_samples(results_file, chain_file=None, row=-1):
    """Read ``EPS1``/``EPS2`` posterior samples from a finished ``ell1fit`` run.

    Parameters
    ----------
    results_file : str
        Path to the ``*_results.ecsv`` written by the fit. Beware which one:
        the chain sits next to the *aggregate* output root, which for a
        single-file run is the event file's own root, and for a multi-file run
        is the combined ``-o`` root.
    chain_file : str, optional
        Path to the emcee HDF5 chain. Defaults to :func:`default_chain_file`.
    row : int
        Which row of the table to use. ``ell1fit`` appends a row per run
        (:func:`ell1fit.results_io.safe_save`), so the default is the last, the
        most recent fit.

    Returns
    -------
    eps1, eps2 : np.ndarray
        Physical posterior samples, with the same burn-in and thinning the fit
        itself applied.
    """
    if chain_file is None:
        chain_file = default_chain_file(results_file)
    if not os.path.exists(chain_file):
        raise FileNotFoundError(
            f"No samples beside {results_file}: expected {chain_file}. A fit run "
            "before samples were saved leaves only its emcee backend; pass that "
            ".h5 as chain_file."
        )

    table = Table.read(results_file)
    results_row = table[row]

    if str(chain_file).endswith(SAMPLES_SUFFIX):
        from .mcmc_utils import load_flat_samples

        flat_chain, labels = load_flat_samples(chain_file)
    else:
        import emcee

        from .mcmc_utils import get_flat_samples

        reader = emcee.backends.HDFBackend(chain_file, read_only=True)
        flat_chain, _ = get_flat_samples(reader)
        labels = None

    logging.info(f"Read {flat_chain.shape[0]} samples from {chain_file}")

    return eps_samples_from_chain(results_row, flat_chain, labels=labels)


def eccentricity_summary_from_run(results_file, chain_file=None, row=-1, **kwargs):
    """Load a finished run and summarize its eccentricity in one call.

    Keyword arguments are passed to :func:`eccentricity_summary`.
    """
    eps1, eps2 = load_eps_samples(results_file, chain_file=chain_file, row=row)
    return eccentricity_summary(eps1, eps2, **kwargs)


def plot_eccentricity_posterior(eps1, eps2, fname="eccentricity.jpg", summary=None, bins=80):
    """Plot the eccentricity posterior, marking either the interval or the limit.

    Parameters
    ----------
    eps1, eps2 : array-like
        Paired posterior samples, in physical units.
    fname : str
        Output image path.
    summary : dict, optional
        Output of :func:`eccentricity_summary`; recomputed with the defaults if
        not given.
    bins : int
        Histogram bins.

    Returns
    -------
    str
        ``fname``, for convenience.
    """
    import matplotlib.pyplot as plt

    if summary is None:
        summary = eccentricity_summary(eps1, eps2)
    eccentricity, _ = eccentricity_and_omega(eps1, eps2)

    with plot_style_context():
        fig, ax = plt.subplots()
        ax.hist(eccentricity, bins=bins, histtype="stepfilled", color="grey", alpha=0.4)
        if summary["ECC_detected"]:
            ax.axvline(summary["ECC_50"], color="k", label="median")
            ax.axvspan(
                summary["ECC_16"], summary["ECC_84"], color="k", alpha=0.12, label="68% interval"
            )
        else:
            ax.axvline(
                summary["ECC_upper_limit"],
                color="k",
                ls="--",
                label=f"{100 * summary['ECC_upper_limit_level']:g}% upper limit",
            )
        ax.set_xlabel("Eccentricity")
        ax.set_ylabel("Posterior samples")
        ax.set_xlim(0, None)
        ax.legend(loc="upper right")
        # The summary line is too long for one 3.5-inch title: break it at the
        # semicolons and give the top margin back the room it needs.
        ax.set_title(summary["ECC_summary"].replace("; ", "\n"), fontsize=5)
        fig.subplots_adjust(top=0.88)
        fig.savefig(fname, dpi=300)
        plt.close(fig)

    return fname
