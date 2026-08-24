"""Unit-cube prior transforms for the log-priors in :mod:`ell1fit.priors`.

Nested sampling does not evaluate a prior; it *samples* from one, by mapping a
point of the unit cube through the prior's inverse CDF. So the evidence needs
something the rest of the package has never had to provide: a transform
``u -> parameter value``, one per free parameter, matching the log-prior that
:func:`ell1fit.priors.assign_logpriors` assigned.

Two consequences worth stating plainly, because they decide whether a Bayes
factor computed on top of this means anything.

**The transform, not the log-prior, defines the prior that** ``log Z`` **is an
integral against.** ``_flat_logprior`` returns ``0`` inside its bounds rather
than ``-log(width)``: the package's priors are a mixture of normalised and
unnormalised factors, which is harmless for an MCMC that only ever looks at
differences and fatal for an evidence. A transform is normalised by
construction, so building one and *verifying it reproduces the log-prior's shape
and support* replaces the missing normalisation instead of inheriting the
inconsistency. The check is therefore for proportionality, not equality --
:func:`check_transform`.

**An improper prior has no evidence**, so an unbounded ``_flat_logprior`` raises
here rather than being silently given some arbitrary wide box. That box would
set the Occam factor, and hence the answer.
"""

import numpy as np
from scipy.special import ndtri, ndtr


#: Values of ``u`` at which a transform is checked against its log-prior.
#: Deliberately avoids exactly 0 and 1: a Gaussian's inverse CDF is infinite
#: there, which is correct and not a defect worth failing on.
CHECK_GRID = np.linspace(1e-4, 1 - 1e-4, 401)


def _prior_parameters(func):
    """Read a prior closure's constants by name.

    The same trick :mod:`jax_posterior` uses, and carrying the same caveat: it
    is only as stable as the free-variable names in :mod:`ell1fit.priors`,
    which is why every transform built here is then checked numerically against
    the prior it claims to invert.
    """
    cells = func.__closure__ or ()
    return dict(zip(func.__code__.co_freevars, (cell.cell_contents for cell in cells)))


#: One transform per prior shape, as data rather than as a closure: ``(kind,
#: centre, scale, lo, hi)``. ``uniform`` maps ``u`` onto ``[centre, scale]``;
#: ``normal`` maps it through the inverse normal CDF restricted to the CDF range
#: ``[lo, hi]``, which is the whole line for an ordinary Gaussian prior and one
#: orbital period for the wrapped one on ``TASC``.
#:
#: Data rather than closures because a closure cannot be pickled, and dynesty
#: sends the prior transform into every worker process whether or not it is told
#: to evaluate it there.
UNIFORM, NORMAL = 0, 1


def transform_spec_for_prior(func):
    """Describe one log-prior as a transform spec; see :data:`UNIFORM`.

    Raises for a prior shape not implemented here, and for an improper one,
    rather than guessing a box whose width would set the Occam factor and hence
    the answer.
    """
    qualname = getattr(func, "__qualname__", "")
    constants = _prior_parameters(func)

    if qualname.startswith("_flat_logprior"):
        low, high = float(constants["bound0"]), float(constants["bound1"])
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError(
                "A uniform prior with infinite bounds is improper and has no "
                f"evidence: bounds ({low}, {high}). Give the parameter a finite "
                "prior in ell1fit.priors before asking for log Z."
            )
        return (UNIFORM, low, high, 0.0, 1.0)

    if qualname.startswith("_periodic_uniform_logprior"):
        # ``half_width`` is the support, which is not always the full period;
        # ``phys_bounds`` reports the period and would be the wrong box here.
        centre = float(constants["center"])
        half_width = float(constants["half_width"])
        return (UNIFORM, centre - half_width, centre + half_width, 0.0, 1.0)

    if qualname.startswith("_periodic_normal_logprior"):
        centre = float(constants["center"])
        sigma = float(constants["sigma"])
        period = float(constants["period"])
        lo = float(ndtr(-0.5 * period / sigma))
        hi = float(ndtr(0.5 * period / sigma))
        if not hi > lo:
            raise ValueError(f"Degenerate wrapped normal: sigma={sigma}, period={period}")
        # Restricting to one period rather than summing images: the same
        # distribution whenever the period is many sigma wide, and unlike a plain
        # normal transform it can never land outside the branch the prior covers.
        return (NORMAL, centre, sigma, lo, hi)

    if hasattr(func, "__self__") and hasattr(func.__self__, "kwds"):
        # A frozen ``scipy.stats.norm``; ``assign_logpriors`` builds these with
        # keyword arguments only.
        frozen = func.__self__
        return (
            NORMAL,
            float(frozen.kwds.get("loc", 0.0)),
            float(frozen.kwds.get("scale", 1.0)),
            0.0,
            1.0,
        )

    raise NotImplementedError(
        f"No unit-cube transform for log-prior {qualname!r}. Add one here "
        "rather than letting a nested sampler run against a prior it does "
        "not implement."
    )


def apply_spec(spec, u):
    """Map ``u`` in ``(0, 1)`` to a physical parameter value under ``spec``."""
    kind, centre, scale, lo, hi = spec
    if kind == UNIFORM:
        return centre + u * (scale - centre)
    return centre + scale * ndtri(lo + u * (hi - lo))


def transform_for_prior(func):
    """Build the unit-cube transform for one log-prior, as a callable."""
    spec = transform_spec_for_prior(func)
    return lambda u: apply_spec(spec, u)


class PriorTransform:
    """The whole unit-cube transform for a fit, as a picklable callable.

    Maps a unit cube point to a position in the *local* coordinates every other
    sampler in the harness uses, so a nested sampler's draws are directly
    comparable with an ensemble's with no second convention to keep straight.
    """

    def __init__(self, specs, factors, baseline):
        self.kinds = np.array([spec[0] for spec in specs], dtype=int)
        self.centre = np.array([spec[1] for spec in specs], dtype=float)
        self.scale = np.array([spec[2] for spec in specs], dtype=float)
        self.lo = np.array([spec[3] for spec in specs], dtype=float)
        self.hi = np.array([spec[4] for spec in specs], dtype=float)
        self.factors = np.asarray(factors, dtype=float)
        self.baseline = np.asarray(baseline, dtype=float)
        self.uniform = self.kinds == UNIFORM

    def __call__(self, unit_cube):
        u = np.asarray(unit_cube, dtype=float)
        physical = np.where(
            self.uniform,
            self.centre + u * (self.scale - self.centre),
            self.centre + self.scale * ndtri(self.lo + u * (self.hi - self.lo)),
        )
        return (physical - self.baseline) / self.factors


def check_transform(func, transform, grid=None, tolerance=1e-8):
    """Verify a transform inverts its log-prior, and return the log normalisation.

    An inverse CDF maps equal intervals of ``u`` onto equal amounts of prior
    probability. So the test is: split the unit interval into equal pieces, push
    them through the transform, integrate ``exp(logprior)`` over each resulting
    parameter interval, and require every one of those masses to come out the
    same. It calls the package's own prior function throughout, which is the
    point -- a transform built from constants misread out of a closure has to
    fail here.

    The obvious alternative, differencing ``T`` to get its density directly,
    does not work and was tried first: the inverse CDF of a Gaussian is steep
    enough in the tails that a central difference on any workable grid reports
    its own truncation error, which on ``TASC`` came to 0.6 nats of a quantity
    that should have been constant. Quadrature over an interval is stable where
    differentiation across it is not.

    Returns the log of the prior's normalising constant --- what the package's
    unnormalised uniforms leave out. It is returned rather than applied: see
    :func:`build_prior_transform`.
    """
    grid = CHECK_GRID if grid is None else grid
    values = np.array([float(transform(u)) for u in grid])
    if not np.all(np.isfinite(values)):
        raise AssertionError(f"Transform for {func!r} left the real line on the check grid")
    if not np.all(np.diff(values) > 0):
        raise AssertionError(f"Transform for {func!r} is not increasing; it is not an inverse CDF")

    logp = np.array([float(func(v)) for v in values])
    if not np.all(np.isfinite(logp)):
        outside = int(np.sum(~np.isfinite(logp)))
        raise AssertionError(
            f"Transform for {func!r} put {outside}/{len(grid)} samples outside the "
            "prior's support: the two disagree about where the prior lives"
        )

    offset = float(np.max(logp))
    nodes, weights = np.polynomial.legendre.leggauss(8)
    masses = np.empty(len(values) - 1)
    for k in range(len(values) - 1):
        low, high = values[k], values[k + 1]
        half = 0.5 * (high - low)
        points = 0.5 * (low + high) + half * nodes
        density = np.exp(np.array([float(func(x)) for x in points]) - offset)
        masses[k] = half * float(np.dot(weights, density))

    if not np.all(masses > 0):
        raise AssertionError(f"Transform for {func!r} produced an interval of zero prior mass")

    # How finely float64 can even represent this parameter, relative to the
    # width of one check interval. ``TASC`` is a prior 1e-6 days wide centred on
    # an MJD near 5.7e4, where one ulp is 7.3e-12 days: consecutive check points
    # are then a few hundred ulps apart and their spacing is quantised at the
    # 1e-3 level. That is a property of representing an epoch as an absolute MJD
    # -- the same reason arrival times are kept relative to each file's PEPOCH --
    # and no transform can do better, so the check must not claim it as an
    # error. Every other prior here is many orders of magnitude clear of it.
    widths = np.diff(values)
    quantisation = float(np.max(np.spacing(np.abs(values[:-1])) / widths))
    floor = 8.0 * quantisation

    spread = float(np.max(masses) / np.min(masses) - 1.0)
    if spread > max(tolerance, floor):
        worst = int(np.argmax(np.abs(masses - np.median(masses))))
        raise AssertionError(
            f"Transform for {func!r} does not invert it: equal steps in u carry "
            f"prior masses differing by {spread:.3e}, worst near u={grid[worst]:.4f} "
            f"(float64 resolution here allows {floor:.3e}). "
            "Equal steps of an inverse CDF must carry equal probability."
        )

    # ``grid`` deliberately stops short of 0 and 1, so the integral above covers
    # ``grid[-1] - grid[0]`` of the prior rather than all of it.
    covered = float(grid[-1] - grid[0])
    return float(np.log(masses.sum() / covered) + offset)


def build_prior_transform(setup, check=True):
    """Build the whole unit-cube transform for a :class:`~ell1fit.setup_types.FitSetup`.

    Returns ``(transform, log_normalisation)``. The transform maps a unit cube
    point to a position in the *local* coordinates every other sampler in the
    harness uses, so a nested sampler's output is directly comparable with an
    ensemble's without a second convention to keep straight.

    ``log_normalisation`` is **not a correction to apply**. The transform is a
    normalised prior by construction, so the evidence integrated against it is
    already the evidence under a proper prior; this number says only how far
    ``ell1fit.priors``' log-prior *function* is from being a density, which for
    two ``EPS`` parameters uniform on ``(-1, 1)`` is ``2 log 2``. It is returned
    so that gap is visible rather than silent -- subtracting it from ``log Z``
    would introduce the error it describes.
    """
    specs = []
    log_norm = 0.0

    for name, func in zip(setup.parameter_names, setup.logprior_funcs):
        try:
            spec = transform_spec_for_prior(func)
            if check:
                log_norm += check_transform(func, lambda u, spec=spec: apply_spec(spec, u))
        except (ValueError, NotImplementedError, AssertionError) as error:
            raise type(error)(f"{name}: {error}") from error
        specs.append(spec)

    return PriorTransform(specs, setup.factors, setup.baseline_values), log_norm
