"""Bundles of related state passed through the ell1fit pipeline.

The pipeline threads two groups of values through nearly every stage. Passing
them individually produced signatures of a dozen positional arguments, where a
transposed pair would still run and quietly give a wrong answer. Grouping them
makes the wrong call a ``TypeError`` instead of a bad number, and makes it
obvious which stage needs what.

Both are frozen. Nothing in the pipeline mutates them in place: stages that
refine a value build a new bundle with :func:`dataclasses.replace`, so a value
read at one stage cannot be silently changed by another. That matters for
iterative template refinement, which reruns the fit with a rebuilt template and
must not have the previous iteration's state leak into the next.

Note that ``ObservationSet`` holds *parallel lists*, one entry per input event
file, rather than a list of per-file objects. That mirrors how the numerical
code consumes them -- ``_calculate_phases`` and friends iterate over file index
-- and keeps this a description of the existing layout rather than a rewrite of
it.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ObservationSet:
    """Everything read from disk: the data and the models describing it.

    Attributes
    ----------
    files : list of str
        Input event file paths.
    models : list
        PINT timing models, one per file, already re-referenced to their own
        ``PEPOCH``.
    ref_model : object
        A single model re-referenced to the mean ``PEPOCH``, used for the
        orbital parameters shared across files. Its ``TASC`` may therefore
        differ from any individual model's by a whole number of orbits.
    pepoch : list of float
        Reference epoch (MJD) of each file.
    times_from_pepoch : list of np.ndarray
        Event arrival times, seconds from each file's own ``PEPOCH``.
    energies : list of np.ndarray
        Event energies, or PI channels when ``--use-pi`` is in effect.
    exposures : np.ndarray
        Live time per file, in seconds, summed from the GTIs.
    observation_length : np.ndarray
        Wall-clock span per file, in seconds. Used for the uncertainty and
        scaling heuristics, which care about the lever arm rather than the
        exposure.
    """

    files: list
    models: list
    ref_model: object
    pepoch: list
    times_from_pepoch: list
    energies: list
    exposures: object
    observation_length: object

    @property
    def n_files(self):
        """Number of input event files."""
        return len(self.files)


@dataclasses.dataclass(frozen=True)
class FitSetup:
    """Everything needed to evaluate the posterior at a trial position.

    This is the bundle that defines *what is being fitted*: which parameters are
    free, where they start, how they are scaled, what they are compared against,
    and what prior constrains them.

    Attributes
    ----------
    parameter_names : list of str
        Names of the free parameters, in the order used by every local-coordinate
        array throughout the pipeline.
    baseline_values : list of float
        Physical starting value of each free parameter -- the ``initial`` term
        in ``physical = local * factor + initial``.
    logprior_funcs : list of callable
        One log-prior per free parameter, evaluated in physical units. Those
        with hard support carry a ``phys_bounds`` attribute.
    factors : list of float
        Per-parameter scale -- the ``factor`` term in the same relation.
    template_funcs : list of callable
        Pulse template per file, evaluated at a phase to give a probability
        density.
    parameters : dict
        Full parameter mapping, including the fixed values the free parameters
        are varied against.
    likelihood_func : callable
        Statistic evaluated on phases.
    weights : list of np.ndarray or None
        Per-event weights, or ``None`` for an unweighted fit.
    tolerance : float
        Deorbiting convergence tolerance, in seconds.
    """

    parameter_names: list
    baseline_values: list
    logprior_funcs: list
    factors: list
    template_funcs: list
    parameters: dict
    likelihood_func: object
    weights: object = None
    tolerance: float = 1e-8

    @property
    def n_parameters(self):
        """Number of free parameters."""
        return len(self.parameter_names)

    def with_baseline_from(self, parameters):
        """Return a copy whose baseline is re-read from ``parameters``.

        Used after a stage moves a fitted value -- the ``Phase_i`` trace, or a
        refinement pass -- so the local coordinate system is re-centred on the
        updated position instead of drifting away from its origin.
        """
        return dataclasses.replace(
            self,
            parameters=parameters,
            baseline_values=[parameters[par] for par in self.parameter_names],
        )
