r"""Which observation actually measures the orbit, and how much of it survives.

A multi-file ``ell1fit`` run has an unusual shape. One binary is shared by
every observation -- a single ``A1``, ``EPS1``, ``EPS2`` -- while each
observation carries its own spin parameters, ``Phase_i``, ``F0_i``, ``F1_i``
and whatever higher derivatives that segment needs. Nothing in the spin block
of one segment touches any other. That is what this module exploits.

Why the local spin fit is the whole problem
-------------------------------------------

The projected semi-major axis enters the arrival times through a term
:math:`A_1 \sin\Phi`, with :math:`\Phi` the orbital phase. Over a stretch of
data much shorter than one orbit, :math:`\sin\Phi` is very nearly a quadratic
in time -- and a quadratic in time is *exactly* what ``Phase_i``, ``F0_i`` and
``F1_i`` are free to absorb. So a short segment can be rich in photons and
still say almost nothing about ``A1``: the local spin fit quietly eats the
signature before it reaches the residuals. The part that survives is the
cubic-and-higher structure, which is why orbital *coverage* matters far more
than exposure.

The same argument run at :math:`2\Phi` -- where the eccentricity lives, through
:math:`\epsilon_1` and :math:`\epsilon_2` -- gives a different answer, because
a wave of half the period departs from a quadratic much sooner. A segment can
therefore be a poor measurement of ``A1`` and a decent one of the
eccentricity, or the reverse. Guessing which is which from exposure times does
not work; this module computes it.

The decomposition
-----------------

Write the observed information (minus the Hessian of the log likelihood at the
best fit) of one segment over its shared and nuisance parameters as

.. math::

   J_i = \begin{pmatrix} A_i & B_i \\ B_i^{T} & D_i \end{pmatrix},

with :math:`A_i` the shared block, :math:`D_i` that segment's own spin block
and :math:`B_i` the coupling between them. Profiling the spin parameters away
leaves the Schur complement

.. math::

   S_i = A_i - B_i D_i^{-1} B_i^{T},

which :func:`schur_information` returns as ``effective``, alongside the
undiluted :math:`A_i` as ``raw``. Their ratio, :attr:`SegmentInformation.dilution`,
is the fraction of that segment's naive information that outlives its own spin
fit -- the number the paragraphs above are about.

Because no nuisance block is shared, assembling every segment into one big
matrix gives an *arrow*, and eliminating all the spin parameters at once gives

.. math::

   I = \sum_i A_i - \sum_i B_i D_i^{-1} B_i^{T} = \sum_i S_i.

The sum is exact, not an approximation: each segment's contribution can be
computed on its own and added. That is what makes the budget cheap -- one small
Hessian per observation instead of one enormous one -- and it is what
``test_total_equals_the_full_arrow_matrix_schur_complement`` checks.

Two different questions about "how much does this segment matter"
-----------------------------------------------------------------

:meth:`InformationBudget.ranking` shares out the *diagonal* information on one
parameter, so the fractions add to one and the table reads like a budget.
:meth:`InformationBudget.without` instead drops a segment and re-inverts, which
is the honest answer to "what happens to my error bar if I throw this away".
The two agree when the shared parameters are uncorrelated and diverge when they
are not -- and when they diverge, the second one is the one to quote.

What this module does not do
----------------------------

It is a curvature calculation at one point. It says what the data *can*
constrain under the model it is handed, and it is silent about whether that
model is right: unmodelled torque noise inside a segment does not show up here,
and neither does the systematic that appears when the same photons are blocked
two different ways. Those are measured by refitting, not by inverting a matrix.
"""

import dataclasses

import numpy as np


__all__ = [
    "InformationBudget",
    "SegmentInformation",
    "schur_information",
]


def _as_symmetric(matrix, name):
    """Validate a square symmetric matrix and return it as a float array."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Information matrix for segment {name!r} must be square, got shape {matrix.shape}"
        )
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-12):
        raise ValueError(
            f"Information matrix for segment {name!r} is not symmetric. A Hessian that "
            "comes out lopsided usually means the finite-difference step was too small "
            "for the likelihood's numerical noise."
        )
    return matrix


@dataclasses.dataclass(frozen=True, repr=False)
class SegmentInformation:
    """One observation's information about the globally shared parameters.

    Attributes
    ----------
    name : str
        Identifier for the segment, used in tables and error messages.
    raw : np.ndarray
        The shared block :math:`A_i` on its own: what this segment would say
        about the shared parameters if its spin parameters were known exactly.
    effective : np.ndarray
        The Schur complement :math:`A_i - B_i D_i^{-1} B_i^{T}`: what survives
        once the segment's own spin parameters have been fitted away. This is
        the quantity that adds across segments.
    coupling : np.ndarray
        The off-diagonal block :math:`B_i`, kept so the full arrow matrix can be
        reassembled.
    nuisance : np.ndarray
        The segment's own spin block :math:`D_i`.
    """

    name: str
    raw: np.ndarray
    effective: np.ndarray
    coupling: np.ndarray
    nuisance: np.ndarray

    @property
    def n_shared(self):
        """Number of globally shared parameters."""
        return self.raw.shape[0]

    @property
    def n_nuisance(self):
        """Number of per-segment spin parameters profiled away."""
        return self.nuisance.shape[0]

    @property
    def dilution(self):
        """Fraction of each shared parameter's information surviving the local fit.

        One means the segment's own spin parameters take nothing; a number near
        zero means the local fit can very nearly reproduce the orbital
        signature by itself, so the segment is close to useless for that
        parameter however many photons it holds.
        """
        raw = np.diag(self.raw)
        effective = np.diag(self.effective)
        return np.divide(effective, raw, out=np.zeros_like(effective), where=raw != 0.0)

    def __repr__(self):
        return (
            f"<SegmentInformation {self.name!r}: "
            f"{self.n_shared} shared, {self.n_nuisance} nuisance>"
        )


def schur_information(joint, n_shared, name):
    """Split one segment's joint information matrix into raw and effective parts.

    Parameters
    ----------
    joint : array-like
        The observed information (minus the log-likelihood Hessian) over the
        shared parameters first, then that segment's own spin parameters.
    n_shared : int
        How many leading rows and columns are the shared parameters.
    name : str
        Identifier for the segment, carried through into errors and tables.

    Returns
    -------
    SegmentInformation

    Raises
    ------
    numpy.linalg.LinAlgError
        If the segment's own spin block is singular, i.e. the data cannot pin
        down its own spin parameters. The message names the segment.
    """
    joint = _as_symmetric(joint, name)
    if not 0 < n_shared <= joint.shape[0]:
        raise ValueError(
            f"Segment {name!r}: n_shared must be between 1 and {joint.shape[0]}, got {n_shared}"
        )

    raw = joint[:n_shared, :n_shared]
    coupling = joint[:n_shared, n_shared:]
    nuisance = joint[n_shared:, n_shared:]

    if nuisance.shape[0] == 0:
        return SegmentInformation(name, raw, raw.copy(), coupling, nuisance)

    try:
        effective = raw - coupling @ np.linalg.solve(nuisance, coupling.T)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Segment {name!r} cannot determine its own spin parameters: the "
            f"{nuisance.shape[0]}x{nuisance.shape[0]} local block is singular. Drop a spin "
            "derivative for this segment, or drop the segment."
        ) from exc

    # Symmetrise: the subtraction is symmetric in exact arithmetic, and letting
    # the last bits drift makes downstream eigen-decompositions complain.
    effective = 0.5 * (effective + effective.T)
    return SegmentInformation(name, raw, effective, coupling, nuisance)


@dataclasses.dataclass(frozen=True)
class InformationBudget:
    """Every segment's contribution to one set of shared parameters.

    Attributes
    ----------
    parameter_names : list of str
        The shared parameters, in the order they index every matrix here.
    segments : list of SegmentInformation
        One entry per observation.
    """

    parameter_names: list
    segments: list

    def __post_init__(self):
        n_shared = len(self.parameter_names)
        seen = set()
        for segment in self.segments:
            if segment.n_shared != n_shared:
                raise ValueError(
                    f"Segment {segment.name!r} carries {segment.n_shared} shared parameters, "
                    f"but {n_shared} names were given: {list(self.parameter_names)}"
                )
            if segment.name in seen:
                raise ValueError(f"Segment {segment.name!r} appears twice in the budget")
            seen.add(segment.name)

    @property
    def total(self):
        """Summed effective information: the precision matrix of the shared parameters."""
        return np.sum([segment.effective for segment in self.segments], axis=0)

    def _index(self, parameter):
        try:
            return list(self.parameter_names).index(parameter)
        except ValueError:
            raise ValueError(
                f"Unknown parameter {parameter!r}; the budget covers {list(self.parameter_names)}"
            ) from None

    def covariance(self, fixed=()):
        """Covariance of the shared parameters, optionally holding some of them fixed.

        Holding a parameter fixed removes its row and column *before* the
        inversion, which is the difference between "we did not fit it" and "we
        fitted it and then ignored it".
        """
        free = [name for name in self.parameter_names if name not in set(fixed)]
        for name in fixed:
            self._index(name)
        keep = [self._index(name) for name in free]
        return free, np.linalg.inv(self.total[np.ix_(keep, keep)])

    def uncertainties(self, fixed=()):
        """Predicted 1-sigma uncertainty per shared parameter, as a dictionary."""
        free, covariance = self.covariance(fixed=fixed)
        return dict(zip(free, np.sqrt(np.diag(covariance))))

    def correlation(self, fixed=()):
        """Correlation matrix of the free shared parameters."""
        _, covariance = self.covariance(fixed=fixed)
        sigma = np.sqrt(np.diag(covariance))
        return covariance / np.outer(sigma, sigma)

    def without(self, name):
        """A budget with one segment removed, for leave-one-out questions."""
        remaining = [segment for segment in self.segments if segment.name != name]
        if len(remaining) == len(self.segments):
            raise ValueError(
                f"No segment named {name!r}; the budget holds "
                f"{[segment.name for segment in self.segments]}"
            )
        return dataclasses.replace(self, segments=remaining)

    def ranking(self, parameter):
        """Each segment's share of the diagonal information on one parameter.

        Returns ``(name, fraction)`` pairs, largest first, summing to one. This
        is a budget, not a sensitivity: for what actually happens to the error
        bar when a segment goes away, use :meth:`without`.
        """
        index = self._index(parameter)
        contributions = np.array([segment.effective[index, index] for segment in self.segments])
        fractions = contributions / contributions.sum()
        order = np.argsort(fractions)[::-1]
        return [(self.segments[i].name, float(fractions[i])) for i in order]

    def cumulative(self, parameter):
        """Names in ranked order, and the running sum of their information shares."""
        ranking = self.ranking(parameter)
        names = [name for name, _ in ranking]
        return names, np.cumsum([fraction for _, fraction in ranking])

    def n_segments_for(self, parameter, fraction):
        """How many of the best segments carry this fraction of the information."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
        _, cumulative = self.cumulative(parameter)
        return int(np.searchsorted(cumulative, fraction - 1e-12) + 1)
