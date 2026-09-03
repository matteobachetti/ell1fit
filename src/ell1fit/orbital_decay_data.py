"""Load ell1fit ecsv results, check they share one orbital model, and build a
``PBDOT = 0`` reference ephemeris from them.

Replaces the external reference ``.par`` that a hand-rolled downstream script
used to require: instead of trusting a separately-maintained ephemeris, the
per-epoch orbital solution embedded in every ``ell1fit`` result file is
cross-checked against every other file's, and the reference model is built
from whatever they agree on.

Only ``PBDOT`` is recorded per file, not ``A1DOT``/``EPS1DOT``/``EPS2DOT``
------------------------------------------------------------------------
``ell1fit``'s ecsv output carries a plain ``PBDOT`` column, but no columns for
the other two `ORBITAL_DERIVATIVES <ell1fit.models.ORBITAL_DERIVATIVES>`. The
``{par}_offset_0`` columns that *are* present look like they should help, but
do not: they are :func:`ell1fit.models._orbital_epoch_offsets`'s own output,
computed by a single-file ``ell1fit`` run whose one-file "mean epoch" is
trivially its own epoch, so ``change_binary_epoch`` is a no-op and every
stored offset is exactly zero (confirmed against real files). So this module
can propagate ``PB`` correctly (via the ``PBDOT`` it *can* read) and weigh a
disagreement in it against how much it would actually bias the fit (see
:func:`check_compatibility`), but for ``A1``/``EPS1``/``EPS2`` it can only
require them to be exactly constant
across files -- correct whenever the underlying ephemeris has no
``A1DOT``/``EPS1DOT``/``EPS2DOT`` (true of every input file this module has
been tested against), but it would incorrectly flag genuine ``A1``/``EPS1``/
``EPS2`` drift as an incompatibility, since that drift is simply invisible in
this file format. Recording the derivatives themselves would need a change to
``ell1fit``'s own ecsv writer, out of scope here.
"""

import copy
import io
import logging
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.table import Table

from .models import ORBITAL_DERIVATIVES, _load_and_validate_models, _orbital_epoch_offsets
from .orbital_decay_model import spurious_tasc_from_pbdot_mismatch

__all__ = [
    "EpochOrbit",
    "OrbitalModelCompatibilityError",
    "build_reference_model",
    "check_compatibility",
    "load_epochs",
    "read_result_table",
    "retrieve_value_and_error",
]


class OrbitalModelCompatibilityError(RuntimeError):
    """An input file is not a valid single-epoch TASC result, or its stored
    orbital model is inconsistent with the other input files'."""


#: Columns every input file must carry, unsuffixed, to be a valid single-epoch
#: ell1fit result -- as opposed to a joint multi-epoch fit's ``_0``/``_1``-
#: suffixed output, or an F0/F1-only (e.g. Rayleigh) fit with no TASC.
_REQUIRED_COLUMNS = (
    "PEPOCH",
    "PB",
    "TASC",
    "A1",
    "EPS1",
    "EPS2",
    "PBDOT",
    "dTASC_mean",
    "dTASC_initial",
    "dTASC_factor",
    "dTASC_16",
    "dTASC_50",
    "dTASC_84",
)

#: Only PBDOT is ever readable from an ecsv row -- see the module docstring.
_READABLE_DERIVATIVES = ("PBDOT",)

#: Orbital parameters checked for cross-file compatibility, and how each is
#: checked: "propagate" means via change_binary_epoch (only possible because
#: its derivative, PBDOT, is readable); "constant" means the parameter must
#: be exactly the same in every file, because its derivative is not
#: observable from this file format.
_PROPAGATED_PARAMETERS = ("PB",)
_CONSTANT_PARAMETERS = ("A1", "EPS1", "EPS2")


def retrieve_value_and_error(row, par, multiply_by=1.0):
    """Fitted value and asymmetric (neg, pos) 1-sigma error for one parameter.

    Moved verbatim (only the source location changed) from the downstream
    ``plot_delta_tasc_new.py`` script's function of the same name, which reads
    the ``d<par>_mean/_initial/_factor/_16/_84`` columns every ``ell1fit``
    sampler writes for a fitted parameter -- the same convention
    :func:`ell1fit.create_parfile.update_model` reads on the ``.par``-writing
    side. Returns ``(None, None)`` if ``par`` was not fitted in this row.
    """
    if (
        f"d{par}_mean" not in row.colnames
        if hasattr(row, "colnames")
        else f"d{par}_mean" not in row
    ):
        return None, None
    mean = row[f"d{par}_mean"]
    initial = row[f"d{par}_initial"]
    factor = row[f"d{par}_factor"]
    value = mean * factor + initial

    err_ne = (row[f"d{par}_50"] - row[f"d{par}_16"]) * factor
    err_pe = (row[f"d{par}_84"] - row[f"d{par}_50"]) * factor
    return value * multiply_by, (err_ne * multiply_by, err_pe * multiply_by)


def _float128_to_float64_header(text):
    """Substitute ``datatype: float128`` for ``datatype: float64`` in an ECSV
    YAML header.

    Applied unconditionally, not as a fallback on a read failure: numpy's
    ``float128`` is a real 80/128-bit type on the x86_64 machines these files
    are written on, but is silently aliased to float64 on Apple Silicon,
    where ``Table.read`` instead raises outright
    (``ValueError: ... data type 'float128' not understood`` -- confirmed
    against real files on this machine). A try-then-fallback approach would
    leave the substitution path untested on any machine where the native
    dtype happens to work; doing it unconditionally makes the read path --
    and its test -- identical on every platform. The precision given up
    (~1e-11 day absolute at MJD ~57000) is far below the second-scale
    ``dTASC`` values this module fits.
    """
    return text.replace("datatype: float128", "datatype: float64")


def read_result_table(fname):
    """Read one ``ell1fit`` ecsv result file's most recent (last) row.

    Raises
    ------
    OrbitalModelCompatibilityError
        If ``fname`` is a joint multi-epoch fit's output (unsuffixed
        ``PEPOCH`` absent, ``PEPOCH_0`` present) or has no fitted TASC (e.g.
        an F0/F1-only fit), instead of a bare ``KeyError`` surfacing later.

    Returns
    -------
    astropy.table.Row
    """
    with open(fname) as fobj:
        text = _float128_to_float64_header(fobj.read())
    table = Table.read(text, format="ascii.ecsv")

    missing = [c for c in _REQUIRED_COLUMNS if c not in table.colnames]
    if missing:
        if "PEPOCH" in missing and "PEPOCH_0" in table.colnames:
            raise OrbitalModelCompatibilityError(
                f"{fname} is a joint multi-epoch fit's output (has PEPOCH_0, not PEPOCH) -- "
                "ell1decay needs one single-epoch result file per epoch, not a joint fit."
            )
        if "dTASC_mean" in missing:
            raise OrbitalModelCompatibilityError(
                f"{fname} has no fitted TASC (no dTASC_mean column) -- it looks like an "
                "F0/F1-only fit (e.g. a Rayleigh-test run), which ell1decay cannot use."
            )
        raise OrbitalModelCompatibilityError(f"{fname} is missing required column(s): {missing}")

    return table[-1]


@dataclass
class EpochOrbit:
    """One file's fitted TASC and stored orbital solution, at its own epoch."""

    fname: str
    pepoch: float  #: MJD
    tasc: float  #: fitted TASC, MJD
    tasc_err: tuple  #: (neg, pos), days
    pbdot: float  #: dimensionless, set on the parsed model via .quantity -- see from_row
    parfile_text: str = field(repr=False)  #: synthetic in-memory parfile, this row's own model

    @classmethod
    def from_row(cls, fname, row):
        tasc_offset_days, tasc_err_days = retrieve_value_and_error(row, "TASC")
        if tasc_offset_days is None:
            raise OrbitalModelCompatibilityError(f"{fname} has no fitted TASC")
        # dTASC's "initial" is the input TASC itself (see retrieve_value_and_error /
        # the original script this was moved from), so mean*factor+initial is already
        # the fitted TASC in MJD, not an offset that needs adding to anything else.
        tasc = tasc_offset_days

        # PBDOT is deliberately *not* written into this parfile text: PINT's
        # parameter parser (floatParameter._set_quantity) silently multiplies any
        # PBDOT magnitude above 1e-7 by 1e-12, assuming it was written in the
        # "x1e-12" pulsar-timing convention -- corrupting a real PBDOT by 12
        # orders of magnitude with no error raised. Real M82 X-2 values
        # (~5.7e-8) happen to stay under that threshold, but nothing here should
        # rely on that. Instead PBDOT is carried separately and applied to the
        # parsed model via ``.quantity`` in _build_models, the same fix already
        # used by orbital_decay._write_parfile.
        parfile_text = (
            "PSR                 EL1DECAY\n"
            "EPHEM               DE421\n"
            "UNITS               TDB\n"
            f"PEPOCH              {float(row['PEPOCH'])!r}\n"
            "F0                  1.0\n"
            "BINARY              ELL1\n"
            f"PB                  {float(row['PB']) / 86400.0!r}\n"
            f"A1                  {float(row['A1'])!r}\n"
            f"TASC                {float(tasc)!r}\n"
            f"EPS1                {float(row['EPS1'])!r}\n"
            f"EPS2                {float(row['EPS2'])!r}\n"
        )
        return cls(
            fname=fname,
            pepoch=float(row["PEPOCH"]),
            tasc=float(tasc),
            tasc_err=(float(tasc_err_days[0]), float(tasc_err_days[1])),
            pbdot=float(row["PBDOT"]),
            parfile_text=parfile_text,
        )


def load_epochs(files):
    """Read and validate every input file into an :class:`EpochOrbit`, sorted by PEPOCH."""
    epochs = []
    for fname in files:
        row = read_result_table(fname)
        epochs.append(EpochOrbit.from_row(fname, row))
    epochs.sort(key=lambda e: e.pepoch)
    return epochs


def _build_models(epochs):
    """``ell1fit.models._load_and_validate_models`` on every epoch's synthetic
    parfile -- reused, not reimplemented -- plus each epoch's PBDOT, applied via
    ``.quantity`` (see the comment in :meth:`EpochOrbit.from_row`) since it is
    never written into the parfile text itself.

    ``_load_and_validate_models``'s own ``ref_model`` is discarded, not reused:
    it is built (deepcopy of ``model_list[0]``, then ``change_binary_epoch`` to
    the mean epoch) *before* this function gets a chance to set PBDOT, which
    would propagate PB to the mean epoch using PBDOT=None (parsed from text
    with no PBDOT line at all). So ``ref_model`` is rebuilt here, identically,
    but only after every model's real PBDOT is in place.

    Returns
    -------
    (model_list, pepoch_list, ref_model)
    """
    parfile_likes = [io.StringIO(e.parfile_text) for e in epochs]
    model_list, pepoch_list, _ = _load_and_validate_models(parfile_likes)

    for epoch, model in zip(epochs, model_list):
        model.PBDOT.quantity = epoch.pbdot * u.dimensionless_unscaled

    ref_model = copy.deepcopy(model_list[0])
    ref_model.change_binary_epoch(np.mean(pepoch_list))

    return model_list, pepoch_list, ref_model


#: A PB discrepancy left over after subtracting off whatever a file's own
#: reported PBDOT would explain (see check_compatibility) should be exactly
#: zero -- PB is a plain linear function of PBDOT under change_binary_epoch,
#: nothing approximate about it. In practice a few milliseconds of residual
#: shows up even between files from the same consistent processing batch
#: (observed: ~13 ms on real M82 X-2 data), presumably from whatever rounding
#: the upstream process that generated each file's fixed PB input used. This
#: is generous enough to absorb that and still catch a genuinely different
#: orbital model (PB0 itself wrong, or files from different targets/BINARY
#: types mixed together), which would be wrong at the PB scale (~days), not
#: milliseconds.
_PB_RESIDUAL_TOLERANCE_SEC = 1.0


def check_compatibility(epochs, tolerance=1e-9, pbdot_impact_fraction=1.0):
    """Verify every file's stored orbital solution is one shared model,
    propagated (where possible) to each file's own epoch via PINT's own
    ``change_binary_epoch``.

    A1/EPS1/EPS2 are checked for exact constancy, since their derivatives are
    not observable from this file format (see the module docstring). PB and
    PBDOT are checked together: a raw PBDOT difference between files (e.g.
    from two processing batches assuming slightly different upstream
    ephemerides) is not itself the thing that matters -- what matters is how
    much spurious signal it would inject into the delta_tasc(t) curve being
    fit, via :func:`ell1fit.orbital_decay_model.spurious_tasc_from_pbdot_mismatch`,
    weighed against that epoch's own TASC uncertainty. Small, real-world
    differences get a warning, not an abort; only a PBDOT difference large
    (or a baseline long) enough to rival that epoch's own statistical
    precision is treated as unsound. Whatever part of the PB discrepancy
    *isn't* explained by the reported PBDOT difference is checked separately,
    strictly -- that residual has no benign explanation (see
    :data:`_PB_RESIDUAL_TOLERANCE_SEC`).

    Parameters
    ----------
    pbdot_impact_fraction : float
        Abort threshold: a PBDOT difference whose spurious-delta_tasc impact
        reaches this fraction of an epoch's own TASC uncertainty aborts;
        below it, only a warning is logged. Default ``1.0`` (the systematic
        would have to rival the epoch's full statistical uncertainty).

    Raises
    ------
    OrbitalModelCompatibilityError
        Hard-abort, listing every offending file and its actual-vs-predicted
        (or actual-vs-reference) values.
    """
    logging.info("Checking orbital model compatibility across %d epochs", len(epochs))
    if len(epochs) < 2:
        raise OrbitalModelCompatibilityError(
            f"Need at least 2 input files to fit an orbital-decay model, got {len(epochs)}."
        )

    model_list, pepoch_list, ref_model = _build_models(epochs)

    problems = []

    # A1/EPS1/EPS2: exact constancy (their derivatives are not observable here).
    for par in _CONSTANT_PARAMETERS:
        reference_value = float(getattr(model_list[0], par).value)
        for epoch, model in zip(epochs, model_list):
            value = float(getattr(model, par).value)
            atol = tolerance if reference_value == 0 else abs(reference_value) * tolerance
            if not np.isclose(value, reference_value, rtol=tolerance, atol=atol):
                problems.append(
                    f"{epoch.fname}: {par}={value!r} disagrees with {epochs[0].fname}'s "
                    f"{par}={reference_value!r} (tolerance={tolerance:.1e} relative) -- if "
                    f"this dataset genuinely has a nonzero {par}DOT, ell1decay cannot see it "
                    "(not recorded in the ecsv format) and this check will misfire; see "
                    "orbital_decay_data's module docstring."
                )

    # PB/PBDOT: propagate ref_model to each file's own epoch, then split the
    # discrepancy into the part this file's own reported PBDOT explains
    # (weighed against TASC precision, lenient) and whatever residual is left
    # (strict -- see _PB_RESIDUAL_TOLERANCE_SEC).
    offsets = _orbital_epoch_offsets(ref_model, pepoch_list)
    reference_pb_seconds = float(ref_model.PB.value) * 86400.0
    reference_pbdot = float(ref_model.PBDOT.value)
    pb0_days = float(ref_model.PB.value)
    mean_epoch = float(np.mean(pepoch_list))

    for epoch, model, offset in zip(epochs, model_list, offsets):
        predicted_pb_seconds = reference_pb_seconds + offset["PB_offset"]
        actual_pb_seconds = float(model.PB.value) * 86400.0
        pb_discrepancy_sec = actual_pb_seconds - predicted_pb_seconds

        delta_pbdot = float(model.PBDOT.value) - reference_pbdot
        dt_days = epoch.pepoch - mean_epoch
        explained_sec = delta_pbdot * dt_days * 86400.0
        residual_sec = pb_discrepancy_sec - explained_sec

        if abs(residual_sec) > _PB_RESIDUAL_TOLERANCE_SEC:
            problems.append(
                f"{epoch.fname}: PB={actual_pb_seconds!r} s disagrees with the value "
                f"predicted by propagating the shared reference model to this file's own "
                f"epoch ({predicted_pb_seconds!r} s) by {residual_sec:+.4g} s, after already "
                f"accounting for this file's own reported PBDOT difference -- unexplained, "
                f"exceeds the {_PB_RESIDUAL_TOLERANCE_SEC:g} s tolerance."
            )

        if delta_pbdot == 0:
            continue
        spurious_tasc_sec = spurious_tasc_from_pbdot_mismatch(delta_pbdot, dt_days, pb0_days)
        tasc_err_sec = min(epoch.tasc_err) * 86400.0
        fraction = spurious_tasc_sec / tasc_err_sec if tasc_err_sec > 0 else np.inf
        if fraction >= pbdot_impact_fraction:
            problems.append(
                f"{epoch.fname}: PBDOT={float(model.PBDOT.value)!r} differs from the group's "
                f"reference ({reference_pbdot!r}) by {delta_pbdot!r} -- at this epoch's distance "
                f"from the reference ({dt_days:+.1f} d), that alone would inject "
                f"~{spurious_tasc_sec:.3g} s of spurious delta_tasc, {fraction:.1%} of this "
                f"epoch's own {tasc_err_sec:.3g} s TASC uncertainty (threshold "
                f"{pbdot_impact_fraction:.0%}) -- unsound."
            )
        elif fraction > 0.01:
            logging.warning(
                f"{epoch.fname}: PBDOT differs from the group's reference by {delta_pbdot!r} -- "
                f"estimated spurious delta_tasc at this epoch: ~{spurious_tasc_sec:.3g} s "
                f"({fraction:.1%} of its {tasc_err_sec:.3g} s TASC uncertainty). Continuing, but "
                "this will bias the fitted PBDOT/PBDDOT somewhat."
            )

    if problems:
        raise OrbitalModelCompatibilityError(
            "Input files are not consistent with one shared orbital model:\n  "
            + "\n  ".join(problems)
        )


def build_reference_model(epochs, reference_epoch=None):
    """Build the ``PBDOT = 0`` reference ephemeris the fit measures ``delta_tasc`` against.

    Must be called only after :func:`check_compatibility` has passed -- it
    does not re-validate.

    Parameters
    ----------
    reference_epoch : float or None
        MJD to reference PB/A1/EPS1/EPS2 to. Defaults to the mean PEPOCH
        across ``epochs``, following :func:`ell1fit.models._load_and_validate_models`'s
        own precedent, and centers the fit's ``x = t - reference_epoch`` axis
        on the data -- material here because the fit measures exactly the
        offset/slope/curvature terms that a centered axis decorrelates.

    Returns
    -------
    pint.models.timing_model.TimingModel
        PBDOT (and any other derivative) forced to 0. PB/A1/EPS1/EPS2 are the
        mean of every file's own value, each propagated to ``reference_epoch``
        (already agree to ~1e-13 relative once propagated, given
        ``check_compatibility`` passed at the default tolerance, so the mean
        is for robustness, not because file-to-file weighting matters here).
    """
    logging.info("Building reference model at epoch %s", reference_epoch or "mean(PEPOCH)")
    model_list, pepoch_list, ref_model = _build_models(epochs)

    if reference_epoch is None:
        reference_epoch = float(np.mean(pepoch_list))

    ref_model = copy.deepcopy(ref_model)
    ref_model.change_binary_epoch(reference_epoch)
    # change_binary_epoch propagates TASC/PB/A1/... but does not touch PEPOCH
    # itself (that stays whatever model[0]'s own PEPOCH happened to be) --
    # set it explicitly so the returned model's own metadata matches the
    # epoch its orbital elements actually describe.
    ref_model.PEPOCH.value = reference_epoch

    offsets = _orbital_epoch_offsets(ref_model, pepoch_list)
    for par in ("PB", "A1", "EPS1", "EPS2"):
        base = float(getattr(ref_model, par).value) * (86400.0 if par == "PB" else 1.0)
        propagated = [base + offset[f"{par}_offset"] for offset in offsets]
        mean_value = float(np.mean(propagated)) / (86400.0 if par == "PB" else 1.0)
        getattr(ref_model, par).value = mean_value

    for derivative in ORBITAL_DERIVATIVES:
        parameter = getattr(ref_model, derivative, None)
        if parameter is not None and parameter.value is not None:
            parameter.value = 0.0

    return ref_model
