"""Write fit results back out as a PINT parameter file.

The pipeline reports every fitted parameter as a posterior summary in local
coordinates -- ``d<par>_mean``, ``d<par>_initial``, ``d<par>_factor`` and
percentiles. This module converts those back into physical values with
uncertainties and folds them into a copy of the input timing model, which is
what makes a fit usable as the ephemeris for the next one.

Two conversions are easy to get wrong and are handled here:

``PB``
    Held in **seconds** throughout the fit, but written to parfiles in **days**.

``Phase``
    Not a PINT parameter at all. The per-file phase offset is expressed instead
    as ``TZRMJD``, the absolute-phase reference epoch, via
    ``TZRMJD = PEPOCH - Phase / F0``: a phase offset is a time offset of that
    many rotations. The ``AbsPhase`` component is created if the input model
    lacks one.

Exposed as the ``ell1par`` command.
"""

import copy
import os
from astropy.table import Table
from pint.models import get_model
from . import splitext_improved
from .logging import configure_logging
import logging


__all__ = [
    "create_new_parfile",
    "main",
    "update_model",
]


def update_model(model, value_dict, include_info=True):
    """Fold fit results into a copy of a timing model.

    Parameters
    ----------
    model : pint.models.TimingModel
        The model to update. Not modified; a deep copy is returned.
    value_dict : dict or astropy.table.Row
        A result row from the pipeline, holding ``d<par>_mean``,
        ``d<par>_initial``, ``d<par>_factor`` and the 16th/84th percentiles for
        each fitted parameter. Parameters absent from it are left alone, so a
        fit of two parameters updates exactly those two.
    include_info : bool, optional
        Whether to let PINT write its provenance header. Forced off on Windows,
        where the call fails.

    Returns
    -------
    pint.models.TimingModel
        A new model carrying the fitted values, their uncertainties, and
        ``frozen = False`` on everything that was fitted.

    Notes
    -----
    Only ``BinaryELL1`` and ``Spindown`` parameters are considered, plus
    ``Phase``. The uncertainty written is the larger of the two one-sigma
    half-widths, since a parfile has room for only a symmetric error.
    """
    if hasattr(value_dict, "colnames"):
        value_dict = dict((key, value_dict[key]) for key in value_dict.colnames)
    new_model = copy.deepcopy(model)
    # Note: phase must be after F0
    pars = []
    for component in model.components:
        if component not in ["BinaryELL1", "Spindown"]:
            continue
        mod = model.components[component]
        for par in mod.params:
            pars.append(par)

    pars.append("Phase")

    PEPOCH = value_dict["PEPOCH"]
    if PEPOCH != new_model.PEPOCH.value:
        new_model.PEPOCH.value = PEPOCH

    for par in pars:
        if f"d{par}_mean" not in value_dict:
            continue
        if par != "Phase":
            logging.info(f"Updating {par}")
        else:
            logging.info("Updating TZRMJD")

        mean = value_dict[f"d{par}_mean"]
        neg = mean - value_dict[f"d{par}_16"]
        pos = value_dict[f"d{par}_84"] - mean
        initial = value_dict[f"d{par}_initial"]
        factor = value_dict[f"d{par}_factor"]
        value = mean * factor + initial
        err = max(neg, pos) * factor
        if par == "Phase":
            tzrmjd = -value / new_model.F0.value / 86400 + PEPOCH
            tzrmjd_uncert = err / new_model.F0.value / 86400

            if "TZRMJD" not in new_model:
                from pint.models.absolute_phase import AbsPhase

                absph = AbsPhase()
                absph.TZRMJD.value = tzrmjd
                absph.TZRMJD.frozen = False
                absph.TZRMJD.uncertainty_value = tzrmjd_uncert
                absph.TZRSITE.value = "@"
                absph.TZRFRQ.value = 0.0
                new_model.add_component(absph)

            try:
                new_model.TZRMJD.value = tzrmjd
                new_model.TZRMJD.uncertainty_value = tzrmjd_uncert
                new_model.TZRMJD.frozen = False
                # new_model.TZRMJD.value =  PEPOCH
            except ValueError:
                pass
            continue
        if par == "PB":
            value /= 86400
            err /= 86400
        # elif par == "TASC":
        #     value = value / 86400 + PEPOCH
        #     err /= 86400

        parameter = getattr(new_model, par)
        if getattr(parameter, "unit_scale", False):
            # PINT reads "PBDOT 7.2" as 7.2e-12, and implements that convention
            # in the *assignment*: a bare float above ``scale_threshold`` (1e-7)
            # is multiplied by ``scale_factor`` (1e-12) on the way in. So
            # ``.value = x`` is not the identity for PBDOT or A1DOT, and a
            # parfile written from a fit that strayed that far -- the A1DOT
            # prior alone reaches 6e-4 -- would disagree with its own result
            # table by twelve orders of magnitude, silently. Assigning a
            # Quantity takes the units-carrying branch, which does not rescale.
            parameter.quantity = value * parameter.units
            parameter.uncertainty = err * parameter.units
            if abs(value) > abs(parameter.scale_threshold):
                # Above the threshold the format itself is ambiguous: the model
                # now holds the right number, but PINT will read the parfile
                # back rescaled, and nothing here can stop it.
                logging.warning(
                    f"{par} = {value:g} exceeds {parameter.scale_threshold:g}, which PINT "
                    f"reads back scaled by {parameter.scale_factor:g}: this parfile will not "
                    f"round-trip. No physical value reaches that magnitude -- check the fit."
                )
        else:
            parameter.value = value
            parameter.uncertainty_value = err
        parameter.frozen = False

    include_info = include_info and os.name != "nt"
    try:
        logging.info(new_model.as_parfile(include_info=include_info))
    except Exception as e:
        print(e)
        pass
    return new_model


def create_new_parfile(fname, parfile, newfile=None, include_info=True):
    """Write a new parfile from a result table and the model it started from.

    Parameters
    ----------
    fname : str
        Result table written by the pipeline. The **last** row is used, which is
        the most recent fit -- :func:`ell1fit.results_io.safe_save` appends
        rather than overwriting.
    parfile : str
        The timing model the fit started from.
    newfile : str or None, optional
        Output path. Defaults to the result table's name with a ``.par``
        extension.
    include_info : bool, optional
        Whether to let PINT write its provenance header.

    Returns
    -------
    str
        The path written.
    """
    model = get_model(parfile)
    row = Table.read(fname)[-1]
    new_model = update_model(model, row, include_info=include_info)
    include_info = include_info and os.name != "nt"
    if newfile is None:
        newfile = splitext_improved(fname)[0] + ".par"
    with open(newfile, "w") as fobj:
        print(new_model.as_parfile(include_info=include_info), file=fobj)
    return newfile


def main(args=None):
    """Main function called by the `ell1par` script"""
    import argparse

    configure_logging()

    description = "Fit an ELL1 model and frequency derivatives to an X-ray pulsar observation."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of ecsv or hdf5 files produced by `ell1fit`", nargs="+")

    parser.add_argument(
        "-p",
        "--parfile",
        type=str,
        default=None,
        help=(
            "Input parameter file. Must contain a simple ELL1 binary model, "
            "with no orbital derivatives, and a number of spin derivatives (F0, F1, ...). "
            "All other models will be ignored."
        ),
        required=True,
    )
    parser.add_argument(
        "--no-include-info",
        action="store_true",
        help="Disable metadata header in output par files.",
    )

    args = parser.parse_args(args)

    for fname in args.files:
        # Read latest measurement
        create_new_parfile(fname, args.parfile, include_info=not args.no_include_info)
