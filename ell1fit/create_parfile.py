import copy
import os
from astropy.table import Table
from pint.models import get_model
from . import splitext_improved
from .logging import configure_logging
import logging


def update_model(model, value_dict, include_info=True):
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

        getattr(new_model, par).value = value
        getattr(new_model, par).uncertainty_value = err
        getattr(new_model, par).frozen = False

    include_info = include_info and os.name != "nt"
    try:
        logging.info(new_model.as_parfile(include_info=include_info))
    except Exception as e:
        print(e)
        pass
    return new_model


def create_new_parfile(fname, parfile, newfile=None, include_info=True):
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

    description = "Fit an ELL1 model and frequency derivatives to an X-ray " "pulsar observation."
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
