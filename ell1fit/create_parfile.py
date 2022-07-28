import copy
from astropy.table import Table
from pint.models import get_model
from .ell1fit import splitext_improved


def update_model(model, value_dict):
    if hasattr(value_dict, "colnames"):
        value_dict = dict((key, value_dict[key]) for key in value_dict.colnames)
    new_model = copy.deepcopy(model)
    # Note: phase must be after F0
    pars = "F0,Phase,TASC,PB,A1,EPS1,EPS2"
    count = 1
    while hasattr(new_model, f"F{count}"):
        pars += f",F{count}"
        count += 1

    PEPOCH = value_dict["PEPOCH"]
    if PEPOCH != new_model.PEPOCH.value:
        new_model.PEPOCH.value = PEPOCH
    print(PEPOCH)

    for par in pars.split(","):
        if f"d{par}_mean" not in value_dict:
            continue
        print(f"Updating {par}")
        mean = value_dict[f"d{par}_mean"]
        initial = value_dict[f"d{par}_initial"]
        factor = value_dict[f"d{par}_factor"]
        value = mean * factor + initial
        print(value, mean, factor, initial)
        if par == "Phase":
            print(value, new_model.F0.value, PEPOCH, value / new_model.F0.value / 86400)
            new_model.TZRMJD.value = value / new_model.F0.value / 86400 + PEPOCH
            # new_model.TZRMJD.value =  PEPOCH
            continue
        if par == "PB":
            value /= 86400
        elif par == "TASC":
            value = value / 86400 + PEPOCH

        getattr(new_model, par).value = value
    return new_model


def create_new_parfile(fname, parfile, newfile=None):
    model = get_model(parfile)
    row = Table.read(fname)[-1]
    new_model = update_model(model, row)
    if newfile is None:
        newfile = splitext_improved(fname)[0] + ".par"
    with open(newfile, "w") as fobj:
        print(new_model.as_parfile(), file=fobj)
    return newfile


def main(args=None):
    """Main function called by the `ell1par` script"""
    import argparse

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

    args = parser.parse_args(args)

    for fname in args.files:
        # Read latest measurement
        create_new_parfile(fname, args.parfile)
