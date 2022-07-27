import copy
from astropy.table import Table
from pint.models import get_model
from .ell1fit import splitext_improved


def update_model(model, value_dict):
    new_model = copy.deepcopy(model)
    pars = "F0,TASC,PB,A1,EPS1,EPS2"
    count = 1
    while hasattr(new_model, f"F{count}"):
        pars += f",F{count}"
        count += 1

    for par in pars.split(","):
        if not f"d{par}_mean" in value_dict.colnames:
            continue
        print(f"Updating {par}")
        mean = value_dict[f"d{par}_mean"]
        initial = value_dict[f"d{par}_initial"]
        factor = value_dict[f"d{par}_factor"]
        value = mean * factor + initial
        if par == "PB":
            value /= 86400
        if par == "TASC":
            value = value / 86400 + value_dict["PEPOCH"]

        getattr(new_model, par).value = value
    return new_model


def main(args=None):
    """Main function called by the `ell1par` script"""
    import argparse

    description = (
        "Fit an ELL1 model and frequency derivatives to an X-ray "
        "pulsar observation."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of ecsv or hdf5 files produced by `ell1fit`", nargs="+")

    parser.add_argument("-p", "--parfile", type=str, default=None, help="Parameter file to be updated with new information. Must contain a simple ELL1 binary model, with no orbital derivatives, and a number of spin derivatives (F0, F1, ...). ", required=True)

    args = parser.parse_args(args)

    model = get_model(args.parfile)
    for fname in args.files:
        # Read latest measurement
        row = Table.read(fname)[-1]
        new_model = update_model(model, row)
        with open(splitext_improved(fname)[0] + ".par", "w") as fobj:
            print(new_model.as_parfile(), file=fobj)
