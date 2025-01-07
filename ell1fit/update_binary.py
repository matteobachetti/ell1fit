import copy

from pint.models import get_model
from pint.models.parameter import funcParameter


def update_binary_model(input_model, reference_model):
    """Update the binary model in input_model to match the one in reference_model."""

    new_model = copy.deepcopy(input_model)
    for par in input_model.components["BinaryELL1"].params:
        if isinstance(getattr(input_model, par), funcParameter):
            print("funcparameter")
            continue
        if getattr(new_model, par).frozen and not getattr(reference_model, par).frozen:
            getattr(new_model, par).frozen = getattr(reference_model, par).frozen
        if (
            getattr(input_model, par).quantity is None
            and getattr(reference_model, par).quantity is None
        ):
            continue
        getattr(new_model, par).value = getattr(reference_model, par).value

        if getattr(new_model, par).uncertainty_value is not None:
            getattr(new_model, par).uncertainty_value = getattr(
                reference_model, par
            ).uncertainty_value
    new_model.change_binary_epoch(new_model.PEPOCH.value)
    return new_model


def main(args=None):
    """Main function called by the `ell1updatebinary` script"""
    import argparse

    description = "Copy the ELL1 model from one parameter file into one or more others."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="Parameter file(s) to be modified", nargs="+")

    parser.add_argument(
        "-p",
        "--parfile",
        type=str,
        default=None,
        help=("Input parameter file. Must contain a simple ELL1 binary model"),
        required=True,
    )

    args = parser.parse_args(args)

    reference_model = get_model(args.parfile)

    for fname in args.files:
        out_fname = fname.replace(".par", "_new.par")

        local_model = get_model(fname)
        new_model = update_binary_model(local_model, reference_model)
        with open(out_fname, "w") as fobj:
            print(new_model.as_parfile(), file=fobj)

        print(new_model.as_parfile())
