"""Timing-model loading and parameter-dictionary assembly for ell1fit.

Bridges PINT timing models and the plain dictionaries the rest of the pipeline
uses. Two representations are in play throughout:

``parameters_with_unc``
    ``{name: [value, uncertainty]}``, used to build priors and scaling factors.
``parameters``
    ``{name: value}``, the working set that phases are computed from.

Per-file bookkeeping
--------------------
Spin parameters are per-file (``F0_0``, ``F0_1``, ...), because each event file
carries its own ``PEPOCH`` and the spin frequency is only valid at that epoch.
Orbital parameters are global: one binary, one solution. The pulse phase offset
``Phase_i`` is per-file too, as a free nuisance parameter absorbing each
observation's arbitrary phase zero.

``TASC`` is only defined modulo ``PB``. :func:`_load_and_validate_models`
re-references the shared model to the mean ``PEPOCH``, so the ``TASC`` it
reports can differ from the input parfile's by a whole number of orbits while
describing the identical orbit.
"""

import copy
import logging

import numpy as np
from pint.models import get_model


__all__ = [
    "_build_parameters_from_models",
    "_load_and_validate_models",
]


def _load_and_validate_models(parfiles):
    """Load PINT models, validate ELL1 constraints, and align binary epochs."""
    model = []
    pepoch = []

    for i in range(len(parfiles)):
        model.append(get_model(parfiles[i]))
        pepoch.append(model[i].PEPOCH.value)

        if hasattr(model[i], "T0") or model[i].BINARY.value != "ELL1":
            raise ValueError("This script wants an ELL1 model, with TASC, not T0, defined")

        model[i].change_binary_epoch(pepoch[i])

    ref_model = copy.deepcopy(model[0])
    ref_model.change_binary_epoch(np.mean(pepoch))

    return model, pepoch, ref_model


def _build_parameters_from_models(model, ref_model, observation_length, ignore_uncertainties=False):
    """Assemble global and per-file parameter dictionaries from timing models."""
    n_files = len(model)
    parameters_with_unc = _get_par_dict(
        ref_model,
        ignore_uncertainties=ignore_uncertainties,
        obs_length=np.min(observation_length),
    )
    del parameters_with_unc["PEPOCH"]

    for i in range(n_files):
        count = 0
        file_parameters_with_unc = _get_par_dict(
            model[i],
            ignore_uncertainties=ignore_uncertainties,
            obs_length=observation_length[i],
        )

        while f"F{count}" in file_parameters_with_unc:
            parameters_with_unc[f"F{count}_{i}"] = [
                file_parameters_with_unc[f"F{count}"][0],
                file_parameters_with_unc[f"F{count}"][1],
            ]
            if f"F{count}" in parameters_with_unc:
                del parameters_with_unc[f"F{count}"]
            count += 1

        parameters_with_unc[f"PEPOCH_{i}"] = [
            file_parameters_with_unc["PEPOCH"][0],
            file_parameters_with_unc["PEPOCH"][1],
        ]
        parameters_with_unc[f"Phase_{i}"] = [
            parameters_with_unc["Phase"][0],
            parameters_with_unc["Phase"][1],
        ]

    # _calculate_phases expects file-specific phase keys.
    del parameters_with_unc["Phase"]
    parameters = {f: parameters_with_unc[f][0] for f in parameters_with_unc}
    return parameters_with_unc, parameters


def _get_par_dict(
    model,
    ignore_uncertainties=False,
    obs_length=1,
):  # The dictionary contains lists [parameter mean, parameter uncertainty]
    """Build a parameter/uncertainty dictionary from a PINT timing model.

    The returned mapping stores ``[value, uncertainty]`` for each parameter and
    fills missing uncertainties with heuristic defaults suitable for priors.
    """

    def return_unc(param):
        if param.uncertainty_value is None or param.uncertainty_value == 0:
            return np.nan
        return param.uncertainty_value.astype(float)

    parameters = {
        "Phase": [0, 0],
        "PB": [model.PB.value.astype(float) * 86400, return_unc(model.PB) * 86400],
        "TASC": [model.TASC.value.astype(float), return_unc(model.TASC)],
        "A1": [model.A1.value.astype(float), return_unc(model.A1)],
        "EPS1": [model.EPS1.value.astype(float), return_unc(model.EPS1)],
        "EPS2": [model.EPS2.value.astype(float), return_unc(model.EPS2)],
        "PBDOT": [model.PBDOT.value.astype(float), return_unc(model.PBDOT)],
        "PEPOCH": [
            model.PEPOCH.value.astype(float),
            return_unc(model.PEPOCH),
        ],  # I added Pepoch
    }

    count = 0
    while hasattr(model, f"F{count}"):
        parameters[f"F{count}"] = [
            getattr(model, f"F{count}").value.astype(float),
            return_unc(getattr(model, f"F{count}")),
        ]
        count += 1

    if ignore_uncertainties:
        # Start from a clean slate
        for par in parameters:
            parameters[par][1] = np.nan

    # Then, give sensible defaults for the uncertainties of some critical
    # parameters that are not set
    def check_uncertainty(par, default_uncertainty):
        if np.isnan(parameters[par][1]) or np.isinf(parameters[par][1]) or ignore_uncertainties:
            parameters[par][1] = default_uncertainty

    check_uncertainty("PB", parameters["PB"][0] / 2)

    Omega = 2 * np.pi / parameters["PB"][0]
    X = parameters["A1"][0]
    f = parameters["F0"][0]

    count = 0

    while hasattr(model, f"F{count}"):
        obs_length_change = 10 / obs_length ** (count + 1)
        max_orbital_change = X * Omega ** (count + 1) * f
        logging.debug(
            f"F{count}: max_orbital_change={max_orbital_change}, "
            f"obs_length_change={obs_length_change}"
        )
        default_unc = 10 * max_orbital_change + obs_length_change
        check_uncertainty(f"F{count}", default_unc)
        count += 1

    return parameters
