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
    "ORBITAL_DERIVATIVES",
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


#: Orbital derivatives that PINT propagates when a binary epoch is moved.
#: Honoured as inputs, so that the orbital parameters are valid at each file's
#: own epoch. Only ``A1DOT`` can also be *fitted*; see :data:`OFFSET_DERIVATIVES`.
ORBITAL_DERIVATIVES = ("PBDOT", "A1DOT", "EPS1DOT", "EPS2DOT")

#: The subset of :data:`ORBITAL_DERIVATIVES` whose effect is carried by a fixed
#: per-file offset, computed once at load.
#:
#: ``A1DOT`` is deliberately absent. Its entire effect is ``A1 += A1DOT * dt``
#: with ``dt`` a constant lever arm, so applying it to the *trial* ``A1DOT``
#: inside the phase model costs one multiply and makes it fittable, where
#: freezing it into an offset would make the likelihood flat in it -- the
#: reason ``PBDOT`` is refused (see
#: :data:`ell1fit.pipeline.UNFITTABLE_PARAMETERS`).
OFFSET_DERIVATIVES = ("PBDOT", "EPS1DOT", "EPS2DOT")

#: Orbital parameters that acquire a fixed per-file offset, with the unit
#: conversion needed to express that offset the way the parameter dictionary
#: stores it (``PB`` in seconds, everything else in the model's own units).
_OFFSET_PARAMETERS = {"PB": 86400.0, "A1": 1.0, "EPS1": 1.0, "EPS2": 1.0}


def _model_has_orbital_derivatives(model, names=ORBITAL_DERIVATIVES):
    """Whether any of ``names`` is set to a nonzero value."""
    for name in names:
        parameter = getattr(model, name, None)
        if parameter is not None and parameter.value is not None and parameter.value != 0:
            return True
    return False


def _orbital_epoch_offsets(ref_model, pepoch):
    """Fixed per-file corrections carrying the orbital solution to each epoch.

    The fit uses **one** binary: a single global ``PB``, ``TASC``, ``A1``,
    ``EPS1`` and ``EPS2``. But those values are only valid at the epoch they are
    referenced to, and with a nonzero ``PBDOT`` (or ``A1DOT``, ``EPS1DOT``,
    ``EPS2DOT``) they drift away from it. Evaluating one mean-epoch snapshot at
    every file's own epoch is wrong by an amount that grows as
    ``PBDOT * baseline**2``, which crosses the precision a fit achieves for
    realistic values over a multi-year baseline.

    So each file gets a *constant* offset, added to the global value before
    phases are computed. The offsets are not fitted and do not depend on the
    trial parameters: they are second order (``PB_offset = PBDOT * dt``), so moving
    ``PB`` by its own uncertainty changes them negligibly, and they can be
    computed once at load.

    PINT does the propagation, through the same ``change_binary_epoch`` used to
    align the models in the first place. Deriving it here instead would make
    agreement with PINT a tautology rather than a check, and would silently drop
    the parameterizations (``FB0``/``FB1``) and derivatives it already handles.

    ``A1DOT`` is the exception, and is not an offset
    -----------------------------------------------
    Everything above rests on the correction being *fixed*. ``A1DOT`` is the one
    derivative for which that assumption costs something real: it is exactly
    what an upper limit on the orbit's changing size is measured from, and a
    frozen offset makes the likelihood flat in it, so it could never be fitted.

    Its effect is also the simplest of the four -- PINT's whole treatment is
    ``A1 += A1DOT * dt_integer_orbits`` -- so this function hands back that
    ``dt`` as ``binary_dt`` (seconds) instead, and
    :func:`ell1fit.phase_utils._calculate_phases` multiplies it by the *trial*
    ``A1DOT`` on every evaluation. ``A1DOT`` is zeroed before the propagation
    here so the drift is applied once, not twice, which leaves ``A1_offset``
    identically zero unless a future PINT moves ``A1`` by some other route.
    The lever arm is PINT's own, so at the parfile's ``A1DOT`` the model is the
    one PINT implies, and a fitted ``A1DOT`` written back to a parfile reads in
    as the same solution.

    The ``TASC`` offset is the subtle one
    -------------------------------------
    ``_calculate_phases`` brings ``TASC`` close to each ``PEPOCH`` by wrapping it
    modulo ``PB``, which re-adds a whole number of orbits ``n * PB`` computed
    with the *trial* period -- and that is exactly right, because it is how a
    ``PB`` error turns into a growing phase offset across epochs, which is what
    makes ``PB`` measurable from multi-epoch data. What the wrap cannot express
    is the quadratic term the exact model accumulates,
    ``n**2 * PB * PBDOT / 2``. ``TASC_offset`` supplies precisely that residual, so
    that wrap and offset together reproduce PINT's ``dt_integer_orbits``.
    Applying ``PB_offset`` *without* it is worse than doing nothing: it flips the sign
    of the error while keeping its magnitude.

    Returns
    -------
    list of dict
        One mapping per entry of ``pepoch``, with keys ``TASC_offset``
        (days), ``PB_offset`` (seconds), ``A1_offset``, ``EPS1_offset``,
        ``EPS2_offset`` and ``binary_dt`` (seconds). Every *offset* is
        exactly ``0.0`` when the model sets none of
        :data:`OFFSET_DERIVATIVES`, which keeps the common case bit-for-bit
        unchanged. ``binary_dt`` is not an offset and is always real: it is the
        lever arm ``A1DOT`` acts over, needed whether or not the parfile sets
        one, since the fit may.
    """
    zero = {
        "TASC_offset": 0.0,
        "PB_offset": 0.0,
        "A1_offset": 0.0,
        "EPS1_offset": 0.0,
        "EPS2_offset": 0.0,
    }

    # A1DOT is applied to the trial value inside the phase model, so it must not
    # also be baked into A1_offset here.
    propagation_model = copy.deepcopy(ref_model)
    if getattr(propagation_model, "A1DOT", None) is not None:
        propagation_model.A1DOT.value = 0.0

    has_offset_derivatives = _model_has_orbital_derivatives(ref_model, OFFSET_DERIVATIVES)
    reference = {par: float(getattr(ref_model, par).value) for par in _OFFSET_PARAMETERS}
    offsets = []

    for epoch in pepoch:
        epoch_model = copy.deepcopy(propagation_model)
        dt_integer_orbits = epoch_model.change_binary_epoch(epoch)

        if dt_integer_orbits is None:
            # PINT returns early when the epoch is already the closest ascending
            # node: nothing was propagated, so there is nothing to correct, and
            # the lever arm is zero because the epoch *is* the reference.
            offsets.append(dict(zero, binary_dt=0.0))
            continue

        dt_seconds = float(dt_integer_orbits.to("d").value) * 86400.0

        if not has_offset_derivatives:
            offsets.append(dict(zero, binary_dt=dt_seconds))
            continue

        offset = {
            f"{par}_offset": (float(getattr(epoch_model, par).value) - reference[par]) * scale
            for par, scale in _OFFSET_PARAMETERS.items()
        }
        offset["binary_dt"] = dt_seconds

        # Whole orbits are the wrap's job; hand it back everything except the
        # residual. The subtraction cancels ~8 leading digits, which still
        # leaves far more precision than the correction itself needs.
        pb_seconds = float(epoch_model.PB.value) * 86400.0
        n_orbits = np.round(dt_seconds / pb_seconds)
        offset["TASC_offset"] = (dt_seconds - n_orbits * pb_seconds) / 86400.0

        logging.info(
            f"Orbital epoch offset at PEPOCH={epoch}: "
            f"TASC{offset['TASC_offset'] * 86400:+.4g} s, "
            f"PB{offset['PB_offset']:+.4g} s, A1{offset['A1_offset']:+.4g} lt-s"
        )
        offsets.append(offset)

    return offsets


def _build_parameters_from_models(model, ref_model, observation_length, ignore_uncertainties=False):
    """Assemble global and per-file parameter dictionaries from timing models."""
    n_files = len(model)
    parameters_with_unc = _get_par_dict(
        ref_model,
        ignore_uncertainties=ignore_uncertainties,
        obs_length=np.min(observation_length),
    )
    del parameters_with_unc["PEPOCH"]

    epoch_offsets = _orbital_epoch_offsets(ref_model, [m.PEPOCH.value for m in model])

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

        # Fixed epoch corrections, not fitted: uncertainty 0 marks them as
        # exact constants rather than quantities with a prior.
        for name, value in epoch_offsets[i].items():
            parameters_with_unc[f"{name}_{i}"] = [value, 0.0]

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
        # Unlike PBDOT, A1DOT has no PINT default: a parfile that omits it
        # leaves the value None, which reads as "no drift", i.e. zero.
        "A1DOT": [
            0.0 if model.A1DOT.value is None else float(model.A1DOT.value),
            np.nan if model.A1DOT.value is None else return_unc(model.A1DOT),
        ],
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
