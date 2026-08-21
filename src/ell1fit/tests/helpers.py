"""Shared scaffolding for tests that need pipeline state, not just a CLI run.

:func:`build_pipeline_state` performs the same sequence :func:`ell1fit.pipeline`
does -- load models, load events, fold, build templates, assemble priors and
scaling -- and hands back the two bundles the fitting machinery takes. Tests
that need to call into ``point_estimate_fit`` or ``refine_templates_and_solution``
directly would otherwise each repeat thirty lines of setup.

Templates are deliberately built from whatever solution the parfiles carry. When
those parfiles are offset from the truth, the resulting templates are smeared and
skewed exactly as they would be in a real analysis with an imperfect ephemeris --
which is the condition the refinement tests exist to exercise.
"""

import os

from ..pipeline import _prepare_fit_setup
from ..events import _load_events_for_all_files
from ..likelihoods import pletsch_clarke_likelihood
from ..models import _build_parameters_from_models, _load_and_validate_models
from ..phase_utils import folded_profile, phases_around_zero
from ..setup_types import ObservationSet
from ..templates import create_template_from_profile_harm, get_template_func


def build_pipeline_state(
    dataset,
    fit_parameters=("F0", "A1"),
    nharm=2,
    tolerance=1e-8,
    likelihood_func=pletsch_clarke_likelihood,
):
    """Build ``(observations, setup)`` from a generated dataset.

    Parameters
    ----------
    dataset : dict
        As returned by :func:`ell1fit.tests.datagen.make_multi_epoch_dataset`.
    fit_parameters : sequence of str, optional
        Parameter tokens to fit, as the ``-P`` flag would supply them.
    nharm : int, optional
        Harmonics retained in the pulse templates.
    tolerance : float, optional
        Deorbiting tolerance, in seconds.
    likelihood_func : callable, optional
        Statistic to fit with. This is not merely stored: it decides which
        parameters are free. ``_collect_parameter_names`` only adds the per-file
        ``Phase_i`` nuisance parameters for the Pletsch-Clarke likelihood,
        because the Rayleigh statistic is invariant under a global phase shift
        and would leave them as unconstrained flat directions.

    Returns
    -------
    observations : ObservationSet
    setup : FitSetup
    """
    outdir = os.path.dirname(dataset["event_files"][0])
    models, pepoch, ref_model = _load_and_validate_models(dataset["par_files"])

    def get_outroot(file_n=None):
        return os.path.join(outdir, f"state_{file_n}")

    times, obs_length, energies, exposures = _load_events_for_all_files(
        dataset["event_files"], None, pepoch, get_outroot
    )

    observations = ObservationSet(
        files=dataset["event_files"],
        models=models,
        ref_model=ref_model,
        pepoch=pepoch,
        times_from_pepoch=times,
        energies=energies,
        exposures=exposures,
        observation_length=obs_length,
    )

    parameters_with_unc, parameters = _build_parameters_from_models(models, ref_model, obs_length)

    nbin = max(32, nharm * 8)
    profiles = folded_profile(times, parameters, nbin=nbin, tolerance=tolerance)

    template_funcs = []
    for i, profile in enumerate(profiles):
        template, phase = create_template_from_profile_harm(
            profile, nharm=nharm, final_nbin=200, plot=False
        )
        template_funcs.append(get_template_func(template))
        offset = -phases_around_zero(phase)
        parameters[f"Phase_{i}"] = offset
        parameters_with_unc[f"Phase_{i}"][0] = offset

    setup = _prepare_fit_setup(
        parameters,
        sorted(fit_parameters),
        likelihood_func,
        parameters_with_unc,
        obs_length,
        models,
        template_funcs=template_funcs,
        weights=None,
        tolerance=tolerance,
    )
    return observations, setup
