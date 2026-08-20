"""Construction of output file-name roots for ell1fit.

Every product of a run -- plots, chains, result tables, updated parfiles -- is
named from a common root that encodes the analysis configuration, so that runs
with different settings do not overwrite each other. This module owns the rules
for building that root and nothing else.

The root is assembled as::

    <base>_<sorted fit parameters><energy><nharm><likelihood><weight><pi>

where ``<base>`` is the event file's name for per-file products, or the
user-supplied ``-o`` value (falling back to ``"out"``) for combined ones.
"""

from . import splitext_improved
from .likelihoods import rayleigh_as_likelihood
from .results_io import _format_energy_string


def _get_likelihood_suffix(likelihood_func):
    """Return output-name suffix for selected likelihood implementation."""
    if likelihood_func == rayleigh_as_likelihood:
        return "_rayleigh"
    return ""


def _get_weight_suffix(use_weight):
    """Return output-name suffix when energy weights are enabled."""
    if use_weight:
        return "_pf_weight"
    return ""


def _get_pi_suffix(use_pi):
    """Return output-name suffix when weighting uses PI channels instead of energy."""
    if use_pi:
        return "_pi"
    return ""


def _get_nharm_suffix(nharm):
    """Return output-name suffix for harmonic count when > 1."""
    if nharm > 1:
        return f"_N{nharm}"
    return ""


def _make_outroot_getter(
    files,
    requested_parameter_names,
    energy_range,
    nharm,
    likelihood_func,
    use_weight,
    use_pi=False,
    general_outroot=None,
):
    """Build a closure that returns the configured output root name.

    Returns
    -------
    callable
        ``get_outroot(file_n=None)``. Passing a file index yields the root for
        that file's own products; passing ``None`` yields the root for the
        combined, multi-file products.
    """
    energy_str = _format_energy_string(energy_range)
    nharm_str = _get_nharm_suffix(nharm)
    likelihood_str = _get_likelihood_suffix(likelihood_func)
    weight_str = _get_weight_suffix(use_weight)
    pi_str = _get_pi_suffix(use_pi)

    def get_outroot(file_n=None):
        if file_n is not None:
            initial_outroot = splitext_improved(files[file_n])[0]
        elif general_outroot is not None:
            initial_outroot = general_outroot
        else:
            initial_outroot = "out"

        outroot = (
            initial_outroot
            + "_"
            + "_".join(requested_parameter_names)
            + energy_str
            + nharm_str
            + likelihood_str
            + weight_str
            + pi_str
        )
        return outroot

    return get_outroot


def _get_outroots(get_outroot, n_files):
    """Return per-file roots plus a final aggregate root.

    With a single input file the aggregate root is that file's own root, so a
    one-file run does not scatter its products across two different names.
    """
    outroots = [get_outroot(i) for i in range(n_files)]
    if n_files == 1:
        outroots += [get_outroot(0)]
    else:
        outroots += [get_outroot(None)]
    return outroots
