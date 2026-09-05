r"""One figure summarising the orbit a fit explored.

The main corner plot is drawn in the sampler's own local coordinates: offsets
from the starting solution, in units of each parameter's preconditioned scale.
That is the right frame for judging whether the chain moved, and the wrong one
for reading a number off. This module draws the other view -- the orbital
parameters in the units they are quoted in, next to the eccentricity posterior
they imply.

Making a physical parameter readable
------------------------------------

An orbital posterior is a spike in an inconvenient place: ``A1`` might be
26.5 light-seconds wide by a millionth of one, ``TASC`` an MJD known to a few
microseconds. Plotted raw, every panel is a vertical line at a tick label with
no information in it. Each axis is therefore drawn as an offset from the
posterior *mean*, in a unit chosen so that one standard deviation is a number
between 1 and 1000 -- hours or minutes for a loosely-known orbital period,
microseconds for a sharply-known epoch. The mean itself moves into the axis
label, printed to enough significant digits that adding it back to a residual
read off the axis lands where the posterior actually is.

The units a parameter is stored in are not the units it is read in. ``PB``
lives in seconds inside the fit but is quoted in days in every parfile;
``TASC`` lives in days, as an MJD, but nobody quotes an epoch uncertainty in
days. :data:`CONVENTIONS` records, per parameter, the unit its centre is quoted
in and the ladder its residuals are measured against.
"""

import dataclasses

import numpy as np


__all__ = [
    "AxisScale",
    "CONVENTIONS",
    "ORBITAL_PARAMETERS",
    "axis_scale",
    "plot_orbit_summary",
]


#: The orbital parameters this summary offers, in the order they are drawn.
#: Whichever of them a given chain explored are the ones that appear.
ORBITAL_PARAMETERS = ("A1", "PB", "TASC", "EPS1", "EPS2")

#: Residual units for a time, largest first, with their size in seconds. The
#: steps are not all decades -- an orbital period known to two hours is read as
#: two hours, not as 7.2 kiloseconds -- so the ladder is walked rather than
#: computed from a logarithm.
TIME_LADDER = (
    (86400.0, "d"),
    (3600.0, "h"),
    (60.0, "min"),
    (1.0, "s"),
    (1e-3, "ms"),
    (1e-6, "µs"),
    (1e-9, "ns"),
)

#: Residual units for a projected semi-major axis, which is a light travel
#: time and is universally quoted as one.
LIGHT_SECOND_LADDER = (
    (1.0, "lt-s"),
    (1e-3, "mlt-s"),
    (1e-6, "µlt-s"),
    (1e-9, "nlt-s"),
)


@dataclasses.dataclass(frozen=True)
class _Convention:
    """How one parameter is stored, and how it should be read.

    Attributes
    ----------
    ladder : tuple or None
        Residual units to choose from. ``None`` means the parameter is
        dimensionless and its residuals get a bare power of ten instead.
    residual_units_per_stored : float
        Size of one stored unit, in the ladder's base unit. ``TASC`` is stored
        in days and read in seconds, so 86400.
    centre_units_per_stored : float
        Size of one stored unit, in the unit the centre is quoted in. ``PB`` is
        stored in seconds and quoted in days, so 1/86400.
    centre_unit : str
        Name of that unit, for the axis label.
    """

    ladder: tuple | None = None
    residual_units_per_stored: float = 1.0
    centre_units_per_stored: float = 1.0
    centre_unit: str = ""


#: Per-parameter reading conventions. A parameter absent from here falls back
#: to the dimensionless treatment, which is always usable if never pretty.
CONVENTIONS = {
    "A1": _Convention(LIGHT_SECOND_LADDER, centre_unit="lt-s"),
    "PB": _Convention(TIME_LADDER, centre_units_per_stored=1 / 86400.0, centre_unit="d"),
    "TASC": _Convention(TIME_LADDER, residual_units_per_stored=86400.0, centre_unit="MJD"),
    "EPS1": _Convention(),
    "EPS2": _Convention(),
}


@dataclasses.dataclass(frozen=True)
class AxisScale:
    """The offset-and-unit view of one parameter's posterior.

    Attributes
    ----------
    parameter : str
        Parameter name.
    centre : float
        Posterior mean, in the parameter's *stored* units.
    scale : float
        Stored units per displayed unit, so that
        ``(samples - centre) / scale`` is what goes on the axis.
    unit : str
        Name of the displayed unit, empty for a dimensionless power of ten.
    label : str
        Two-line axis label: the parameter and its subtracted centre on the
        first, the unit of what is left on the second.
    """

    parameter: str
    centre: float
    scale: float
    unit: str
    label: str

    def apply(self, samples):
        """Put samples, in stored units, onto this axis."""
        return (np.asarray(samples, dtype=float) - self.centre) / self.scale


def _from_ladder(width, ladder):
    """Largest unit of ``ladder`` that one width still covers.

    Falls off the bottom onto the finest unit available, which keeps a
    pathologically sharp posterior plottable rather than crashing it.
    """
    for size, name in ladder:
        if width / size >= 1.0:
            return size, name
    return ladder[-1]


def _power_of_ten(width):
    """Decade below ``width``, so that the width itself reads between 1 and 10."""
    exponent = int(np.floor(np.log10(width)))
    return 10.0**exponent, f"$\\times 10^{{{exponent}}}$"


def _format_centre(centre, resolution):
    """``centre``, with enough significant digits to resolve ``resolution``.

    Fixed-point formatting is unusable across the range here -- an epoch needs
    five digits before the point and seven after, a frequency derivative needs
    neither -- so the precision is set in significant digits and handed to
    ``%g``, which then picks fixed or exponential notation for itself.
    """
    if centre == 0.0 or not np.isfinite(centre):
        return f"{centre:g}"
    digits = int(np.ceil(np.log10(abs(centre)) - np.log10(resolution)))
    return f"{centre:.{np.clip(digits, 1, 17)}g}"


def axis_scale(parameter, samples):
    """Choose the centre and unit that make one parameter's posterior readable.

    Parameters
    ----------
    parameter : str
        Parameter name, used to look up :data:`CONVENTIONS`.
    samples : array-like
        Physical posterior samples, in the parameter's stored units.

    Returns
    -------
    AxisScale
        Centre, unit and label; call :meth:`AxisScale.apply` to place samples.
    """
    samples = np.asarray(samples, dtype=float)
    centre = float(samples.mean())
    width = float(samples.std())

    convention = CONVENTIONS.get(parameter, _Convention())

    if not np.isfinite(width) or width <= 0:
        # Nothing moved, so there is no width to pick a unit from. Stored units
        # and an empty label unit leave the panel drawable and honest.
        scale, unit, unit_text = 1.0, convention.centre_unit, convention.centre_unit
    elif convention.ladder is None:
        size, unit_text = _power_of_ten(width)
        scale, unit = size, ""
    else:
        size, unit = _from_ladder(width * convention.residual_units_per_stored, convention.ladder)
        scale, unit_text = size / convention.residual_units_per_stored, unit

    centre_display = centre * convention.centre_units_per_stored
    # A hundredth of the width: fine enough that the rounding of the centre is
    # invisible against the spread the panel is showing.
    resolution = max(width * convention.centre_units_per_stored, np.finfo(float).tiny) / 100
    sign = "-" if centre_display >= 0 else "+"
    centre_text = _format_centre(abs(centre_display), resolution)
    suffix = f" {convention.centre_unit}" if convention.centre_unit else ""

    label = f"{parameter} {sign} {centre_text}{suffix}"
    if unit_text:
        label += f"\n({unit_text})"

    return AxisScale(parameter=parameter, centre=centre, scale=scale, unit=unit, label=label)


#: Inches per corner panel, and the margins around the block of them. Sized so
#: a two-line axis label fits under the bottom row without being clipped.
PANEL_INCHES = 1.15
LABEL_INCHES = 1.05
ECC_PANEL_INCHES = 3.0
PANEL_GAP_INCHES = 0.6
TOP_INCHES = 0.55
TITLE_INCHES = 0.35


def _corner_axes(fig, n, geometry):
    """A block of ``n`` x ``n`` axes, immune to ``corner``'s own layout call.

    ``corner.corner`` ends by calling ``fig.subplots_adjust``, which would drag
    a second panel drawn beside it out of place -- or rather, would drag the
    corner block over the whole figure and leave the second panel underneath
    it. Axes taken from a ``GridSpec`` with explicit margins ignore the
    figure's subplot parameters entirely, so the call lands on nothing and both
    blocks stay where they were put.
    """
    from matplotlib.gridspec import GridSpec

    grid = GridSpec(n, n, figure=fig, wspace=0.06, hspace=0.06, **geometry)
    return [fig.add_subplot(grid[i, j]) for i in range(n) for j in range(n)]


def plot_orbit_summary(
    samples_by_parameter, fname="orbit.jpg", summary=None, bins=80, label_fontsize=5.5
):
    """Draw the orbital corner plot beside the eccentricity it implies.

    Left, the parameters of :data:`ORBITAL_PARAMETERS` that this chain actually
    explored, in physical units, each axis centred on its posterior mean; see
    :func:`axis_scale` for how the centre and unit are chosen. Right, the
    eccentricity posterior, drawn by
    :func:`ell1fit.eccentricity.draw_eccentricity_posterior` so that it is the
    same panel the standalone plot shows and cannot drift from it.

    Parameters
    ----------
    samples_by_parameter : dict
        ``{parameter: physical samples}``, as
        :func:`ell1fit.eccentricity.physical_samples_from_chain` returns.
        Parameters outside :data:`ORBITAL_PARAMETERS` are ignored.
    fname : str
        Output image path.
    summary : dict, optional
        Output of :func:`ell1fit.eccentricity.eccentricity_summary`, reused for
        the right-hand panel rather than recomputed.
    bins : int
        Histogram bins for the eccentricity panel.
    label_fontsize : float
        Size of the corner axis labels, which carry a subtracted centre and so
        run longer than a bare parameter name.

    Returns
    -------
    str
        ``fname``, for convenience.

    Raises
    ------
    ValueError
        If ``EPS1`` and ``EPS2`` are not both present. Without them there is no
        eccentricity posterior, and this figure is the one that pairs the two;
        the plain corner plot of the fit already covers everything else.
    """
    import corner
    import matplotlib.pyplot as plt

    from .eccentricity import draw_eccentricity_posterior
    from .plotting import plot_style_context

    missing = [par for par in ("EPS1", "EPS2") if par not in samples_by_parameter]
    if missing:
        raise ValueError(
            f"No eccentricity posterior to show: {', '.join(missing)} was not sampled. "
            "This figure pairs the orbital corner plot with the eccentricity, so it is "
            "only drawn when both EPS1 and EPS2 were fitted."
        )

    drawn = [par for par in ORBITAL_PARAMETERS if par in samples_by_parameter]
    scales = [axis_scale(par, samples_by_parameter[par]) for par in drawn]
    columns = np.column_stack(
        [scale.apply(samples_by_parameter[par]) for par, scale in zip(drawn, scales)]
    )

    n = len(drawn)
    # The corner block is laid out in inches and only then converted to figure
    # fractions, so its panels stay square however many parameters there are
    # and however tall the eccentricity beside them makes the figure.
    grid_inches = PANEL_INCHES * n
    corner_block = grid_inches + LABEL_INCHES
    width = corner_block + PANEL_GAP_INCHES + ECC_PANEL_INCHES + 0.3
    height = max(corner_block + TOP_INCHES, ECC_PANEL_INCHES + 1.2)
    grid_bottom = LABEL_INCHES / height
    grid_top = (LABEL_INCHES + grid_inches) / height

    with plot_style_context():
        fig = plt.figure(figsize=(width, height))
        axes = _corner_axes(
            fig,
            n,
            dict(
                left=LABEL_INCHES / width,
                right=corner_block / width,
                bottom=grid_bottom,
                top=grid_top,
            ),
        )
        corner.corner(
            columns,
            labels=[scale.label for scale in scales],
            quantiles=[0.16, 0.5, 0.84],
            fig=fig,
            label_kwargs={"fontsize": label_fontsize},
            max_n_ticks=3,
        )
        for ax in axes:
            ax.tick_params(labelsize=label_fontsize)

        from matplotlib.gridspec import GridSpec

        # Centred on the corner block rather than stretched to it: a two-parameter
        # fit must not produce a letterbox histogram, nor a five-parameter one a
        # column. TITLE_INCHES is the two-line eccentricity summary above it.
        ecc_inches = min(ECC_PANEL_INCHES, grid_inches)
        ecc_middle = (grid_bottom + grid_top) / 2
        right = GridSpec(
            1,
            1,
            figure=fig,
            left=(corner_block + PANEL_GAP_INCHES) / width,
            right=1 - 0.15 / width,
            bottom=ecc_middle - ecc_inches / 2 / height,
            top=ecc_middle + (ecc_inches / 2 - TITLE_INCHES) / height,
        )
        ecc_ax = fig.add_subplot(right[0, 0])
        draw_eccentricity_posterior(
            ecc_ax,
            samples_by_parameter["EPS1"],
            samples_by_parameter["EPS2"],
            summary=summary,
            bins=bins,
        )
        # A small eccentricity gives tick labels like "0.00015", seven characters
        # wide, and the default five of them run into each other in a panel this
        # narrow. Fewer and smaller; the standalone plot, which is wider relative
        # to its labels, is left exactly as it was.
        ecc_ax.locator_params(axis="x", nbins=4)
        ecc_ax.tick_params(labelsize=label_fontsize + 1)

        fig.savefig(fname, dpi=300)
        plt.close(fig)

    return fname
