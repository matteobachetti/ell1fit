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
