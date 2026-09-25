r"""The figure conventions every plot in this package follows.

A journal page is 7 inches wide and its columns are 3.5. A figure that is not
one of those two widths gets rescaled by the typesetter, and rescaling a figure
rescales its text with it: 7-point labels drawn at 4.7 inches arrive at the
reader as 5.2-point labels. So this module offers two widths and nothing else,
as a small table of named presets, and every figure in the package is built from
one of them.

Appearance and geometry are kept apart
--------------------------------------

:data:`PLOT_RC_PARAMS` describes only how a figure *looks* -- text sizes, the
grid, the colour cycle, how fonts are embedded. It deliberately says nothing
about margins. An earlier version of this style fixed ``figure.subplot.left``
and its three siblings to values tuned for one 3.5x3.5-inch corner plot; every
figure of a different shape then had to abandon the entire style to get its
margins right, which is exactly what the orbital-decay plot did.

Margins are instead handled per figure, by passing ``layout="constrained"`` when
the figure is built. Constrained layout shrinks the *axes* until the labels fit
and leaves the *figure* the size it was asked for, which is the property that
makes a 3.5-inch figure still 3.5 inches wide when it lands in the paper. For
the same reason ``savefig.bbox`` is pinned to ``"standard"``: ``"tight"`` crops
the canvas down to its ink, and a figure cropped to its ink is no longer any
particular width.

Corner plots are the exception. :func:`corner.corner` ends by calling
``subplots_adjust`` on whatever figure it is handed, which constrained layout
refuses, and :mod:`ell1fit.orbit_plot` places its panels in absolute inches on
purpose. Those figures keep their manual layout -- which is why the layout
engine is chosen per figure rather than set here.

Reading a figure
----------------

The drawing conventions, which the plotting code follows and which a new figure
should follow too:

* measurements are black points with error bars (:data:`DATA_COLOR`);
* a model is a colour from the cycle -- solid for the one being adopted, dashed
  for the alternative it is being compared against;
* a model's uncertainty is the same colour as a ``fill_between``, at
  :data:`BAND_ALPHA` for one sigma and :data:`WIDE_BAND_ALPHA` for two;
* guides, zero lines and anything else that is not data are grey and dotted
  (:data:`GUIDE_COLOR`);
* the grid is light, dotted, and drawn *behind* the data. On an axis holding an
  image -- a phaseogram, say -- it is turned off instead, by
  :func:`image_axes`: behind a ``pcolormesh`` it cannot be seen, and in front of
  one it is clutter;
* a figure destined for a paper carries no title. A diagnostic may.
"""

import logging
import os

import matplotlib as mpl


__all__ = [
    "BAND_ALPHA",
    "COLUMN_WIDTH",
    "CORNER_LABEL_SIZE",
    "CORNER_SCATTER_NDIM",
    "DATA_COLOR",
    "DEFAULT_FIGURE_FORMAT",
    "FIGURE_FORMATS",
    "FIGURE_SIZES",
    "GUIDE_COLOR",
    "LARGE_FIGURE_FORMAT",
    "MAX_CANVAS_PX",
    "MIN_RASTER_DPI",
    "PLOT_RC_PARAMS",
    "RASTER_DPI",
    "SUMMARY_TITLE_SIZE",
    "TEXT_WIDTH",
    "WIDE_BAND_ALPHA",
    "add_figure_format_argument",
    "corner_plot_kwargs",
    "current_figure_format",
    "figure_path",
    "figure_size",
    "fitted_raster_dpi",
    "image_axes",
    "plot_style_context",
    "save_figure",
    "set_figure_format",
]


#: Width of one journal column, in inches.
COLUMN_WIDTH = 3.5

#: Width of the full text block, in inches.
TEXT_WIDTH = 7.0

#: The figure shapes this package draws, by name. Every width here is either
#: :data:`COLUMN_WIDTH` or :data:`TEXT_WIDTH`; only the height varies, and it
#: varies over a handful of aspect ratios rather than continuously, so that two
#: figures side by side in a paper look like a pair.
FIGURE_SIZES = {
    #: The default: one panel, 4:3, filling a column.
    "column": (COLUMN_WIDTH, 2.65),
    #: A histogram or anything else whose natural aspect is square.
    "column-square": (COLUMN_WIDTH, COLUMN_WIDTH),
    #: A data panel over a residual panel, still inside one column.
    "column-tall": (COLUMN_WIDTH, 4.2),
    #: Landscape across the text width, for a figure with many details.
    "wide": (TEXT_WIDTH, COLUMN_WIDTH),
    #: Two stacked panels across the text width.
    "wide-tall": (TEXT_WIDTH, 5.25),
    #: Square across the text width: corner plots, phaseogram grids.
    "page": (TEXT_WIDTH, TEXT_WIDTH),
}

#: Base text size, in points. Everything else is quoted relative to it.
BASE_FONT_SIZE = 7.0

#: Corner-plot axis labels carry a subtracted centre as well as a parameter
#: name, so they run long and are set smaller than the rest.
CORNER_LABEL_SIZE = BASE_FONT_SIZE - 1.5

#: The two-line numerical summary printed above a posterior panel.
SUMMARY_TITLE_SIZE = BASE_FONT_SIZE - 2.0

#: Resolution used whenever a figure is written in a raster format.
RASTER_DPI = 300


#: What a figure too large to be written as vector falls back to. JPEG rather
#: than PNG because these are diagnostics read by zooming in, where a file half
#: the size matters more than the artifacts described under "Output format" in
#: ``docs/ell1fit/figures.rst``.
LARGE_FIGURE_FORMAT = "jpg"

#: Longest side, in pixels, a raster canvas is allowed to reach before the
#: resolution is reduced to fit. It is what bounds the cost of a figure whose
#: size nobody chose; at :data:`RASTER_DPI` it leaves every ordinary figure --
#: and every corner plot up to about twelve parameters -- untouched.
MAX_CANVAS_PX = 8000

#: The resolution :func:`fitted_raster_dpi` will not go below, whatever
#: :data:`MAX_CANVAS_PX` would ask for. The smallest text in these figures is
#: 7 pt, which at 150 dpi renders about 15 pixels tall -- the size of ordinary
#: screen text, and checked by eye to be clearly legible. Below that the axis
#: labels start to go, and an illegible diagnostic is not worth the memory it
#: saves.
MIN_RASTER_DPI = 150

#: Number of parameters above which a corner plot stops drawing the individual
#: samples behind its contours.
#:
#: This is the whole of the size problem. A corner plot grows by about 2.1
#: inches per parameter in each direction, and :func:`corner.corner` marks the
#: scatter of individual samples in every one of its ``n(n-1)/2`` panels as
#: rasterized. A vector back-end honours that by allocating one *whole-canvas*
#: pixel buffer per panel, so the cost grows as the square of the parameter
#: count: measured peak memory for a twelve-parameter fit was 6.7 GB, which is
#: what a cluster run died of. Without the scatter the same figure has no
#: rasterized element at all, needs no canvas, and costs 0.45 GB.
#:
#: Eight is where a corner plot also stops being a figure anyone would place in
#: a paper -- it is 18 inches on a side -- and where the point cloud stops
#: being readable anyway: each panel is by then small enough that the samples
#: are a grey smudge under the contours that already describe them.
CORNER_SCATTER_NDIM = 8

#: Measurements.
DATA_COLOR = "black"

#: Zero lines, reference levels, and anything else that is not data.
GUIDE_COLOR = "0.5"

#: Alpha for a one-sigma band, and for the two-sigma band drawn under it.
BAND_ALPHA = 0.30
WIDE_BAND_ALPHA = 0.15

#: Okabe-Ito, minus its yellow, which is unreadable on white. Chosen over
#: matplotlib's ``tab10`` because it stays distinguishable both for a reader
#: with a red/green deficiency and in greyscale print. Pure black is left out
#: so that ``C0`` and friends never collide with :data:`DATA_COLOR`.
COLOR_CYCLE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8C8C8C",  # grey
)


PLOT_RC_PARAMS = {
    # Text. One size for everything the reader has to read off an axis; the
    # legend a shade smaller, since it is read once rather than scanned.
    "font.size": BASE_FONT_SIZE,
    "font.family": "sans-serif",
    "axes.labelsize": BASE_FONT_SIZE,
    "axes.titlesize": BASE_FONT_SIZE,
    "xtick.labelsize": BASE_FONT_SIZE,
    "ytick.labelsize": BASE_FONT_SIZE,
    "legend.fontsize": BASE_FONT_SIZE - 0.5,
    "legend.title_fontsize": BASE_FONT_SIZE - 0.5,
    "figure.titlesize": BASE_FONT_SIZE + 1,
    # Set explicitly so that $\Delta$TASC and $\sigma$ are drawn in the same
    # family as the words around them rather than in matplotlib's own default.
    "mathtext.fontset": "dejavusans",
    # Ticks: short, inward, on all four spines.
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    # Grid: on the major ticks only -- a grid on the minor ticks as well, now
    # that they are drawn, is a mesh -- and underneath the data, so that a line
    # of it never crosses a marker or an error bar.
    "axes.grid": True,
    "axes.grid.axis": "both",
    "axes.grid.which": "major",
    "axes.axisbelow": True,
    "grid.color": "grey",
    "grid.linewidth": 0.3,
    "grid.linestyle": ":",
    # Marks.
    "axes.linewidth": 0.6,
    "axes.prop_cycle": mpl.cycler(color=list(COLOR_CYCLE)),
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "errorbar.capsize": 1.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.3,
    # Output. ``savefig.bbox`` is pinned rather than left to the user's own
    # matplotlibrc: "tight" would crop the canvas to its ink and the figure
    # would no longer be a column wide. Type 42 embeds fonts as TrueType;
    # several journals reject the Type 3 that matplotlib emits by default.
    "figure.figsize": FIGURE_SIZES["column"],
    "figure.dpi": RASTER_DPI,
    "savefig.dpi": RASTER_DPI,
    "savefig.bbox": "standard",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


#: Formats a figure may be written in.
FIGURE_FORMATS = ("pdf", "png", "svg", "eps", "jpg")

#: What is used when nothing says otherwise. Vector, so that text stays sharp
#: at whatever size the journal ends up placing the figure.
DEFAULT_FIGURE_FORMAT = "pdf"

#: Environment variable consulted when no format has been set explicitly.
FIGURE_FORMAT_ENV_VAR = "ELL1FIT_FIGURE_FORMAT"

#: Extensions recognised as "the caller already said what they wanted".
#: Deliberately wider than :data:`FIGURE_FORMATS`: we refuse to *choose* a
#: format we do not endorse, but we honour one that is asked for by name.
_KNOWN_EXTENSIONS = frozenset(
    {".pdf", ".png", ".svg", ".eps", ".ps", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
)

_figure_format = None


def plot_style_context():
    """Return a matplotlib rc_context using the project plotting style.

    Nothing is set globally: importing this module leaves a caller's own
    matplotlib configuration untouched, and the style applies only inside the
    ``with`` block.
    """
    return mpl.rc_context(PLOT_RC_PARAMS)


def figure_size(preset):
    """Return ``(width, height)`` in inches for one of :data:`FIGURE_SIZES`.

    Parameters
    ----------
    preset : str
        A key of :data:`FIGURE_SIZES`, e.g. ``"column"`` or ``"wide-tall"``.

    Returns
    -------
    tuple of float
        Width and height in inches, ready for ``figsize=``.
    """
    try:
        width, height = FIGURE_SIZES[preset]
    except KeyError:
        raise ValueError(
            f"Unknown figure preset {preset!r}. Available: {', '.join(sorted(FIGURE_SIZES))}"
        ) from None
    return (width, height)


def set_figure_format(fmt):
    """Choose the format figures are written in for the rest of this run.

    Called once by each command-line entry point, from ``--figure-format``.
    Passing ``None`` clears the choice, so that
    :data:`FIGURE_FORMAT_ENV_VAR` and then :data:`DEFAULT_FIGURE_FORMAT` decide
    again.
    """
    global _figure_format
    if fmt is not None:
        fmt = _validate_format(fmt, source="figure format")
    _figure_format = fmt
    return fmt


def current_figure_format():
    """The format figures are written in, absent a per-call override.

    An explicit :func:`set_figure_format` wins; failing that the
    :data:`FIGURE_FORMAT_ENV_VAR` environment variable; failing that
    :data:`DEFAULT_FIGURE_FORMAT`.
    """
    if _figure_format is not None:
        return _figure_format
    from_env = os.environ.get(FIGURE_FORMAT_ENV_VAR)
    if from_env:
        return _validate_format(from_env, source=FIGURE_FORMAT_ENV_VAR)
    return DEFAULT_FIGURE_FORMAT


def _validate_format(fmt, source):
    fmt = str(fmt).strip().lower().lstrip(".")
    if fmt not in FIGURE_FORMATS:
        raise ValueError(f"Unsupported {source} {fmt!r}. Available: {', '.join(FIGURE_FORMATS)}")
    return fmt


def figure_path(path, fmt=None):
    """Resolve where a figure should be written.

    ``path`` is normally an output root with no extension, and gets the run's
    current format appended. If it already ends in an image extension, that is
    taken as the caller having said what they want -- the public plotting
    functions take an ``fname`` argument, and somebody who asks for a ``.png``
    by name gets a ``.png``.

    Parameters
    ----------
    path : str
        Output root, or a complete file name.
    fmt : str or None
        Format to use for a bare root, overriding
        :func:`current_figure_format`.

    Returns
    -------
    str
        The path to write.
    """
    if os.path.splitext(path)[1].lower() in _KNOWN_EXTENSIONS:
        return path
    fmt = _validate_format(fmt, source="figure format") if fmt else current_figure_format()
    return f"{path}.{fmt}"


#: Formats there is no point retrying a failed save in: JPEG is what the
#: fallback writes, and PNG holds the same uncompressed canvas in memory.
_NO_CHEAPER_FALLBACK = frozenset({".jpg", ".jpeg", ".png"})


def _save_smaller(fig, fname, dpi, kwargs):
    """Rewrite a figure that ran out of memory as :data:`LARGE_FIGURE_FORMAT`.

    Called only from :func:`save_figure`'s ``except MemoryError`` branch, so a
    bare ``raise`` here re-raises that error with its original traceback.
    """
    root, ext = os.path.splitext(fname)
    if ext.lower() in _NO_CHEAPER_FALLBACK:
        raise
    fallback = f"{root}.{LARGE_FIGURE_FORMAT}"
    logging.warning(
        f"Ran out of memory writing {fname}: a vector canvas needs one whole-figure "
        f"pixel buffer per rasterized element. Falling back to {fallback}. It stays "
        f"readable zoomed in, and `img2pdf {os.path.basename(fallback)} -o "
        f"{os.path.basename(root)}.pdf` wraps it back into a PDF if one is needed."
    )
    # The partial file is not a figure and must not be left looking like one:
    # anything downstream globbing for the output would pick it up.
    if os.path.exists(fname):
        os.remove(fname)
    fig.savefig(fallback, dpi=dpi, **kwargs)
    return fallback


def fitted_raster_dpi(fig):
    """The resolution ``fig`` can be rasterized at without an unbounded canvas.

    :data:`RASTER_DPI` for anything of a size somebody chose -- every figure
    built from :func:`figure_size` is orders of magnitude below the ceiling.
    It only bites on a figure whose size is a consequence rather than a
    decision, which in this package means a corner plot: it grows by about
    2.1 inches per parameter in each direction, so a twenty-epoch fit asks for
    a canvas 13000 pixels on a side.

    The resolution is then reduced continuously to hold the canvas at
    :data:`MAX_CANVAS_PX`, and never below :data:`MIN_RASTER_DPI` -- past that
    point the labels stop being legible, and a figure nobody can read is not a
    cheaper figure but a wasted one.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure about to be written.

    Returns
    -------
    float
        Resolution in dots per inch.
    """
    side = max(fig.get_size_inches())
    if side <= 0:
        return RASTER_DPI
    dpi = min(RASTER_DPI, max(MIN_RASTER_DPI, MAX_CANVAS_PX / side))
    if dpi < RASTER_DPI:
        logging.info(
            f"Figure is {side:.0f} inches on its longer side: writing it at {dpi:.0f} dpi "
            f"instead of {RASTER_DPI:.0f}, to hold the canvas near {MAX_CANVAS_PX} pixels."
        )
    return dpi


def corner_plot_kwargs(ndim):
    """Extra arguments :func:`corner.corner` should be called with for ``ndim``.

    Empty up to :data:`CORNER_SCATTER_NDIM`. Above it the per-sample scatter
    is turned off, for the reasons on that constant. A caller's own keywords
    are applied after these, so an explicit choice still wins.

    Parameters
    ----------
    ndim : int
        Number of parameters the corner plot covers.

    Returns
    -------
    dict
        Keywords to pass before the caller's own.
    """
    if ndim <= CORNER_SCATTER_NDIM:
        return {}
    logging.info(
        f"Corner plot has {ndim} parameters (over {CORNER_SCATTER_NDIM}): not drawing "
        "the individual samples behind the contours, which is what makes it expensive."
    )
    return {"plot_datapoints": False}


def save_figure(fig, path, fmt=None, dpi=None, **kwargs):
    """Write a figure and close it.

    Closing here rather than at the call site is not tidiness: two of the
    corner-plot writers used to leak a figure per call, and one of them fires
    every thousand sampler iterations.

    ``dpi`` is passed explicitly rather than left to the rc, so that a figure
    saved outside :func:`plot_style_context` still comes out at
    :data:`RASTER_DPI` instead of matplotlib's 100. Left unset it comes from
    :func:`fitted_raster_dpi`, which is :data:`RASTER_DPI` for every figure of
    a size somebody chose and lower only for one that outgrew the ceiling. It
    is *not* ignored by the vector formats: a rasterized element inside a PDF
    is rendered at this resolution too.

    A figure large enough to exhaust memory while being written falls back to
    :data:`LARGE_FIGURE_FORMAT` rather than taking the run down with it. A
    whole fit used to be lost at the last step, after the sampling was over and
    the chain was safely on disk, because the final corner plot could not be
    drawn. :func:`corner_plot_kwargs` and :func:`fitted_raster_dpi` are meant
    to keep that from arising; this is the net underneath them, for the figure
    nobody predicted.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to write.
    path : str
        Output root or complete file name; see :func:`figure_path`.
    fmt : str or None
        Format for a bare root.
    dpi : float or None
        Resolution for raster formats. Defaults to :data:`RASTER_DPI`.

    Returns
    -------
    str
        The path written -- which is not always the path asked for, if the
        fallback above fired. Callers log it rather than reconstructing it.
    """
    import matplotlib.pyplot as plt

    fname = figure_path(path, fmt=fmt)
    dpi = fitted_raster_dpi(fig) if dpi is None else dpi
    try:
        fig.savefig(fname, dpi=dpi, **kwargs)
    except MemoryError:
        fname = _save_smaller(fig, fname, dpi, kwargs)
    finally:
        # In ``finally`` rather than after the ``try``: a save that fails for
        # good would otherwise leak the very figure this function closes on
        # everyone's behalf, and a figure that large is the one that can least
        # afford to be held.
        plt.close(fig)
    return fname


def image_axes(ax):
    """Mark an axis as holding an image, and drop its grid.

    A grid behind a ``pcolormesh`` or an ``imshow`` cannot be seen, and one in
    front of it is clutter over the data. Returns ``ax``, so it can be applied
    inline.
    """
    ax.grid(False)
    return ax


def add_figure_format_argument(parser):
    """Add ``--figure-format`` to a command-line parser.

    The entry point is expected to pass the parsed value straight to
    :func:`set_figure_format`; ``None`` there means "nothing was asked for",
    which leaves the environment variable and the default in charge.
    """
    parser.add_argument(
        "--figure-format",
        choices=FIGURE_FORMATS,
        default=None,
        help=(
            f"Format for every figure this command writes. Default: "
            f"{DEFAULT_FIGURE_FORMAT}, or ${FIGURE_FORMAT_ENV_VAR} if it is set. "
            f"pdf, svg and eps are vector, so their text stays sharp at whatever "
            f"size the figure is printed; png and jpg are written at "
            f"{RASTER_DPI} dpi."
        ),
    )
    return parser
