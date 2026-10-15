Figures
=======

Several of the figures ``ell1fit`` writes are meant to end up in a paper, and
the rest are read alongside them, so they are all drawn to one set of
conventions. This page records them, so that a new figure can follow them and
an old one can be checked against them.

Everything here is implemented in :mod:`ell1fit.plotting`, and every figure in
the package is drawn inside :func:`ell1fit.plotting.plot_style_context`.

Two widths, and nothing between them
------------------------------------

A journal page is 7 inches wide and its columns are 3.5. A figure that is not
one of those widths gets rescaled by the typesetter, and rescaling a figure
rescales its text with it: 7-point labels drawn on a 4.7-inch canvas arrive at
the reader as 5.2-point labels, which is why a figure that looked fine on
screen can be unreadable in print.

So :data:`ell1fit.plotting.FIGURE_SIZES` offers exactly two widths, and a
handful of heights against each:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Preset
     - Inches
     - For
   * - ``"column"``
     - 3.5 × 2.65
     - The default. One panel, 4:3, filling a column.
   * - ``"column-square"``
     - 3.5 × 3.5
     - A histogram, or anything whose natural aspect is square.
   * - ``"column-tall"``
     - 3.5 × 4.2
     - A data panel over a residual panel, still inside one column.
   * - ``"wide"``
     - 7.0 × 3.5
     - Landscape across the text width, for a figure with many details.
   * - ``"wide-tall"``
     - 7.0 × 5.25
     - Two stacked panels across the text width.
   * - ``"page"``
     - 7.0 × 7.0
     - Square across the text width: phaseogram grids.

Call :func:`ell1fit.plotting.figure_size` rather than writing a pair of numbers
into a ``figsize=``. Heights vary over a short list of aspect ratios rather
than continuously, so that two figures placed side by side in a paper look like
a pair rather than like two unrelated pictures.

Why ``constrained_layout``, and not ``bbox="tight"``
----------------------------------------------------

Margins are set per figure, by passing ``layout="constrained"`` where the
figure is built. Constrained layout shrinks the *axes* until the labels fit and
leaves the *figure* the size it was asked for — which is the property that makes
a 3.5-inch figure still 3.5 inches wide in the file.

``savefig(bbox="tight")`` does the opposite: it crops the canvas down to its
ink, so the saved figure is whatever width the labels happened to need. The
style pins ``savefig.bbox`` to ``"standard"`` so that a stray setting in a
user's own ``matplotlibrc`` cannot reintroduce it.

This is also why the style itself says nothing about margins. An earlier
version of it fixed ``figure.subplot.left`` and its three siblings to values
tuned for one 3.5 × 3.5-inch corner plot; every figure of a different shape
then had to abandon the entire style to get its margins right, which is exactly
what the orbital-decay plot did.

Corner plots are the exception, twice over. :func:`corner.corner` ends by
calling ``subplots_adjust`` on whatever figure it is handed, which constrained
layout refuses, and :mod:`ell1fit.orbit_plot` places its panels in absolute
inches on purpose. They also size themselves from the number of parameters
rather than from a preset, because squeezing ten parameters into 7 inches gives
0.7-inch panels that no font size can rescue. Those figures keep their manual
layout and their own size, and are scaled in the document instead.

Text
----

7 point everywhere the reader has to read a number off an axis — labels, ticks,
titles — with legends half a point smaller and figure titles one point larger.
Sans-serif, with ``mathtext.fontset`` set explicitly so that ``$\Delta$TASC``
and ``$\sigma$`` are drawn in the same family as the words beside them instead
of falling back to matplotlib's own maths font.

Fonts are embedded as TrueType (``pdf.fonttype = 42``). Matplotlib's default is
Type 3, which several journals reject outright.

Grid and ticks
--------------

Light grey dotted, on the major ticks only, and **behind** the data, so that a
grid line never crosses a marker or an error bar. Ticks are short, inward, and
on all four spines.

On an axis holding an image — a phaseogram, say — the grid is turned off
instead, with :func:`ell1fit.plotting.image_axes`: behind a ``pcolormesh`` it
cannot be seen, and in front of one it is clutter over the data.

Reading a figure
----------------

The drawing conventions, which a new figure should follow:

- measurements are black points with error bars
  (:data:`ell1fit.plotting.DATA_COLOR`);
- a model is a colour from the cycle — solid for the one being adopted, dashed
  for the alternative it is being compared against;
- a model's uncertainty is the same colour as a ``fill_between``, at
  :data:`ell1fit.plotting.BAND_ALPHA` for one sigma and
  :data:`ell1fit.plotting.WIDE_BAND_ALPHA` for two;
- guides, zero lines and anything else that is not data are grey and dotted
  (:data:`ell1fit.plotting.GUIDE_COLOR`);
- a figure destined for a paper carries no title. A diagnostic may — the
  weighting figure puts its input file name in one, and is the only figure that
  does.

The colour cycle is Okabe–Ito rather than matplotlib's ``tab10``: it stays
distinguishable both for a reader with a red/green deficiency and in greyscale
print. Its yellow is dropped, being unreadable on white, and pure black is left
out so that ``C0`` and its successors never collide with the colour reserved
for data.

Output format
-------------

PDF by default — vector, so the text stays sharp at whatever size the figure is
finally placed. ``--figure-format`` on ``ell1fit``, ``ell1decay`` and
``ell1ecc`` selects another; so does the ``ELL1FIT_FIGURE_FORMAT`` environment
variable, which the flag overrides. Raster formats are written at 300 dpi.

Figures used to be written as JPEG, which is a lossy format built for
photographs: it puts ringing and blocking artifacts around thin black lines and
small text, which is what every figure here is made of.

A caller who names a file explicitly still gets what they asked for. The
public plotting functions take an ``fname`` argument, and
:func:`ell1fit.plotting.figure_path` leaves a path that already carries an image
extension alone; only a bare output root picks up the run's format.

Large corner plots are the exception
------------------------------------

Every other figure in the package is a size somebody chose. A corner plot is
not: it grows by about 2.1 inches per parameter in each direction, and with a
per-file ``Phase_i`` nuisance parameter the parameter count grows with the
number of observations. A twenty-epoch fit is a fifty-inch figure.

That is not merely large. :func:`corner.corner` marks the scatter of individual
samples in each of its ``n(n-1)/2`` panels as rasterized, and a vector back-end
honours that by allocating one *whole-canvas* pixel buffer per panel at 300 dpi
— so the cost grows as the square of the parameter count. A twelve-parameter
fit needs 6.7 GB to write a PDF, which is what a cluster run once died of, at
the very last step, after the sampling was over and the chain was safely on
disk.

So above :data:`ell1fit.plotting.CORNER_SCATTER_NDIM` parameters the scatter is
not drawn. That is the whole fix: with no rasterized element left, the figure
needs no canvas, and the vector back-end goes back to being the cheapest option
as well as the most readable one. Measured, for 10000 samples:

==========  ==============  ==============  ==============
Parameters  PDF, scatter    PDF, no         JPEG, no
            on              scatter         scatter
==========  ==============  ==============  ==============
12          6.7 GB          0.45 GB         0.91 GB
20          —               0.67 GB         1.14 GB
28          —               0.98 GB         1.60 GB
==========  ==============  ==============  ==============

Nothing is lost by it. At this many parameters each panel is small enough that
the point cloud is a grey smudge under the contours that already describe it,
and the figure has long stopped being something anyone would place in a paper.

Two guards sit underneath, for the figure nobody predicted:

* :func:`ell1fit.plotting.fitted_raster_dpi` holds any raster canvas at
  :data:`ell1fit.plotting.MAX_CANVAS_PX` pixels on its longer side, lowering the
  resolution continuously to fit but never below
  :data:`ell1fit.plotting.MIN_RASTER_DPI`. The floor is where legibility goes:
  the smallest text here is 7 pt, which at 150 dpi renders about 15 pixels tall,
  the size of ordinary screen text. It is a no-op for every figure built from a
  preset, and bites only on a raster-format run of an oversized figure.
* :func:`ell1fit.plotting.save_figure` catches a ``MemoryError`` from any save,
  warns, and retries in JPEG rather than taking the run down. ``img2pdf`` wraps
  the result back into a PDF if one is needed.

One limit neither guard covers: a PDF page cannot exceed 200 inches, which a
corner plot reaches at about 94 parameters.
