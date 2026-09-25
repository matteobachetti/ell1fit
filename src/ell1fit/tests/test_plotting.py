r"""The shared figure conventions: sizes, output format, and the style itself.

These are the guarantees the rest of the package's plotting code leans on, and
each of them has a way of quietly breaking. A preset that drifts from the
journal widths it is named after, a ``bbox="tight"`` creeping back into the
style and cropping a 3.5-inch figure to something else, a grid that goes back to
being drawn over the data -- none of those raise, and none of them are visible
in a test that only checks that a file appeared. So they are checked here by
name.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..plotting import (
    COLUMN_WIDTH,
    CORNER_RASTER_NDIM,
    DEFAULT_FIGURE_FORMAT,
    FIGURE_FORMATS,
    FIGURE_SIZES,
    LARGE_FIGURE_FORMAT,
    PLOT_RC_PARAMS,
    TEXT_WIDTH,
    corner_figure_format,
    current_figure_format,
    figure_path,
    figure_size,
    image_axes,
    plot_style_context,
    save_figure,
    set_figure_format,
)


@pytest.fixture
def clean_format(monkeypatch):
    """Run with no format override in force, from either source."""
    monkeypatch.delenv("ELL1FIT_FIGURE_FORMAT", raising=False)
    set_figure_format(None)
    yield
    set_figure_format(None)


class TestFigureSizes:
    def test_every_preset_is_one_of_the_two_journal_widths(self):
        for name, (width, _) in FIGURE_SIZES.items():
            assert width in (COLUMN_WIDTH, TEXT_WIDTH), f"{name} is neither a column nor a page"

    def test_no_preset_is_taller_than_a_page(self):
        # A figure taller than the text width is taller than it is wide by more
        # than any single journal figure has any business being.
        for name, (_, height) in FIGURE_SIZES.items():
            assert height <= TEXT_WIDTH, f"{name} is taller than the text width"

    def test_lookup_returns_a_copy_that_cannot_corrupt_the_table(self):
        size = figure_size("column")
        assert size == FIGURE_SIZES["column"]
        assert size is not FIGURE_SIZES["column"]

    def test_unknown_preset_names_the_ones_that_exist(self):
        with pytest.raises(ValueError, match="column-square"):
            figure_size("colunm")


class TestFigureFormat:
    def test_defaults_to_pdf(self, clean_format):
        assert current_figure_format() == "pdf"

    def test_environment_overrides_the_default(self, clean_format, monkeypatch):
        monkeypatch.setenv("ELL1FIT_FIGURE_FORMAT", "png")
        assert current_figure_format() == "png"

    def test_an_explicit_setting_overrides_the_environment(self, clean_format, monkeypatch):
        monkeypatch.setenv("ELL1FIT_FIGURE_FORMAT", "png")
        set_figure_format("svg")
        assert current_figure_format() == "svg"

    def test_setting_none_hands_control_back(self, clean_format):
        set_figure_format("png")
        set_figure_format(None)
        assert current_figure_format() == "pdf"

    def test_an_unsupported_format_is_refused_where_it_is_set(self, clean_format):
        with pytest.raises(ValueError, match="tiff"):
            set_figure_format("tiff")

    def test_an_unsupported_format_in_the_environment_is_refused(self, clean_format, monkeypatch):
        monkeypatch.setenv("ELL1FIT_FIGURE_FORMAT", "tiff")
        with pytest.raises(ValueError, match="ELL1FIT_FIGURE_FORMAT"):
            current_figure_format()


class TestFigurePath:
    def test_a_bare_root_gets_the_current_format(self, clean_format):
        assert figure_path("out_corner") == "out_corner.pdf"

    def test_an_explicit_format_wins(self, clean_format):
        assert figure_path("out_corner", fmt="png") == "out_corner.png"

    def test_a_caller_supplied_extension_is_left_alone(self, clean_format):
        # Public plotting functions take ``fname=``; somebody who asks for a
        # .png by name gets a .png, whatever the run-wide default is.
        assert figure_path("out_corner.png") == "out_corner.png"
        assert figure_path("out_corner.jpg", fmt="pdf") == "out_corner.jpg"

    def test_a_dot_in_the_root_is_not_an_extension(self, clean_format):
        # Output roots are built from observation ids, which have dots in them.
        assert figure_path("nu30401_v1.2_corner") == "nu30401_v1.2_corner.pdf"


class TestCornerFigureFormat:
    """Corner plots are the one figure whose size is not chosen by anybody.

    It grows with the number of fitted parameters -- which, with a per-file
    ``Phase_i``, grows with the number of observations -- until the vector
    back-ends cannot write it.
    """

    def test_a_small_corner_plot_is_left_to_the_runs_own_format(self, clean_format):
        """Nothing changes for a fit of the size anyone would put in a paper."""
        assert corner_figure_format(CORNER_RASTER_NDIM) is None
        assert corner_figure_format(2) is None

    def test_a_large_corner_plot_is_written_as_a_raster(self, clean_format):
        """Past the limit the format is forced, whatever the run was asked for."""
        set_figure_format("pdf")
        assert corner_figure_format(CORNER_RASTER_NDIM + 1) == LARGE_FIGURE_FORMAT
        assert corner_figure_format(30) == LARGE_FIGURE_FORMAT

    def test_the_forced_format_is_one_the_package_endorses(self):
        # It is handed straight to ``figure_path``, which refuses anything else.
        assert LARGE_FIGURE_FORMAT in FIGURE_FORMATS

    @pytest.mark.parametrize(
        "ndim, suffix", [(4, "." + DEFAULT_FIGURE_FORMAT), (12, "." + LARGE_FIGURE_FORMAT)]
    )
    def test_the_corner_writer_actually_asks(
        self, ndim, suffix, tmp_path, clean_format, monkeypatch
    ):
        """``plot_mcmc_results`` sizes its format by the width of the samples.

        The limit above is worth nothing if the writer that blows up forgets to
        consult it, so the wiring is pinned rather than the policy. ``corner``
        is stubbed out because drawing the real 26-inch figure is the expensive
        thing this whole change exists to avoid.
        """
        from .. import mcmc_utils

        monkeypatch.setattr(mcmc_utils.corner, "corner", lambda *a, **k: plt.figure())
        path = mcmc_utils.plot_mcmc_results(
            flat_samples=np.zeros((10, ndim)), fname=os.path.join(str(tmp_path), "corner")
        )
        assert path.endswith(suffix)


class TestSaveFigure:
    def test_writes_the_file_and_returns_its_path(self, tmp_path, clean_format):
        fig = plt.figure()
        path = save_figure(fig, os.path.join(str(tmp_path), "fig"))
        assert path == os.path.join(str(tmp_path), "fig.pdf")
        assert os.path.getsize(path) > 0

    def test_closes_the_figure(self, tmp_path, clean_format):
        # Two of the corner-plot writers used to leak a figure per call, and one
        # of them fires every thousand sampler iterations.
        fig = plt.figure()
        save_figure(fig, os.path.join(str(tmp_path), "fig"))
        assert fig.number not in plt.get_fignums()

    def test_honours_a_per_call_format(self, tmp_path, clean_format):
        fig = plt.figure()
        path = save_figure(fig, os.path.join(str(tmp_path), "fig"), fmt="png")
        assert path.endswith(".png")
        assert os.path.getsize(path) > 0

    def test_a_figure_too_large_for_its_format_falls_back_to_a_raster(self, tmp_path, clean_format):
        """A save that runs out of memory writes a JPEG instead of taking the run down.

        A whole fit used to be lost at the last step, after the chain was safely
        on disk, because the final corner plot could not be drawn.
        """
        fig = plt.figure()
        real_savefig = fig.savefig
        attempted = []

        def savefig(fname, *args, **kwargs):
            attempted.append(fname)
            if fname.endswith(".pdf"):
                raise MemoryError("std::bad_alloc")
            return real_savefig(fname, *args, **kwargs)

        fig.savefig = savefig
        path = save_figure(fig, os.path.join(str(tmp_path), "fig"))

        assert path == os.path.join(str(tmp_path), "fig.jpg")
        assert os.path.getsize(path) > 0
        assert [os.path.basename(p) for p in attempted] == ["fig.pdf", "fig.jpg"]
        # A half-written PDF is not a figure, and anything globbing the output
        # directory would treat it as one.
        assert not os.path.exists(os.path.join(str(tmp_path), "fig.pdf"))

    def test_a_raster_that_runs_out_of_memory_is_not_retried(self, tmp_path, clean_format):
        """Falling back from JPEG to JPEG would only fail again, more slowly."""
        fig = plt.figure()

        def savefig(fname, *args, **kwargs):
            raise MemoryError("std::bad_alloc")

        fig.savefig = savefig
        with pytest.raises(MemoryError):
            save_figure(fig, os.path.join(str(tmp_path), "fig"), fmt="jpg")
        # A save that fails for good must still not leak the figure -- and a
        # figure this large is the one that can least afford to be held.
        assert fig.number not in plt.get_fignums()

    def test_keeps_the_figure_exactly_the_size_it_was_built(self, tmp_path, clean_format):
        # The whole point of the presets: a figure declared 3.5 inches wide has
        # to still be 3.5 inches wide in the file, or it will not fill a column.
        # ``savefig(bbox="tight")`` would crop it to its ink instead.
        with plot_style_context():
            fig = plt.figure(figsize=figure_size("column"), layout="constrained")
            ax = fig.add_subplot()
            ax.plot([0, 1], [0, 1])
            ax.set_xlabel("a label long enough to need room")
            path = save_figure(fig, os.path.join(str(tmp_path), "fig"), fmt="png")

        with plt.rc_context({"figure.dpi": 100}):
            image = plt.imread(path)
        height, width = image.shape[:2]
        dpi = PLOT_RC_PARAMS["savefig.dpi"]
        assert width == pytest.approx(figure_size("column")[0] * dpi, abs=1)
        assert height == pytest.approx(figure_size("column")[1] * dpi, abs=1)


class TestStyle:
    def test_the_grid_is_drawn_behind_the_data(self):
        assert PLOT_RC_PARAMS["axes.axisbelow"] is True

    def test_the_style_carries_no_fixed_margins(self):
        # Hard-coded ``figure.subplot.*`` margins are what forced the orbital
        # decay plot to abandon the shared style: they were tuned for one
        # 3.5x3.5 figure and are wrong for every other shape.
        assert not [key for key in PLOT_RC_PARAMS if key.startswith("figure.subplot.")]

    def test_the_style_does_not_crop_saved_figures(self):
        assert PLOT_RC_PARAMS.get("savefig.bbox") in (None, "standard")

    def test_the_style_does_not_turn_on_constrained_layout_globally(self):
        # ``corner`` calls ``subplots_adjust`` on whatever figure it is handed,
        # which constrained layout refuses; the corner-based figures place their
        # axes by hand. So the layout engine is chosen per figure.
        assert not PLOT_RC_PARAMS.get("figure.constrained_layout.use", False)

    def test_text_is_uniform_and_small_enough_for_a_column(self):
        base = PLOT_RC_PARAMS["font.size"]
        assert base <= 7
        for key in ("axes.labelsize", "xtick.labelsize", "ytick.labelsize"):
            assert PLOT_RC_PARAMS[key] == base
        assert PLOT_RC_PARAMS["legend.fontsize"] <= base

    def test_maths_is_set_to_match_the_text(self):
        assert PLOT_RC_PARAMS["mathtext.fontset"].startswith("dejavusans")

    def test_fonts_embed_as_truetype(self):
        # Several journals reject Type 3 fonts outright.
        assert PLOT_RC_PARAMS["pdf.fonttype"] == 42
        assert PLOT_RC_PARAMS["ps.fonttype"] == 42

    def test_the_context_actually_applies_the_parameters(self):
        with plot_style_context():
            assert plt.rcParams["font.size"] == PLOT_RC_PARAMS["font.size"]
            assert plt.rcParams["axes.axisbelow"] is True

    def test_the_context_restores_what_it_found(self):
        before = plt.rcParams["font.size"]
        with plot_style_context():
            pass
        assert plt.rcParams["font.size"] == before

    def test_the_colour_cycle_avoids_pure_red_and_green(self):
        # Okabe-Ito: safe to read for the ~8% of male readers with a red/green
        # deficiency, and still distinguishable in greyscale print.
        colors = [c["color"] for c in PLOT_RC_PARAMS["axes.prop_cycle"]]
        assert len(colors) >= 5
        assert all(color.startswith("#") for color in colors)
        assert "#ff0000" not in [c.lower() for c in colors]


class TestImageAxes:
    def test_turns_the_grid_off(self):
        # A grid behind a pcolormesh is invisible; in front of one it is noise.
        with plot_style_context():
            fig = plt.figure()
            ax = fig.add_subplot()
            assert ax.xaxis.get_gridlines()[0].get_visible()
            image_axes(ax)
            assert not any(line.get_visible() for line in ax.xaxis.get_gridlines())
            assert not any(line.get_visible() for line in ax.yaxis.get_gridlines())
            plt.close(fig)

    def test_returns_the_axis_for_chaining(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        assert image_axes(ax) is ax
        plt.close(fig)


class TestCommandLineOption:
    """``--figure-format`` on each entry point that writes figures.

    Only the wiring is checked here -- that the option exists, that it reaches
    :func:`set_figure_format`, and that it refuses a format the package cannot
    write. Whether the figures then land under the right names is the business
    of each command's own tests.
    """

    @staticmethod
    def _parser():
        import argparse

        from ..plotting import add_figure_format_argument

        return add_figure_format_argument(argparse.ArgumentParser())

    def test_the_option_reaches_the_module_state(self, clean_format):
        parsed = self._parser().parse_args(["--figure-format", "png"])
        set_figure_format(parsed.figure_format)
        assert current_figure_format() == "png"

    def test_omitting_it_leaves_the_default_in_charge(self, clean_format):
        parsed = self._parser().parse_args([])
        set_figure_format(parsed.figure_format)
        assert current_figure_format() == DEFAULT_FIGURE_FORMAT

    def test_an_unwritable_format_is_refused_by_argparse(self, clean_format):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--figure-format", "tiff"])

    @pytest.mark.parametrize(
        "entry_point",
        ["ell1fit.cli", "ell1fit.orbital_decay", "ell1fit.eccentricity"],
    )
    def test_every_figure_writing_command_offers_it(self, entry_point, capsys):
        import importlib

        module = importlib.import_module(entry_point)
        with pytest.raises(SystemExit):
            module.main(["--help"])
        assert "--figure-format" in capsys.readouterr().out
