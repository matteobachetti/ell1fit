"""End-to-end CLI smoke tests against the checked-in event files.

These confirm the command-line tools run, produce the files they promise, and
round-trip through ``ell1par``. They do not check the numbers -- that is
``test_recovery.py``, which fits data with a known injected solution.

A limitation worth knowing about
--------------------------------
``events0.par`` and ``events1.par`` are **byte-identical**: same ``PEPOCH``,
same ``TASC``, same ``F0``. So although these tests pass two files, they do not
exercise multi-epoch behaviour at all -- not the per-file ``F0_i``/``PEPOCH_i``
bookkeeping, not the modulo-``PB`` aliasing of ``TASC``, not the reference-model
epoch alignment. Every one of those is covered instead by the synthetic
generator in :mod:`ell1fit.tests.datagen`, which builds genuinely distinct
epochs. Replacing this data would mean regenerating the event files too, since
the events were simulated from these exact models.
"""

import glob
import os

import numpy as np
import pytest
from astropy.table import Table

from ell1fit.cli import main as main_ell1fit
from ell1fit.create_parfile import main as main_ell1par

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


class TestExecution:
    """Run the shipped CLI tools end to end and check the products appear."""

    @classmethod
    def setup_class(cls):
        cls.event_files = sorted(glob.glob(os.path.join(datadir, "events[01].nc")))
        cls.param_files = sorted(glob.glob(os.path.join(datadir, "events[01].par")))

    @pytest.mark.parametrize("likelihood", ["PC", "Rayleigh"])
    def test_ell1fit_and_ell1par(self, likelihood):
        cmdlines = (
            self.event_files
            + ["-p"]
            + self.param_files
            + ["-P", "F0,PB,A1,TASC", "--likelihood", likelihood]
        )

        cmdline1 = cmdlines + ["--nsteps", "100"]
        cmdline2 = cmdlines + ["--nsteps", "200"]

        # Get to 100, then continue up to 200
        main_ell1fit(cmdline1)
        main_ell1fit(cmdline2)
        label = "_A1_F0_PB_TASC"
        if likelihood == "Rayleigh":
            label += "_rayleigh"

        for ev in self.event_files:
            root = ev.replace(".nc", "")
            ecsv_res = f"{root}{label}_results.ecsv"
            ecsv_par = f"{root}{label}_results.par"
            assert os.path.exists(ecsv_res)
            table = Table.read(ecsv_res)
            assert "fname" in table.colnames
            assert table["fname"][-1] == ev
            assert os.path.exists(ecsv_par)
            os.rename(ecsv_par, ecsv_par.replace(".par", "_e1fit.par"))

        # No -o was given, so the combined multi-file output falls back to
        # "out" and lands in the current directory rather than next to the
        # input files -- confirm that, and see teardown_class for cleanup.
        assert os.path.exists(f"out{label}_results.ecsv")

        # If we want to reproduce them with ell1par... we can do that
        from pint.models import get_model

        for ev, par in zip(self.event_files, self.param_files):
            root = ev.replace(".nc", "")
            ecsv_res = f"{root}{label}_results.ecsv"
            par_res = f"{root}{label}_results.par"
            main_ell1par(f"{ecsv_res} -p {par}".split())
            # NB: we renamed the one from ell1fit
            assert os.path.exists(par_res)
            model1 = get_model(par_res)
            model2 = get_model(par_res.replace(".par", "_e1fit.par"))
            # Compare the two models. They have to be identical
            # (They are produced by the same function!)
            comparison_table = model1.compare(model2, verbosity="min", format="text").split("\n")

            for line in comparison_table:
                if "parameter" in line.lower():
                    continue
                if "-----" in line:
                    continue
                raise AssertionError(f"Comparison failed: {line}")

    def test_ell1fit_minimize_first(self):
        cmdline = (
            self.event_files
            + ["-p"]
            + self.param_files
            + ["-P", "F0,PB,A1,TASC", "--likelihood", "PC", "--minimize-first", "--nsteps", "100"]
        )

        main_ell1fit(cmdline)

        label = "_A1_F0_PB_TASC"
        for ev in self.event_files:
            root = ev.replace(".nc", "")
            ecsv_res = f"{root}{label}_results.ecsv"
            initial_phaseogram = f"{root}{label}.jpg"
            assert os.path.exists(ecsv_res)
            assert os.path.exists(initial_phaseogram)

            table = Table.read(ecsv_res)
            assert np.isfinite(table["rough_dPB"][-1])
            assert np.isfinite(table["rough_dA1"][-1])
            assert np.isfinite(table["rough_dTASC"][-1])
            assert np.isfinite(table["rough_dF0"][-1])

        # See test_ell1fit_and_ell1par: no -o given, so the combined output
        # lands in the current directory.
        assert os.path.exists(f"out{label}_results.ecsv")

    def test_fitting_pbdot_is_rejected(self):
        """``-P PBDOT`` must fail loudly instead of sampling its own prior.

        ``PBDOT`` reaches the computation only through PINT's binary-epoch
        alignment; the phase model itself holds ``PB`` constant, so the
        likelihood is flat in it. See
        ``test_ell1fit.test_phases_are_flat_in_pbdot`` for the direct
        demonstration.
        """
        cmdline = (
            self.event_files + ["-p"] + self.param_files + ["-P", "F0,PBDOT", "--nsteps", "10"]
        )

        with pytest.raises(ValueError, match="PBDOT cannot be fitted"):
            main_ell1fit(cmdline)

    @classmethod
    def teardown_class(cls):
        outs = glob.glob(os.path.join(datadir, "*A1_*TASC*"))
        # Combined multi-file outputs (no -o given) fall back to "out" and
        # land in the current directory instead of datadir -- clean those
        # up too, wherever the test process actually ran from.
        outs += glob.glob(os.path.join(os.getcwd(), "out_A1_*TASC*"))
        for out in outs:
            os.remove(out)
