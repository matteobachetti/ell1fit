import os
import glob
import pytest
from ell1fit.ell1fit import main as main_ell1fit
from ell1fit.create_parfile import main as main_ell1par


curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


class TestExecution:
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

        outputs = sorted(glob.glob(os.path.join(datadir, f"events[01]{label}_results.ecsv")))
        for out in outputs:
            assert os.path.exists(out)

        main_ell1par(f"{outputs[0]} -p {self.param_files[0]}".split())
        main_ell1par(f"{outputs[1]} -p {self.param_files[1]}".split())

        out_param = sorted(glob.glob(os.path.join(datadir, "events[01]_A1_F0_PB_TASC_results.par")))
        for out in out_param:
            assert os.path.exists(out)

    @classmethod
    def teardown_class(cls):
        outs = glob.glob(os.path.join(datadir, "*A1_*TASC*"))
        for out in outs:
            os.remove(out)
