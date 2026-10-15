"""Tests for resuming a stored chain safely.

A sample in the HDF5 backend is a *local* coordinate: the physical value is
``sample * factor + initial``. Resuming a chain written under a different
``factor`` or ``initial`` therefore continues a chain of nothing, silently, and
the only outward sign is an acceptance rate that collapses. These tests pin the
fingerprint that stops it.
"""

import emcee
import numpy as np
import pytest

from ell1fit.mcmc_utils import local_frame_key, safe_run_sampler


def _flat_posterior(pars):
    """A trivial, cheap, well-behaved log-posterior for chains that go nowhere."""
    return -0.5 * float(np.sum(np.asarray(pars) ** 2))


def _existing_chain(outroot, n_steps=120, ndim=2, frame=None):
    """Write a real emcee chain to ``outroot + '.h5'``, optionally fingerprinted."""
    backend = emcee.backends.HDFBackend(outroot + ".h5")
    nwalkers = 32
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, _flat_posterior, backend=backend)
    sampler.run_mcmc(np.random.normal(size=(nwalkers, ndim)), n_steps, progress=False)
    if frame is not None:
        from ell1fit.mcmc_utils import _write_local_frame

        _write_local_frame(outroot + ".h5", backend.name, frame)
    return outroot


@pytest.fixture
def frame():
    """The local frame of a two-parameter fit."""
    return local_frame_key(["F0_0", "A1"], [1e-9, 0.2], [0.7, 22.2])


def test_the_key_changes_with_the_factors_and_with_the_origin(frame):
    """Both halves of ``sample * factor + initial`` have to be fingerprinted.

    Changing either one redefines what a stored sample means, so either one
    must produce a different key.
    """
    assert local_frame_key(["F0_0", "A1"], [1e-9, 0.2], [0.7, 22.2]) == frame
    assert local_frame_key(["F0_0", "A1"], [1e-12, 0.2], [0.7, 22.2]) != frame
    assert local_frame_key(["F0_0", "A1"], [1e-9, 0.2], [0.7, 22.3]) != frame
    assert local_frame_key(["F0_0", "PB"], [1e-9, 0.2], [0.7, 22.2]) != frame


def test_a_chain_written_in_the_same_frame_is_resumed(tmp_path, frame, caplog):
    """The check must not cost the restart support it is guarding."""
    outroot = _existing_chain(str(tmp_path / "same"), frame=frame)
    with caplog.at_level("INFO"):
        safe_run_sampler(_flat_posterior, [0.0, 0.0], max_n=130, outroot=outroot, local_frame=frame)
    assert "Starting from where we left" in caplog.text


def test_a_chain_written_in_a_different_frame_is_discarded(tmp_path, frame, caplog):
    """The bug this exists for: a rescaled chain must not be continued.

    Resuming it is worse than losing it, because the stored coordinates mean
    something else now and nothing downstream can tell.
    """
    stale = local_frame_key(["F0_0", "A1"], [1e-12, 0.2], [0.7, 22.2])
    outroot = _existing_chain(str(tmp_path / "moved"), frame=stale)
    with caplog.at_level("INFO"):
        safe_run_sampler(_flat_posterior, [0.0, 0.0], max_n=130, outroot=outroot, local_frame=frame)
    assert "Starting from zero" in caplog.text
    assert "scale" in caplog.text.lower()


def test_a_chain_with_no_recorded_frame_is_discarded(tmp_path, frame, caplog):
    """An unfingerprinted chain predates the check, so it cannot be vouched for.

    Redoing a chain is cheap; continuing one that may have been written under a
    different scaling is the failure this is here to prevent.
    """
    outroot = _existing_chain(str(tmp_path / "old"))
    with caplog.at_level("INFO"):
        safe_run_sampler(_flat_posterior, [0.0, 0.0], max_n=130, outroot=outroot, local_frame=frame)
    assert "Starting from zero" in caplog.text


def test_a_caller_that_names_no_frame_keeps_the_old_behaviour(tmp_path, caplog):
    """Without a frame to compare there is nothing to check, so resume as before."""
    outroot = _existing_chain(str(tmp_path / "unchecked"))
    with caplog.at_level("INFO"):
        safe_run_sampler(_flat_posterior, [0.0, 0.0], max_n=130, outroot=outroot)
    assert "Starting from where we left" in caplog.text
