Worked examples
===============

The examples on this page are executed as part of the test suite, so they
cannot drift out of step with the code.

.. contents::
   :local:
   :depth: 1

Command line
------------

The basic invocation takes one or more event files and a matching parameter
file for each::

    ell1fit obs1.nc -p obs1.par -P F0,A1 --minimize-first --nsteps 20000

``-P`` selects what to fit. Spin parameters expand per file, so ``F0`` on a
three-file run fits ``F0_0``, ``F0_1`` and ``F0_2`` independently, while
orbital parameters stay shared.

Fitting several observations jointly, with a sharper pulse model, energy
weighting, and self-consistent templates::

    ell1fit obs1.nc obs2.nc obs3.nc -p obs1.par obs2.par obs3.par \
        -P F0,F1,A1,PB,TASC -N 3 --use-weight --template-iterations 3 \
        -o campaign --nsteps 100000

A run can be extended simply by asking for more steps: the chain is stored in
``<outroot>.h5`` and sampling resumes from where it stopped.

Turning a fit back into an ephemeris::

    ell1par campaign_A1_F0_PB_TASC_results.ecsv -p obs1.par

Generating data with a known answer
-----------------------------------

Testing a timing pipeline requires data whose true solution is known.
:mod:`ell1fit.tests.datagen` synthesises events from an injected solution,
using a forward model written independently of the code under test:

.. doctest::

    >>> import io, tempfile
    >>> from contextlib import redirect_stdout
    >>> from ell1fit.tests.datagen import InjectedSolution, make_multi_epoch_dataset
    >>> solution = InjectedSolution()
    >>> round(solution.PB, 6), solution.A1
    (2.532971, 22.215)

Each epoch gets its own ``PEPOCH``, its own ``TASC`` alias, and its own ``F0``
propagated through ``F1``:

.. doctest::

    >>> outdir = tempfile.mkdtemp()
    >>> with redirect_stdout(io.StringIO()):   # the writer announces each file
    ...     dataset = make_multi_epoch_dataset(
    ...         outdir, epoch_offsets=(0.0, 37.0), n_events=400,
    ...         duration=20_000.0, seed=1, prefix="demo",
    ...     )
    >>> len(dataset["event_files"]), len(dataset["par_files"])
    (2, 2)
    >>> epochs = dataset["epochs"]
    >>> bool(epochs[0]["pepoch"] != epochs[1]["pepoch"])
    True
    >>> bool(epochs[0]["TASC"] != epochs[1]["TASC"])   # different orbital alias
    True

``TASC`` differs between the two epochs by a whole number of orbits, which is
the same physical solution:

.. doctest::

    >>> n_orbits = (epochs[1]["TASC"] - epochs[0]["TASC"]) / solution.PB
    >>> bool(abs(n_orbits - round(n_orbits)) < 1e-6)
    True

Deliberately wrong starting parameters are what a realistic test needs, and
``offsets`` provides them:

.. doctest::

    >>> with redirect_stdout(io.StringIO()):
    ...     shifted = make_multi_epoch_dataset(
    ...         outdir, epoch_offsets=(0.0,), n_events=200, duration=20_000.0,
    ...         offsets={"A1": 0.02}, seed=2, prefix="shifted",
    ...     )
    >>> "shifted0.par" in shifted["par_files"][0]
    True

Checking a solution by hand
---------------------------

The orbital inversion has a physical limit: the projected orbital velocity
:math:`A_1\omega` is expressed in units of *c*, and beyond :math:`c` the
arrival-time map is no longer monotonic, so no inverse exists.

.. doctest::

    >>> from ell1fit.phase_utils import orbit_is_invertible
    >>> PB_seconds = 2.532971 * 86400
    >>> orbit_is_invertible(PB_seconds, 22.215, 1.5e-4, -2.1e-4)   # a real pulsar
    True
    >>> orbit_is_invertible(PB_seconds, 40000.0)                   # superluminal
    False

Trial positions in that region are rejected by the posterior rather than
attempted.

Evaluating a template directly
------------------------------

A template is a callable mapping pulse phase to probability density, and it is
periodic, so any real phase is valid:

.. doctest::

    >>> import numpy as np
    >>> from ell1fit.templates import get_template_func
    >>> profile = 1 + 0.3 * np.cos(2 * np.pi * np.arange(64) / 64)
    >>> template = get_template_func(profile)
    >>> phases = np.array([0.0, 0.25, 0.5, 0.75])
    >>> bool(np.allclose(template(phases), template(phases + 3.0)))
    True

The fast path is validated against ``scipy`` at all times, and the reference
implementation stays available for comparison:

.. doctest::

    >>> reference = get_template_func(profile, backend="scipy")
    >>> probe = np.linspace(0, 1, 501)
    >>> bool(np.max(np.abs(template(probe) - reference(probe))) < 1e-12)
    True
