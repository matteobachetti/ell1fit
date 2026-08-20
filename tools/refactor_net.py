#!/usr/bin/env python
"""Before/after snapshot tool for verifying that a refactor changed nothing.

Why this is a tool and not a test
---------------------------------
Bit-for-bit comparison is the right way to check that a pure restructuring left
the numbers alone: the arithmetic is deterministic, so a rename or a module move
must reproduce every float exactly. But it is the *wrong* thing to check into a
test suite, because pinned numbers cannot distinguish "correct" from
"consistently wrong", and they turn every library upgrade into a spurious
failure. See ``ell1fit/tests/test_recovery.py`` for the checked-in tests, which
assert physics instead.

So this lives here: run it on one machine, before and after a change, and diff.

Usage
-----
::

    python tools/refactor_net.py capture -o /tmp/before.json
    ... make the change ...
    python tools/refactor_net.py capture -o /tmp/after.json
    python tools/refactor_net.py diff /tmp/before.json /tmp/after.json

Add ``--full`` to include a seeded MCMC run. That is slower and more fragile
(it depends on the global numpy RNG stream), so the default omits it; the
deterministic core plus the point estimate catch essentially everything a
restructuring can break.

Two layers
----------
**Layer 1, end-to-end.** Runs the CLI exactly as a user would and records every
numeric field it writes. This layer depends on *no* internal function names, so
it keeps working no matter how the package is reorganised -- which makes it the
authoritative check.

**Layer 2, deterministic core.** Probes individual computations (phases,
profiles, templates, likelihood) on a grid of parameter values. These imports
are listed in ``CORE_IMPORTS`` below; when a refactor moves one of them, update
that table. Layer 2 exists to *localise* a difference that layer 1 detects, so
a mistake while updating it cannot hide a regression -- layer 1 would still
fail.

Arrays are compared by SHA-256 of their raw bytes, so the check is bitwise. A
few sample values are recorded alongside each hash purely to make a reported
difference readable.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

# Import table for layer 2. When a refactor moves one of these, change it here.
CORE_IMPORTS = {
    "calculate_phases": ("ell1fit.phase_utils", "_calculate_phases"),
    "folded_profile": ("ell1fit.phase_utils", "folded_profile"),
    "create_template": ("ell1fit.templates", "create_template_from_profile_harm"),
    "template_func": ("ell1fit.templates", "get_template_func"),
    "likelihood": ("ell1fit.likelihoods", "pletsch_clarke_likelihood"),
    "assign_logpriors": ("ell1fit.priors", "assign_logpriors"),
    "get_factors": ("ell1fit.scaling", "get_factors"),
}

SEED = 20260820
SEC_PER_DAY = 86400.0

#: Entries that legitimately change between two runs of identical code: the
#: wall-clock stamp the pipeline records, and the input paths, which live in a
#: fresh temporary directory each time. Comparing these would make every diff
#: report a false difference.
VOLATILE_PREFIXES = ("result::date", "result::fname_")


def _is_volatile(key):
    """True if an entry is expected to differ between runs of unchanged code."""
    return key.startswith(VOLATILE_PREFIXES)


def _resolve(name):
    """Import one entry of :data:`CORE_IMPORTS`, or report why it failed."""
    module_name, attribute = CORE_IMPORTS[name]
    try:
        module = __import__(module_name, fromlist=[attribute])
    except ImportError as exc:
        return None, f"module {module_name!r} not importable: {exc}"
    if not hasattr(module, attribute):
        return None, f"{module_name!r} has no attribute {attribute!r}"
    return getattr(module, attribute), None


def _describe_array(values):
    """Summarize an array: bitwise hash plus a readable sample."""
    array = np.ascontiguousarray(np.asarray(values, dtype=float))
    finite = array[np.isfinite(array)]
    return {
        "kind": "array",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "first": [repr(float(v)) for v in array.ravel()[:3]],
        "last": [repr(float(v)) for v in array.ravel()[-3:]],
        "sum": repr(float(np.sum(finite))) if finite.size else None,
        "n_nonfinite": int(array.size - finite.size),
    }


def _describe_scalar(value):
    """Record a scalar at full precision. ``repr`` round-trips float64 exactly."""
    return {"kind": "scalar", "value": repr(float(value))}


def _synthetic_dataset(outdir):
    """Build the fixed synthetic dataset both layers run against."""
    from ell1fit.tests.datagen import make_multi_epoch_dataset

    return make_multi_epoch_dataset(
        outdir,
        epoch_offsets=(0.0, 37.0),
        n_events=1500,
        duration=60_000.0,
        seed=SEED,
        prefix="net",
    )


def _parameter_dict(solution, epochs, **override):
    """Assemble the parameter mapping the phase machinery expects."""
    parameters = {
        "PB": np.float64(solution.PB_sec),
        "A1": np.float64(solution.A1),
        "TASC": np.float64(solution.TASC),
        "EPS1": np.float64(solution.EPS1),
        "EPS2": np.float64(solution.EPS2),
        "PBDOT": np.float64(0.0),
    }
    for i, epoch in enumerate(epochs):
        parameters[f"F0_{i}"] = np.float64(epoch["F0"])
        parameters[f"F1_{i}"] = np.float64(epoch["F1"])
        parameters[f"PEPOCH_{i}"] = np.float64(epoch["pepoch"])
        parameters[f"Phase_{i}"] = np.float64(0.0)
    parameters.update({k: np.float64(v) for k, v in override.items()})
    return parameters


def capture_core(dataset):
    """Layer 2: probe the deterministic core on a grid of parameter values."""
    record = {}
    resolved = {}
    for name in CORE_IMPORTS:
        func, error = _resolve(name)
        if error is not None:
            record[f"MISSING::{name}"] = {"kind": "missing", "reason": error}
        resolved[name] = func

    solution = dataset["solution"]
    epochs = dataset["epochs"]
    times = [epoch["times_from_pepoch"] for epoch in epochs]

    # A small grid of parameter offsets, so the probe covers more than one point
    # in parameter space and a difference that only shows up off-centre is seen.
    grid = [
        ("truth", {}),
        ("A1+1e-3", {"A1": solution.A1 + 1e-3}),
        ("PB+1e-2s", {"PB": solution.PB_sec + 1e-2}),
        ("TASC+1e-5d", {"TASC": solution.TASC + 1e-5}),
    ]

    for label, override in grid:
        parameters = _parameter_dict(solution, epochs, **override)

        if resolved["calculate_phases"] is not None:
            phases = resolved["calculate_phases"](times, parameters)
            for i, p in enumerate(phases):
                record[f"phases::{label}::file{i}"] = _describe_array(p)

        if resolved["folded_profile"] is not None:
            profiles = resolved["folded_profile"](times, parameters, nbin=32)
            for i, p in enumerate(profiles):
                record[f"profile::{label}::file{i}"] = _describe_array(p)

    # Templates and likelihood, evaluated at the truth.
    parameters = _parameter_dict(solution, epochs)
    if resolved["folded_profile"] is not None and resolved["create_template"] is not None:
        profiles = resolved["folded_profile"](times, parameters, nbin=32)
        tmpdir = tempfile.mkdtemp()
        try:
            for i, profile in enumerate(profiles):
                template, additional_phase = resolved["create_template"](
                    profile,
                    nharm=2,
                    final_nbin=200,
                    imagefile=os.path.join(tmpdir, f"t{i}.jpg"),
                )
                record[f"template::file{i}"] = _describe_array(template)
                record[f"template_phase::file{i}"] = _describe_scalar(additional_phase)

                if resolved["template_func"] is not None:
                    func = resolved["template_func"](template)
                    probe = np.linspace(0.0, 1.0, 501)
                    record[f"template_func::file{i}"] = _describe_array(func(probe))

                    if resolved["likelihood"] is not None:
                        phases = resolved["calculate_phases"](times, parameters)
                        record[f"loglike::file{i}"] = _describe_scalar(
                            resolved["likelihood"](phases[i], func)
                        )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Priors and scaling factors, which drive the optimizer's conditioning.
    if resolved["assign_logpriors"] is not None:
        names = ["A1", "PB", "TASC", "F0_0", "F0_1", "Phase_0", "Phase_1"]
        with_unc = {
            "A1": [np.float64(solution.A1), np.float64(1e-3)],
            "PB": [np.float64(solution.PB_sec), np.float64(1e-6 * SEC_PER_DAY)],
            "TASC": [np.float64(solution.TASC), np.float64(1e-6)],
            "F0_0": [np.float64(epochs[0]["F0"]), np.float64(1e-9)],
            "F0_1": [np.float64(epochs[1]["F0"]), np.float64(1e-9)],
            "Phase_0": [np.float64(0.1), np.float64(np.nan)],
            "Phase_1": [np.float64(0.2), np.float64(np.nan)],
        }
        priors = resolved["assign_logpriors"](names, with_unc, obs_length=[60_000.0, 60_000.0])
        for name, prior in zip(names, priors):
            centre = float(with_unc[name][0])
            scale = abs(float(with_unc[name][1]))
            if not np.isfinite(scale) or scale == 0:
                scale = 0.1
            probe = centre + scale * np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
            record[f"logprior::{name}"] = _describe_array([prior(v) for v in probe])
            bounds = getattr(prior, "phys_bounds", None)
            if bounds is not None:
                record[f"logprior_bounds::{name}"] = _describe_array(list(bounds))

    return record


def capture_end_to_end(dataset, outdir, full=False):
    """Layer 1: run the CLI as a user would and record every numeric output."""
    from astropy.table import Table

    from ell1fit.ell1fit import main as main_ell1fit

    record = {}
    outroot = os.path.join(outdir, "net")

    # emcee and the weighted-profile scatter estimate both draw from the global
    # numpy RNG, so seed it to make the run reproducible.
    np.random.seed(SEED)

    argv = (
        list(dataset["event_files"])
        + ["-p"]
        + list(dataset["par_files"])
        + ["-P", "F0,A1", "-N", "2", "--minimize-first", "-o", outroot]
        + ["--nsteps", "1000" if full else "200"]
    )
    main_ell1fit(argv)

    results_path = outroot + "_A1_F0_N2_results.ecsv"
    if not os.path.exists(results_path):
        record["MISSING::results"] = {"kind": "missing", "reason": results_path}
        return record

    table = Table.read(results_path)
    row = table[-1]
    for column in sorted(table.colnames):
        value = row[column]
        try:
            record[f"result::{column}"] = _describe_scalar(value)
        except (TypeError, ValueError):
            record[f"result::{column}"] = {"kind": "text", "value": str(value)}

    return record


def do_capture(args):
    """Run both layers and write a snapshot."""
    workdir = tempfile.mkdtemp(prefix="refactor_net_")
    try:
        dataset = _synthetic_dataset(workdir)

        record = {}
        for key, value in capture_core(dataset).items():
            record[key] = value

        if not args.core_only:
            for key, value in capture_end_to_end(dataset, workdir, full=args.full).items():
                record[key] = value
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    payload = {
        "seed": SEED,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "entries": record,
    }
    with open(args.out, "w") as fobj:
        json.dump(payload, fobj, indent=1, sort_keys=True)

    print(f"Captured {len(record)} entries to {args.out}")
    missing = [k for k in record if k.startswith("MISSING::")]
    if missing:
        print(f"  WARNING: {len(missing)} probe(s) could not run: {', '.join(missing)}")
    return 0


def do_diff(args):
    """Compare two snapshots and report every difference."""
    with open(args.before) as fobj:
        before = json.load(fobj)
    with open(args.after) as fobj:
        after = json.load(fobj)

    if before["numpy"] != after["numpy"] or before["python"] != after["python"]:
        print(
            "NOTE: environments differ "
            f"(python {before['python']}->{after['python']}, "
            f"numpy {before['numpy']}->{after['numpy']}). "
            "Bitwise differences may not be caused by your change."
        )

    entries_before = before["entries"]
    entries_after = after["entries"]

    keys_before = {k for k in entries_before if not _is_volatile(k)}
    keys_after = {k for k in entries_after if not _is_volatile(k)}
    n_volatile = len(entries_before) - len(keys_before)

    only_before = sorted(keys_before - keys_after)
    only_after = sorted(keys_after - keys_before)
    shared = sorted(keys_before & keys_after)

    differences = []
    for key in shared:
        b, a = entries_before[key], entries_after[key]
        if b == a:
            continue
        if b.get("kind") == "array" and a.get("kind") == "array":
            detail = (
                f"sha256 {b['sha256'][:12]} -> {a['sha256'][:12]}, "
                f"sum {b['sum']} -> {a['sum']}"
            )
        else:
            detail = f"{b.get('value')} -> {a.get('value')}"
        differences.append((key, detail))

    print(
        f"{len(shared)} entries compared, {len(differences)} differ "
        f"({n_volatile} volatile entries ignored)."
    )
    if only_before:
        print(f"\nPresent only BEFORE ({len(only_before)}):")
        for key in only_before:
            print(f"  - {key}")
    if only_after:
        print(f"\nPresent only AFTER ({len(only_after)}):")
        for key in only_after:
            print(f"  + {key}")

    if differences:
        print(f"\nDIFFERENCES ({len(differences)}):")
        for key, detail in differences:
            print(f"  ! {key}: {detail}")

    identical = not differences and not only_before and not only_after
    print("\nIDENTICAL" if identical else "\nNOT IDENTICAL")
    return 0 if identical else 1


def main(argv=None):
    """Entry point for the refactor net."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Record a snapshot")
    capture.add_argument("-o", "--out", required=True, help="Snapshot file to write")
    capture.add_argument(
        "--full",
        action="store_true",
        help="Use a longer MCMC run (slower; depends on the global numpy RNG stream)",
    )
    capture.add_argument(
        "--core-only",
        action="store_true",
        help="Skip the end-to-end run and probe only the deterministic core",
    )
    capture.set_defaults(func=do_capture)

    diff = subparsers.add_parser("diff", help="Compare two snapshots")
    diff.add_argument("before")
    diff.add_argument("after")
    diff.set_defaults(func=do_diff)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
