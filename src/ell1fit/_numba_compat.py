"""Import numba, or stand in for it when it is not installed.

Every compiled kernel in this package imports its decorators from here rather
than from ``numba`` directly, so that importing :mod:`ell1fit` never depends on
numba being present.

Why bother
----------
numba is a hard requirement for *running* a fit: the deorbiting loop and the
template evaluation are the hot path, and in pure Python they are slower by
orders of magnitude. But numba pulls in ``llvmlite``, which is a recurrent
source of installation trouble for reasons unrelated to this package -- most
often no wheel yet existing for a newly released Python. Several things want to
*import* the package without running anything: building the documentation, which
must introspect every module, and any tool that inspects signatures. Those
should not be hostage to an LLVM build.

What the fallbacks do
---------------------
``njit``
    Returns the function untouched. It runs as ordinary Python: correct, and
    slow.
``vectorize``
    Falls back to :func:`numpy.vectorize`, which preserves the element-wise
    semantics the decorated functions rely on -- several use scalar control flow
    that would otherwise not broadcast over an array.
``prange``
    Plain :func:`range`, so parallel loops become sequential ones.

Both spellings of the decorators are supported, bare (``@njit``) and
parametrised (``@njit(parallel=True)``).

:data:`HAS_NUMBA` records which path was taken. Anything that cares about speed
rather than correctness should check it and warn.
"""

import numpy as np

__all__ = [
    "HAS_NUMBA",
    "float32",
    "float64",
    "int64",
    "njit",
    "prange",
    "vectorize",
]

try:
    from numba import float32, float64, int64, njit, prange, vectorize

    HAS_NUMBA = True

except ImportError:  # pragma: no cover - exercised only where numba is absent
    HAS_NUMBA = False

    prange = range

    # Placeholders for the numba type objects used in ``@vectorize`` signatures.
    # The fallback vectorize ignores its signature argument entirely.
    float32 = float64 = int64 = None

    def njit(*args, **kwargs):
        """No-op stand-in for ``numba.njit``: run the function as Python."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    def vectorize(*args, **kwargs):
        """Stand-in for ``numba.vectorize`` backed by :func:`numpy.vectorize`.

        Unlike :func:`njit`, returning the function unchanged would be wrong
        here: the decorated functions use scalar control flow and rely on the
        decorator to broadcast them over arrays.
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return np.vectorize(args[0])

        def decorator(func):
            return np.vectorize(func)

        return decorator
