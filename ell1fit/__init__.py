# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa

# ----------------------------------------------------------------------------

__all__ = []
from .logging import logging, logger

logger.setLevel(logging.INFO)


def splitext_improved(path):
    """
    Examples
    --------
    >>> import os
    >>> import numpy as np
    >>> np.all(splitext_improved("a.tar.gz") ==  ('a', '.tar.gz'))
    True
    >>> np.all(splitext_improved("a.tar") ==  ('a', '.tar'))
    True
    >>> path_with_dirs = os.path.join("a.f", "a.tar")
    >>> path_without_ext = os.path.join("a.f", "a")
    >>> np.all(splitext_improved(path_with_dirs) ==  (path_without_ext, '.tar'))
    True
    >>> path_with_dirs = os.path.join("a.a.a.f", "a.tar.gz")
    >>> path_without_ext = os.path.join("a.a.a.f", "a")
    >>> np.all(splitext_improved(path_with_dirs) ==  (path_without_ext, '.tar.gz'))
    True
    >>> path_with_dirs = os.path.join("a.a.a.f", "a.1.tar")
    >>> path_without_ext = os.path.join("a.a.a.f", "a.1")
    >>> np.all(splitext_improved(path_with_dirs) ==  (path_without_ext, '.tar'))
    True
    """
    import os

    dir, file = os.path.split(path)

    if len(file.split(".")) > 2 and file.endswith(".gz"):
        froot, ext = file.split(".")[0], "." + ".".join(file.split(".")[-2:])
    else:
        froot, ext = os.path.splitext(file)

    return os.path.join(dir, froot), ext
