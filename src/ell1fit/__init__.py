try:
    from .version import version as __version__
except ImportError:
    __version__ = ""

__all__ = ["__version__"]


def splitext_improved(path):
    """
    Examples
    --------
    >>> import os
    >>> import numpy as np
    >>> assert np.all(splitext_improved("a.tar.gz") ==  ('a', '.tar.gz'))
    >>> assert np.all(splitext_improved("a.tar") ==  ('a', '.tar'))
    >>> path_with_dirs = os.path.join("a.f", "a.tar")
    >>> path_without_ext = os.path.join("a.f", "a")
    >>> assert np.all(splitext_improved(path_with_dirs) ==  (path_without_ext, '.tar'))
    >>> path_with_dirs = os.path.join("a.a.a.f", "a.tar.gz")
    >>> path_without_ext = os.path.join("a.a.a.f", "a")
    >>> assert np.all(splitext_improved(path_with_dirs) ==  (path_without_ext, '.tar.gz'))
    >>> path_with_dirs = os.path.join("a.a.a.f", "a.1.tar")
    >>> path_without_ext = os.path.join("a.a.a.f", "a.1")
    >>> assert np.all(splitext_improved(path_with_dirs) ==  (path_without_ext, '.tar'))
    """
    import os

    dir, file = os.path.split(path)

    if len(file.split(".")) > 2 and file.endswith(".gz"):
        froot, ext = file.split(".")[0], "." + ".".join(file.split(".")[-2:])
    else:
        froot, ext = os.path.splitext(file)

    return os.path.join(dir, froot), ext
