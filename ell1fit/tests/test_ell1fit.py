import os

import pytest
import numpy as np
from astropy.table import Table
from ..ell1fit import add_circular_orbit_numba, add_ell1_orbit_numba, safe_save
from ..ell1fit import simple_circular_deorbit_numba, simple_ell1_deorbit_numba


@pytest.mark.parametrize("A1", [1, 10, 40])
@pytest.mark.parametrize("PB", [0.3, 3, 30])
def test_circular_orbit(PB, A1):
    A1 /= 86400
    times = np.random.uniform(56000, 59000, 10)
    TASC = np.random.uniform(56000, 59000)

    orbited = add_circular_orbit_numba(times, PB, A1, TASC)
    deorbited = simple_circular_deorbit_numba(orbited, PB, A1, TASC, tolerance=1e-8)
    assert np.all(np.abs(deorbited - times) < 1e-8)


@pytest.mark.parametrize("A1", [10, 40])
@pytest.mark.parametrize("PB", [0.3, 3])
@pytest.mark.parametrize("E1", [0.0001, 0.1])
@pytest.mark.parametrize("E2", [0.0001, 0.1])
def test_ell1_orbit(PB, A1, E1, E2):
    A1 /= 86400
    times = np.random.uniform(56000, 59000, 10)
    TASC = np.random.uniform(56000, 59000)

    orbited = add_ell1_orbit_numba(times, PB, A1, TASC, E1, E2)
    deorbited = simple_ell1_deorbit_numba(orbited, PB, A1, TASC, E1, E2, tolerance=1e-8)
    assert np.all(np.abs(deorbited - times) < 1e-8)


def test_safe_save():
    results = Table({"a": [2]})
    results_2 = Table({"a": ["3"]})
    output_file = "blabla.csv"
    safe_save(results, output_file)
    safe_save(results_2, output_file)
    with pytest.warns(UserWarning, match="Merging old and new"):
        safe_save(results_2, output_file)
    os.unlink(output_file)
    os.unlink("old_" + output_file)
