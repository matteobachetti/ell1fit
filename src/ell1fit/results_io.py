"""Result-table and output filename helpers for ell1fit."""

import copy
import os
import warnings

from astropy.table import Table, vstack, TableMergeError

from .phase_utils import simple_freq_re


def _format_energy_string(energy_range):
    """Format an energy-range suffix used in output filenames.

    Returns
    -------
    str
        Empty string if no range is provided, otherwise
        ``_<emin>-<emax>keV`` with open bounds represented by ``**``.
    """
    if energy_range is None:
        return ""
    if energy_range[0] is None and energy_range[1] is None:
        return ""
    lower = "**" if energy_range[0] is None else f"{energy_range[0]:g}"
    upper = "**" if energy_range[1] is None else f"{energy_range[1]:g}"

    return f"_{lower}-{upper}keV"


def look_for_string_in_list_of_strings(input_list, string):
    """Return all strings in ``input_list`` containing ``string``.

    Returns
    -------
    list of str
        Matching entries in input order.
    """
    output_list = []
    for value in input_list:
        if string in value:
            output_list.append(value)
    return output_list


def look_for_list_of_strings_in_string(input_list, string):
    """Return the first candidate from ``input_list`` found inside ``string``.

    Returns
    -------
    str or None
        First matching candidate or ``None`` if no match is found.
    """
    for value in input_list:
        if value in string:
            return value
    return None


def split_output_results(result_table, n_files, fit_parameters):
    """
    Examples
    --------
    >>> vals_dict = {"dF0_1": [234], "dF0_1_16": [4], "TASC_0": [3.], "TASC_10": [5.], "PB": [3.]}
    >>> result_table = Table(vals_dict)
    >>> output_tables = split_output_results(result_table, 2, ["F0", "F1", "TASC"])
    >>> assert sorted(output_tables[0].colnames) == ["PB", "TASC_0", "TASC_10"]
    >>> assert sorted(output_tables[1].colnames) == ["PB", "TASC_0", "TASC_10", "dF0", "dF0_16"]
    """
    tier_2_parameters = [par for par in fit_parameters if simple_freq_re.match(par)]

    tier_2_parameters = tier_2_parameters + [
        "Phase",
        "PEPOCH",
        "Start",
        "Stop",
        "fname",
        "ctrate",
        "pf",
        "additional_phase",
    ]
    common_table = copy.deepcopy(result_table)
    output_tables = [Table() for _ in range(n_files)]

    for par in tier_2_parameters:
        # Use reverse order, so that we eliminate 10, 11, etc. before going to 1
        for i in list(range(n_files))[::-1]:
            par_to_test = f"{par}_{i}"

            cols = look_for_string_in_list_of_strings(common_table.colnames, par_to_test)
            for colname in cols:
                clean_colname = colname.replace(f"{par}_{i}", f"{par}")

                output_tables[i][clean_colname] = common_table[colname]
                common_table.remove_column(colname)

    for i in range(n_files):
        for col in common_table.colnames:
            output_tables[i][col] = common_table[col]

    return output_tables


def safe_save(results, output_file, **write_kwargs):
    """
    Examples
    --------
    >>> results = Table({"a": [2]})
    >>> results_2 = Table({"a": ["3"]})
    >>> output_file = "blabla.csv"
    >>> safe_save(results, output_file)
    >>> safe_save(results, output_file)
    >>> out = Table.read(output_file)
    >>> len(out)
    2
    >>> os.unlink(output_file)
    >>> os.unlink("old_" + output_file)
    """
    if os.path.exists(output_file):
        old = Table.read(output_file)
        old.write("old_" + output_file, overwrite=True)
        try:
            results = vstack([old, results])
        except TableMergeError:
            warnings.warn(
                "Merging old and new results failed. Old results were saved in a separate file."
            )

    results.write(output_file, overwrite=True, **write_kwargs)
    return
