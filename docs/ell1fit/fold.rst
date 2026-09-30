Folding without fitting: ``ell1fold``
=====================================

``ell1fold`` does what ``ell1fit`` does first, and stops there: it loads the
parfiles and the events, removes the orbit, and folds. Use it to check that a
solution holds before spending time on a fit, or to get per-event phases for
some other analysis.

Quickstart
----------

.. code-block:: console

   $ ell1fold epoch*_ev.nc -p epoch*_ev.par -e 3 30 --save-phases

As with ``ell1fit``, give one parfile per event file, in the same order. Any
event file HENDRICS reads is accepted: its own ``.nc`` and pickle files and
OGIP FITS event files (``.evt``, ``.fits``, optionally gzipped).

What it writes
--------------

``<name>_phaseogram.pdf``
   The folded profile over the phaseogram, two cycles wide. The phaseogram is
   mean-subtracted and smoothed, exactly as in the diagnostic plots ``ell1fit``
   draws after a fit, with ``-b/--nbin`` phase bins per pulse cycle (32 by
   default). The time axis is in days from the parfile's ``PEPOCH``, which the
   label quotes. The energy selection (``-e``) and the good time intervals
   apply here. ``--figure-format`` picks another format.

``<name>_phase.<ext>`` (with ``--save-phases``)
   A copy of the event file in the same format, with three columns added:

   ``PULSE_PHASE``
      Pulse phase in [0, 1): the phase ``ell1fit`` folds with.
   ``ABS_PHASE``
      The same phase, unwrapped: cycles since the parfile's ``PEPOCH``. Its
      zero is the phase the spin model gives at ``PEPOCH`` after the orbit is
      removed, *not* the ``TZRMJD`` reference PINT uses. Two files folded with
      different parfiles therefore have unrelated phase zeros.
   ``ORBIT_PHASE``
      ``(t - TASC) / PB`` in [0, 1), in observed time: 0 at the ascending
      node. Any orbital derivatives in the parfile are applied, as in the fit.

   Every event in the file is phased, with no energy or GTI selection, so the
   new columns line up with the existing rows.

   FITS files are copied with the columns appended to their ``EVENTS``
   extension; all other headers, columns and extensions are kept as they were.
   Other formats are rewritten by HENDRICS. HDF5 and ECSV store the column
   names in lower case.

Event weights
-------------

``--weight-column NAME`` weights each event by that column in the profile and
the phaseogram, e.g. a column of pulse-probability weights computed elsewhere.
The name is matched in either case, and it is selected along with the events,
so the energy and GTI cuts cannot misalign it. For FITS files the column is
read even though HENDRICS would normally drop non-standard columns.

This is not ``ell1fit --use-weight``, which computes its own energy-based
weights. ``ell1fold`` only uses weights that are already in the file.
