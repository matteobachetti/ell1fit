Fit an ELL1 orbital model to pulsar data, accounting for spin derivatives
-------------------------------------------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge


Development
-----------

Everything below runs through `tox <https://tox.wiki>`_, which builds an
isolated environment for each task, so the only thing to install first is tox
itself::

    python -m pip install tox

Run the test suite. This also executes the examples embedded in the
documentation, so they cannot drift out of step with the code::

    tox -e py311-test

Build the HTML documentation. Warnings are treated as errors, so a malformed
docstring or a broken cross-reference fails the build rather than quietly
producing degraded pages. The result lands in ``docs/_build/html``::

    tox -e build_docs

Check code style — both the linter and the formatter::

    tox -e codestyle

Check that external links in the documentation still resolve. This one is also
run weekly by CI, since links rot on their own schedule rather than yours::

    tox -e linkcheck

If you would rather work in an environment you already have, install the
package with its test and documentation extras and call the tools directly::

    python -m pip install -e ".[test,docs]"
    pytest --pyargs ell1fit docs
    cd docs && sphinx-build -W -b html . _build/html

To confirm that a change left the numerical results untouched — which is what
you want when restructuring rather than altering behaviour — take a snapshot
before and after and compare them bit for bit::

    python tools/refactor_net.py capture -o before.json
    # ... make the change ...
    python tools/refactor_net.py capture -o after.json
    python tools/refactor_net.py diff before.json after.json

License
-------

This project is Copyright (c) Matteo Bachetti and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.


Contributing
------------

We love contributions! ell1fit is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
ell1fit based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
