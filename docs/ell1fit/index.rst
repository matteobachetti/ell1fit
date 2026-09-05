*********************
ell1fit Documentation
*********************

``ell1fit`` fits ELL1 binary and spin parameters directly to X-ray photon
arrival times, without first compressing them into pulse times of arrival. It
is aimed at sources faint enough that no single observation yields a usable
arrival time.

Start with :doc:`motivation` for what the code is for and why the method is
what it is, then :doc:`pipeline` for how a fit actually proceeds.

.. toctree::
   :maxdepth: 2
   :caption: Using ell1fit

   motivation
   pipeline
   examples
   orbital_derivatives
   eccentricity

.. toctree::
   :maxdepth: 2
   :caption: Reference

   performance
   limitations
   design

Reference/API
=============

.. automodapi:: ell1fit.pipeline
   :no-inheritance-diagram:
.. automodapi:: ell1fit.phase_utils
   :no-inheritance-diagram:
.. automodapi:: ell1fit.templates
   :no-inheritance-diagram:
.. automodapi:: ell1fit.likelihoods
   :no-inheritance-diagram:
.. automodapi:: ell1fit.priors
   :no-inheritance-diagram:
.. automodapi:: ell1fit.scaling
   :no-inheritance-diagram:
.. automodapi:: ell1fit.posterior
   :no-inheritance-diagram:
.. automodapi:: ell1fit.fitting
   :no-inheritance-diagram:
.. automodapi:: ell1fit.refinement
   :no-inheritance-diagram:
.. automodapi:: ell1fit.setup_types
   :no-inheritance-diagram:
.. automodapi:: ell1fit.models
   :no-inheritance-diagram:
.. automodapi:: ell1fit.events
   :no-inheritance-diagram:
.. automodapi:: ell1fit.weighting
   :no-inheritance-diagram:
.. automodapi:: ell1fit.outputs
   :no-inheritance-diagram:
.. automodapi:: ell1fit.results_io
   :no-inheritance-diagram:
.. automodapi:: ell1fit.create_parfile
   :no-inheritance-diagram:
.. automodapi:: ell1fit.update_binary
   :no-inheritance-diagram:
.. automodapi:: ell1fit.eccentricity
   :no-inheritance-diagram:
.. automodapi:: ell1fit.orbit_plot
   :no-inheritance-diagram:
.. automodapi:: ell1fit.mcmc_utils
   :no-inheritance-diagram:
.. automodapi:: ell1fit.profile_plotting
   :no-inheritance-diagram:
.. automodapi:: ell1fit.plotting
   :no-inheritance-diagram:
.. automodapi:: ell1fit.cli
   :no-inheritance-diagram:
