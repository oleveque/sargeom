SAR Geometry
============

.. toctree::
   :hidden:

   Installation Guide <installation_guide.rst>
   API Reference <api_reference.rst>
   Examples <examples.rst>

``sargeom`` is a Python package designed for Synthetic Aperture Radar (SAR) geometry calculations.
This guide will walk you through the steps required to set up and start using the package in your projects.

Installation
------------

You can install the package directly from GitHub using ``pip``:

.. code-block:: bash

   pip install git+https://github.com/oleveque/sargeom.git

For more detailed installation instructions, please refer to the `Installation Guide <installation_guide.rst>`_.

Usage
-----

To use the package, you can import it in your Python code as follows:

.. code-block:: python

   import sargeom.coordinates import Cartographic, CartesianECEF

   # Define a geographic coordinate (longitude, latitude, height)
   geo_coord = Cartographic(longitude=2.2945, latitude=48.8584, height=100.0)

   # Convert to ECEF coordinates
   ecef_coord = geo_coord.to_ecef()

   # Convert back to geographic coordinates
   geo_coord_converted = ecef_coord.to_cartographic()

For more examples of how to use the package, please refer to the `Examples <examples.rst>`_ section.

Reporting Issues
----------------

If you encounter any problems, please `file an issue <https://github.com/oleveque/sargeom/issues>`_ with a detailed description.
Your feedback is valuable in improving the package.