SAR Geometry
============

.. toctree::
   :hidden:

   quickstart
   api_reference
   examples

``sargeom`` is a Python package designed for Synthetic Aperture Radar (SAR) geometry calculations.

Installation
------------

You can install the package directly from GitHub using ``pip``:

.. code-block:: bash

   pip install git+https://github.com/oleveque/sargeom.git@latest

If you want to add the package to your dependencies, you can add it to your ``pyproject.toml``:

.. code-block:: toml

   [project]
   dependencies = [
       "sargeom @ git+https://github.com/oleveque/sargeom@v0.3.0"
   ]

Or to your ``requirements.txt`` file:

.. code-block:: text

   sargeom @ git+https://github.com/oleveque/sargeom@v0.3.0

For more information on the latest updates, check the `CHANGELOG <https://github.com/oleveque/sargeom/blob/main/CHANGELOG.md>`_.

Quick Example
-------------

.. code-block:: python

   from sargeom import Cartographic, CartesianECEF

   # Create geographic coordinate
   coord = Cartographic(longitude=2.3522, latitude=48.8566, height=35.0)
   
   # Convert to ECEF
   ecef = coord.to_ecef()
   print(f"ECEF: {ecef}")

See :doc:`examples` for more detailed examples