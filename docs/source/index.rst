SAR Geometry
============

.. toctree::
   :hidden:

   quickstart
   api_reference
   examples

``sargeom`` is a Python package designed for Synthetic Aperture Radar (SAR) geometry calculations 
and coordinate system transformations.

Installation
------------

**Quick Install**

You can install the package directly from GitHub using ``pip``:

.. code-block:: bash

   pip install git+https://github.com/oleveque/sargeom.git@latest

**Development Install**

For development or to access examples and documentation:

.. code-block:: bash

   git clone https://github.com/oleveque/sargeom.git
   cd sargeom
   pip install -e .[dev,docs,examples]

**Adding to Dependencies**

Add to your ``pyproject.toml``:

.. code-block:: toml

   [project]
   dependencies = [
       "sargeom @ git+https://github.com/oleveque/sargeom@v0.4.0"
   ]

Or to your ``requirements.txt`` file:

.. code-block:: text

   sargeom @ git+https://github.com/oleveque/sargeom@v0.4.0

For more information on the latest updates, check the `CHANGELOG <https://github.com/oleveque/sargeom/blob/main/CHANGELOG.md>`_.

Quick Examples
--------------

**Basic Coordinate Conversion**

.. code-block:: python

   from sargeom.coordinates import Cartographic, CartesianECEF

   # Create geographic coordinate (Paris)
   paris = Cartographic(longitude=2.3522, latitude=48.8566, height=35.0)
   
   # Convert to ECEF
   paris_ecef = paris.to_ecef()
   print(f"ECEF coordinates: {paris_ecef}")

**Multiple Points**

.. code-block:: python

   # Process multiple cities at once
   cities = Cartographic(
       longitude=[2.3522, -0.1276, 139.6917],  # Paris, London, Tokyo
       latitude=[48.8566, 51.5074, 35.6895],
       height=[35.0, 11.0, 40.0]
   )
   cities_ecef = cities.to_ecef()

**Local Coordinate Systems**

.. code-block:: python

   # Convert to local East-North-Up (ENU) coordinates
   origin = Cartographic(longitude=2.3522, latitude=48.8566, height=35.0)
   london = Cartographic(longitude=-0.1276, latitude=51.5074, height=11.0)
   london_ecef = london.to_ecef()
   london_enu = london_ecef.to_enu(origin=origin)

**SAR Trajectory Modeling**

.. code-block:: python

   from sargeom import Trajectory
   import numpy as np

   # Create trajectory points
   timestamps = np.array([0, 1, 2, 3])
   positions = Cartographic(
       longitude=[3.8777, 4.8391, 5.4524, 6.2345],
       latitude=[43.6135, 43.9422, 43.5309, 43.7891],
       height=[300.0, 400.0, 500.0, 600.0]
   )
   
   # Create trajectory
   traj = Trajectory(timestamps, positions)

See :doc:`examples` for more detailed examples