Quickstart
==========

Basic Usage
-----------

Simple Example
~~~~~~~~~~~~~~

.. code-block:: python

   from sargeom import Cartographic

   # Create coordinate in Paris
   paris = Cartographic(longitude=2.3522, latitude=48.8566, height=35.0)
   
   # Convert to ECEF and back
   paris_ecef = paris.to_ecef()
   paris_geo = paris_ecef.to_cartographic()

Multiple Points
~~~~~~~~~~~~~~~

.. code-block:: python
   
   # Multiple cities
   cities = Cartographic(
       longitude=[2.3522, -0.1276, 139.6917],  # Paris, London, Tokyo
       latitude=[48.8566, 51.5074, 35.6895],
       height=[35.0, 11.0, 40.0]
   )
   
   # Convert all to ECEF
   cities_ecef = cities.to_ecef()

Local Coordinate Systems
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Reference point (Paris)
   ref = Cartographic(longitude=2.3522, latitude=48.8566, height=35.0)
   
   # Target point (London)
   london = Cartographic(longitude=-0.1276, latitude=51.5074, height=11.0)
   london_ecef = london.to_ecef()
   
   # Convert to local ENU relative to Paris
   london_enu = london_ecef.to_enu(origin=ref)

Trajectories
~~~~~~~~~~~~

.. code-block:: python

   from sargeom import Trajectory
   import numpy as np

   # Time vector
   time = np.linspace(0, 60, 100)  # 60 seconds, 100 points
   
   # Aircraft trajectory
   coords = Cartographic(
       longitude=2.0 + 0.01 * time,
       latitude=48.0 + 0.005 * time,
       height=1000 + 10 * np.sin(0.1 * time)
   )
   
   # Create trajectory
   traj = Trajectory(
      timestamps=time,
      positions=coords
   )

Distance Calculations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sargeom import Cartesian3

   # Two points
   point1 = Cartographic(longitude=2.3522, latitude=48.8566, height=0)
   point2 = Cartographic(longitude=-0.1276, latitude=51.5074, height=0)
   
   # Convert to ECEF for distance calculation
   p1_ecef = point1.to_ecef()
   p2_ecef = point2.to_ecef()
   
   # Euclidean distance
   distance = Cartesian3.distance(p2_ecef, p1_ecef)
   print(f"Distance: {distance/1000:.1f} km")

Advanced Usage
--------------

Large Dataset Processing
~~~~~~~~~~~~~~~~~~~~~~~~

Efficient processing of large coordinate arrays:

.. code-block:: python

   # Generate large dataset
   n_points = 100000
   
   # Random coordinates
   coords = Cartographic(
       longitude=np.random.uniform(-180, 180, n_points),
       latitude=np.random.uniform(-90, 90, n_points),
       height=np.random.uniform(0, 10000, n_points)
   )
   
   # Batch conversion (much faster than individual conversions)
   ecef_coords = coords.to_ecef()

Working with Grids
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a grid of coordinates
   lon_grid, lat_grid = np.meshgrid(
       np.linspace(2.0, 3.0, 10),   # Longitude range
       np.linspace(48.0, 49.0, 10)  # Latitude range
   )
   
   # Flatten for coordinate creation
   grid_coords = Cartographic(
       longitude=lon_grid.flatten(),
       latitude=lat_grid.flatten(),
       height=np.zeros(100)  # on IAG GRS 80 ellipsoid
   )

   # Convert all points to ECEF at once
   grid_ecef = grid_coords.to_ecef()

Next Steps
----------

- Explore the :doc:`examples` for more detailed use cases
- Check the :doc:`api_reference` for complete function documentation
