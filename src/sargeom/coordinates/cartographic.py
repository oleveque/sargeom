from pathlib import Path
import numpy as np

from sargeom.coordinates.transforms import WGS84
from sargeom.coordinates.utils import negativePiToPi


class Cartographic(np.ndarray):
    """
    A Cartographic object represents the position of a point in a geodetic coordinate system.
    This class is inspired by the `CesiumJS library <https://cesium.com/learn/cesiumjs/ref-doc/Cartographic.html>`_.

    Parameters
    ----------
    longitude : :class:`float` or :class:`numpy.ndarray`
        The longitude, in degrees, originates at the prime meridian.
    latitude : :class:`float` or :class:`numpy.ndarray`
        The latitude, in degrees, originates at the equator.
    height : :class:`float` or :class:`numpy.ndarray`, optional
        The height, in meters, above the ellipsoid. The default value is 0.0.
    degrees : bool, optional
        If True (default), takes input angles in degrees. If False, takes input angles in radians.

    Notes
    -----
    The coordinate reference system (CRS) used is the GPS satellite navigation and positioning system : WGS84 Geographic System `EPSG:4979 <https://epsg.org/crs_4979/WGS-84.html>`_.

    Raises
    ------
    :class:`ValueError`
        If the longitude, latitude, and height are not of equal size.
        If the longitude, latitude, and height are not 0- or 1-dimensional arrays.

    Returns
    -------
    :class:`sargeom.coordinates.Cartographic`
        The cartographic position.

    Examples
    --------
    Define a Cartographic position:

    >>> Cartographic(longitude=10.0, latitude=20.0, height=30.0)
    Lon.Lat.Height Cartographic position
    [10. 20. 30.]

    Define multiple Cartographic positions:

    >>> Cartographic(longitude=[15.0, 25.0], latitude=[30.0, 40.0])
    Lon.Lat.Height Cartographic positions
    [[15. 30.  0.]
     [25. 40.  0.]]

    Define a Cartographic position using radians:

    >>> Cartographic(longitude=np.pi/6, latitude=np.pi/3, degrees=False)
    Lon.Lat.Height Cartographic position
    [30. 60.  0.]

    Slice a Cartographic position:
    
    >>> carto = Cartographic(longitude=[10.0, 20.0, 30.0], latitude=[40.0, 50.0, 60.0], height=[70.0, 80.0, 90.0])
    >>> carto[1]
    Lon.Lat.Height Cartographic position
    [20. 50. 80.]
    >>> carto[1:]
    Lon.Lat.Height Cartographic positions
    [[20. 50. 80.]
     [30. 60. 90.]]
    """

    def __new__(cls, longitude, latitude, height=None, degrees=True):
        # Convert to degrees if necessary
        if not degrees:
            longitude = np.rad2deg(longitude)
            latitude = np.rad2deg(latitude)

        # Convert to the range from -180.0 to 180.0 degrees
        longitude = negativePiToPi(longitude)
        latitude = negativePiToPi(latitude)

        # Check if the input arrays are numpy arrays
        if not isinstance(longitude, np.ndarray):
            longitude = np.array(longitude)
        if not isinstance(latitude, np.ndarray):
            latitude = np.array(latitude)
        if height is None:
            height = np.zeros(longitude.shape)
        elif not isinstance(height, np.ndarray):
            height = np.array(height)

        # Check if the input arrays have the same size
        if not (longitude.shape == latitude.shape == height.shape):
            raise ValueError(
                "The longitude, latitude and height must be of equal size."
            )

        # Check if the input arrays are 0- or 1-dimensional
        if longitude.ndim == latitude.ndim == height.ndim == 0:
            obj = np.array([[longitude], [latitude], [height]]).T.view(cls)
        elif longitude.ndim == latitude.ndim == height.ndim == 1:
            obj = np.array([longitude, latitude, height]).T.view(cls)
        else:
            raise ValueError(
                "The longitude, latitude, and height components must be 0- or 1-dimensional arrays."
            )

        return obj

    # @staticmethod
    # def crs():
    #     """
    #     Returns the WGS84 Geocentric System `EPSG:4979 <https://epsg.org/crs_4979/WGS-84.html>`_.

    #     Returns
    #     -------
    #     :class:`pyproj.crs.CRS`
    #         A pythonic coordinate reference system (CRS) manager.
    #     """
    #     return wgs84_GCS

    def __repr__(self):
        """
        Returns a string representation of the Lon.Lat.Height Cartographic position(s).

        Returns
        -------
        :class:`str`
            A string representation of the Lon.Lat.Height Cartographic position(s)).
        """
        if self.is_collection():
            return f"Lon.Lat.Height Cartographic positions\n{self.__array__().__str__()}"
        else:
            return f"Lon.Lat.Height Cartographic position\n{self.__array__().squeeze().__str__()}"

    def __getitem__(self, key):
        """
        Allows access to the Cartographic element(s) using the bracket notation.

        Parameters
        ----------
        key : :class:`int`, :class:`slice`, or :class:`tuple`
            The index or indices of the element(s) to access.

        Returns
        -------
        :class:`sargeom.coordinates.Cartographic`
            The element(s) at the specified index or indices.
        """
        return Cartographic.from_array(self.__array__()[key])

    @staticmethod
    def from_array(array, degrees=True):
        """
        Initializes a Cartographic instance using a numpy array representing Lon-Lat-Height coordinates.

        Parameters
        ----------
        array : array_like
            A numpy array object representing a list of Lon-Lat-Height coordinates.
        degrees : :class:`bool`, optional
            If True (default), takes input angles in degrees. If False, takes input angles in radians.

        Raises
        ------
        :class:`ValueError`
            If the numpy array has not at least 1 row and only 3 columns.

        Returns
        -------
        :class:`sargeom.coordinates.Cartographic`
            The Cartographic instance initialized by the input numpy array.

        Examples
        --------
        >>> array = np.array([10.0, 20.0, 30.0])
        >>> Cartographic.from_array(array)
        Lon.Lat.Height Cartographic position
        [10. 20. 30.]

        >>> array = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
        >>> Cartographic.from_array(array)
        Lon.Lat.Height Cartographic positions
        [[10. 20. 30.]
         [15. 25. 35.]]
        """
        # Check if the input array is a numpy array
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        else:
            array = array.__array__()

        # Check if the input array has one dimension and three elements
        if array.ndim == 1 and array.shape[0] == 3:
            return Cartographic(array[0], array[1], array[2], degrees)
        
        # Check if the input array has two dimensions and three columns
        elif array.ndim == 2 and array.shape[1] == 3:
            return Cartographic(array[:, 0], array[:, 1], array[:, 2], degrees)
        
        # Raise an error if the input array does not meet the requirements
        else:
            raise ValueError(
                "The numpy array must have at least 1 row and only 3 columns."
            )

    @property
    def longitude(self):
        """
        The longitude, in degrees, originates at the prime meridian.

        More specifically, the longitude of a point is the angle that a plane containing the ellipsoid center and the meridian containing that point makes with the plane containing the ellipsoid center and prime meridian.
        Positive longitudes are measured in a counterclockwise direction from a vantage point above the North Pole.
        Typically, longitude is within the range [-180°, 180°] or [0°, 360°].

        Returns
        -------
        :class:`numpy.ndarray`
            The longitude, in degrees.
        
        Examples
        --------
        >>> position = Cartographic(longitude=10.0, latitude=20.0)
        >>> position.longitude
        array(10.)

        >>> positions = Cartographic(longitude=[10.0, 15.0], latitude=[20.0, 25.0])
        >>> positions.longitude
        array([10., 15.])
        """
        return self.__array__()[:, 0].squeeze()

    @property
    def latitude(self):
        """
        The latitude, in degrees, originates at the equator.

        More specifically, the latitude of a point is the angle a normal to the ellipsoid at that point makes with the equatorial plane, which contains the center and equator of the ellipsoid.
        An angle of latitude is within the range [-90°, 90°].
        Positive latitudes correspond to north and negative latitudes correspond to south.

        Returns
        -------
        :class:`numpy.ndarray`
            The latitude, in degrees.
        
        Examples
        --------
        >>> position = Cartographic(longitude=10.0, latitude=20.0)
        >>> position.latitude
        array(20.)

        >>> positions = Cartographic(longitude=[10.0, 15.0], latitude=[20.0, 25.0])
        >>> positions.latitude
        array([20., 25.])
        """
        return self.__array__()[:, 1].squeeze()

    @property
    def height(self):
        """
        The ellipsoidal height, in meters, is measured along a normal of the reference spheroid.

        Returns
        -------
        :class:`numpy.ndarray`
            The ellipsoidal height, in meters.
        
        Examples
        --------
        >>> position = Cartographic(longitude=10.0, latitude=20.0, height=30.0)
        >>> position.height
        array(30.)

        >>> positions = Cartographic(longitude=[10.0, 15.0], latitude=[20.0, 25.0], height=[30.0, 35.0])
        >>> positions.height
        array([30., 35.])
        """
        return self.__array__()[:, 2].squeeze()

    @staticmethod
    def ZERO(N=()):
        """
        A Cartographic instance initialized to (0.0, 0.0, 0.0).

        Parameters
        ----------
        N : :class:`int`, optional
            Number of points to initialize. The default is only 1.

        Returns
        -------
        :class:`sargeom.coordinates.Cartographic`
            Instance initialized to (0.0, 0.0, 0.0).
        
        Examples
        --------
        >>> Cartographic.ZERO()
        Lon.Lat.Height Cartographic position
        [0. 0. 0.]

        >>> Cartographic.ZERO(2)
        Lon.Lat.Height Cartographic positions
        [[0. 0. 0.]
         [0. 0. 0.]]
        """
        return Cartographic(np.zeros(N), np.zeros(N), np.zeros(N))

    @staticmethod
    def ONERA_SDP():
        """
        A cartographic instance initialized at ONERA's position in Salon-de-Provence (5.117724, 43.619212, 0.0).

        Returns
        -------
        :class:`sargeom.coordinates.Cartographic`
            Instance initialized to (longitude=5.117724, latitude=43.619212, height=0.0).
        
        Examples
        --------
        >>> Cartographic.ONERA_SDP()
        Lon.Lat.Height Cartographic position
        [ 5.117724 43.619212  0.      ]
        """
        return Cartographic(longitude=5.117724, latitude=43.619212, height=0.0)

    @staticmethod
    def ONERA_CP():
        """
        A cartographic instance initialized at ONERA's position in Palaiseau (2.230784, 48.713028, 0.0).

        Returns
        -------
        :class:`sargeom.coordinates.Cartographic`
            Instance initialized to (longitude=2.230784, latitude=48.713028, height=0.0).
        
        Examples
        --------
        >>> Cartographic.ONERA_CP()
        Lon.Lat.Height Cartographic position
        [ 2.230784 48.713028  0.      ]
        """
        return Cartographic(longitude=2.230784, latitude=48.713028, height=0.0)

    def append(self, positions):
        """
        Creates a new Cartographic instance with the appended positions.

        This is a convenience method for concatenating this instance with another single instance.
        For concatenating multiple instances, use the concatenate() class method.

        Parameters
        ----------
        positions : :class:`sargeom.coordinates.Cartographic`
            The Cartographic instance to append.

        Raises
        ------
        :class:`ValueError`
            If the instance to append is not a Cartographic instance.

        Examples
        --------
        >>> A = Cartographic(longitude=10.0, latitude=20.0, height=30.0)
        >>> B = Cartographic(longitude=15.0, latitude=25.0, height=35.0)
        >>> A.append(B)
        Lon.Lat.Height Cartographic positions
        [[10. 20. 30.]
         [15. 25. 35.]]
         
        See Also
        --------
        concatenate : Class method for concatenating multiple instances
        """
        if isinstance(positions, Cartographic):
            return self.concatenate([self, positions])
        else:
            raise ValueError("The instance to append must be a Cartographic instance.")

    @classmethod
    def concatenate(cls, positions):
        """
        Concatenate a sequence of Cartographic instances into a single instance.

        Parameters
        ----------
        positions : sequence of :class:`sargeom.coordinates.Cartographic`
            The sequence of Cartographic instances to concatenate. Can be a list, tuple, or any iterable.
        
        Returns
        -------
        :class:`sargeom.coordinates.Cartographic`
            A new Cartographic instance containing all positions from the input instances.

        Raises
        ------
        :class:`ValueError`
            - If the input list is empty.
            - If not all items in the list are instances of Cartographic.

        Examples
        --------
        Concatenate multiple Cartographic positions:

        >>> pos_1 = Cartographic(longitude=[1.0, 2.0], latitude=[45.0, 46.0], height=[100, 200])
        >>> pos_2 = Cartographic(longitude=3.0, latitude=47.0, height=300)
        >>> pos_3 = Cartographic(longitude=[4.0, 5.0], latitude=[48.0, 49.0], height=[400, 500])
        >>> combined = Cartographic.concatenate([pos_1, pos_2, pos_3])
        >>> len(combined)
        5
        >>> combined.longitude
        array([1., 2., 3., 4., 5.])

        Concatenate single positions:

        >>> paris = Cartographic(longitude=2.3522, latitude=48.8566, height=35.0)
        >>> london = Cartographic(longitude=-0.1276, latitude=51.5074, height=11.0)
        >>> cities = Cartographic.concatenate([paris, london])
        >>> len(cities)
        2
        """
        # Convert to list if not already a sequence
        if not hasattr(positions, '__iter__'):
            raise TypeError("positions must be an iterable (list, tuple, etc.)")
        
        positions = list(positions)
        
        # Check if the input list is empty
        if not positions:
            raise ValueError("Input list is empty.")

        # Check if all items in the list are instances of Cartographic
        if not all(isinstance(pos, cls) for pos in positions):
            raise ValueError("All items in the list must be Cartographic instances.")

        # Concatenate the positions into a single Cartographic instance
        return cls.from_array(np.concatenate([pos.__array__() for pos in positions], axis=0))

    def __eq__(self, right):
        """
        Compares this Cartographic instance against the provided one componentwise and returns `True` if they are equal, `False` otherwise.

        Parameters
        ----------
        right : :class:`sargeom.coordinates.Cartographic`
            The cartographic position to compare against.

        Returns
        -------
        :class:`bool`
            `True` if the cartographic positions are equal, `False` otherwise.

        Examples
        --------
        >>> A = Cartographic(longitude=10.0, latitude=20.0, height=30.0)
        >>> B = Cartographic(longitude=10.0, latitude=20.0, height=30.0)
        >>> A == B
        True

        >>> A = Cartographic(longitude=10.0, latitude=20.0, height=30.0)
        >>> B = Cartographic(longitude=15.0, latitude=25.0, height=35.0)
        >>> A == B
        False
        """
        return np.allclose(self.__array__(), right.__array__(), rtol=1e-12, atol=1e-9)

    def is_collection(self):
        """
        Check if the Cartographic instance represents a set of positions.

        Returns
        -------
        :class:`bool`
            `true` if the instance is a collections of positions, `false` otherwise.
        
        Examples
        --------
        >>> position = Cartographic(longitude=10.0, latitude=20.0)
        >>> position.is_collection()
        False

        >>> positions = Cartographic(longitude=[10.0, 15.0], latitude=[20.0, 25.0])
        >>> positions.is_collection()
        True
        """
        return self.shape[0] > 1

    def bounding_box(self):
        """
        Get the bounding box of the Cartographic instance.

        Returns
        -------
        east : :class:`float`
            The maximum longitude.
        west : :class:`float`
            The minimum longitude.
        north : :class:`float`
            The maximum latitude.
        south : :class:`float`
            The minimum latitude.

        Raises
        ------
        :class:`ValueError`
            If the Cartographic instance is not a collection of positions.

        Examples
        --------
        >>> positions = Cartographic(longitude=[10.0, 15.0], latitude=[20.0, 25.0])
        >>> positions.bounding_box()
        (np.float64(15.0), np.float64(10.0), np.float64(25.0), np.float64(20.0))
        """
        if self.is_collection():
            east = self.longitude.max()
            west = self.longitude.min()
            north = self.latitude.max()
            south = self.latitude.min()
            return east, west, north, south
        
        else:
            raise ValueError(
                "This Cartographic instance must be a collection of positions."
            )

    def to_ecef(self):
        """
        Convert Cartographic positions to geocentric Earth-centered Earth-fixed (ECEF) coordinates.

        Returns
        -------
        :class:`sargeom.coordinates.CartesianECEF`
            The corresponding CartesianECEF instance.

        Examples
        --------
        >>> position = Cartographic(longitude=2.230784, latitude=48.713028, height=0.0)
        >>> position.to_ecef() # doctest: +ELLIPSIS
        XYZ CartesianECEF point
        [4213272.203...  164124.695... 4769561.521...]
        """
        from sargeom.coordinates.cartesian import CartesianECEF
        x, y, z = WGS84.to_cartesian_ecef(
            np.deg2rad(self.longitude),
            np.deg2rad(self.latitude),
            self.height
        )
        return CartesianECEF(x, y, z)

    def save_kml(self, filename):
        """
        Saves Cartographic positions to a KML file.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The name of the file to save the positions.

        Notes
        -----
        The ellipsoidal height is converted to orthometric height using the EGM96 geoid model.
        As a result, the output file is compatible with Google Earth and can be easily opened within the application.

        Examples
        --------
        >>> positions = Cartographic(longitude=[10.0, 15.0], latitude=[20.0, 25.0], height=[30.0, 35.0])
        >>> positions.save_kml("my_positions.kml")
        """
        try:
            import simplekml
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "SimpleKML is not installed. Please follow the instructions on https://simplekml.readthedocs.io/en/latest/index.html"
            )

        kml = simplekml.Kml(open=1)  # the folder will be open in the table of contents
        lon    = self.__array__()[:, 0]
        lat    = self.__array__()[:, 1]
        height = self.__array__()[:, 2]
        for coords in zip(lon, lat, height):
            pnt = kml.newpoint()
            pnt.name = "Lon {lon:.6f}, Lat: {lat:.6f}, Height: {height:.6f}".format(
                lon=coords[0], lat=coords[1], height=coords[2]
            )
            pnt.coords = [coords]
        kml.save(Path(filename).with_suffix(".kml"))

    def to_geojson(self, clamp_to_Ground=False, link_markers=False):
        """
        Convert Cartographic positions to GeoJSON format.

        Parameters
        ----------
        clamp_to_Ground : :class:`bool`, optional
            If True, clamp positions to the ground. Default is False.
        link_markers : :class:`bool`, optional
            If True, link markers with a line. Default is False.

        Returns
        -------
        GeoJSON object
            The GeoJSON representation of the Cartographic positions.

        Notes
        -----
        If `link_markers` is True and the list of coordinates is cyclic, a GeoJSON Polygon is returned otherwise a GeoJSON LineString.
        If `link_markers` is False, a GeoJSON MultiPoint is returned.

        Examples
        --------
        >>> position = Cartographic(longitude=10.0, latitude=20.0, height=30.0)
        >>> position.to_geojson()
        {'coordinates': [10.0, 20.0, 30.0], 'type': 'Point'}
        >>> positions = Cartographic(longitude=[10.0, 15.0, 20.0], latitude=[20.0, 25.0, 30.0], height=[30.0, 35.0, 40.0])
        >>> positions.to_geojson()
        {'coordinates': [[10.0, 20.0, 30.0], [15.0, 25.0, 35.0], [20.0, 30.0, 40.0]], 'type': 'MultiPoint'}
        >>> positions.to_geojson(clamp_to_Ground=True, link_markers=True)
        {'coordinates': [[10.0, 20.0], [15.0, 25.0], [20.0, 30.0]], 'type': 'LineString'}
        """
        coords = self.__array__()[:, :2]  if clamp_to_Ground else self.__array__()
        coords = coords.squeeze().tolist()

        if self.is_collection():
            if link_markers:
                if np.all(coords[0] == coords[-1]):
                    return {"coordinates": [coords], "type": "Polygon"}
                else:
                    return {"coordinates": coords, "type": "LineString"}
            else:
                return {"coordinates": coords, "type": "MultiPoint"}
        else:
            return {"coordinates": coords, "type": "Point"}

    def to_trajviewer(self, name=None, description=None, clamp_to_Ground=False, link_markers=False, url="https://oleveque.github.io/trajviewer"):
        """
        Open the Cartographic instance in the TrajViewer web application.

        Parameters
        ----------
        name : :class:`str`, optional
            The name of the Cartographic instance. Default is None.
        description : :class:`str`, optional
            The description of the Cartographic instance. Default is None.
        clamp_to_Ground : :class:`bool`, optional
            If True, clamp positions to the ground. Default is False.
        link_markers : :class:`bool`, optional
            If True, link markers with a line. Default is False.
        url : :class:`str`, optional
            The URL of the TrajViewer web application. Default is "https://oleveque.github.io/trajviewer".

        Notes
        -----
        TrajViewer is a web-based application for visualizing spatial data, such as trajectories, locations, and raster files.
        For more information, visit the GitHub repository: https://github.com/oleveque/trajviewer
        """
        import json
        import base64
        import urllib.parse
        import webbrowser
        
        # Convert the Cartographic instance to GeoJSON format
        geometry = self.to_geojson(clamp_to_Ground=clamp_to_Ground, link_markers=link_markers)

        # Create a GeoJSON Feature with the provided name and description
        geojson = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "name": name,
                "description": description
            }
        }

        # Convert the GeoJSON Feature to a string and encode it in base64
        geojson_str = json.dumps(geojson, separators=(',', ':'))
        geojson_b64 = base64.b64encode(geojson_str.encode('utf-8')).decode('ascii')
        geojson_enc = urllib.parse.quote(geojson_b64)

        # Open the TrajViewer web application with the encoded GeoJSON data
        webbrowser.open(f"{url}?geojsonData={geojson_enc}")

    def to_shapely(self, clamp_to_Ground=False, link_markers=False):
        """
        Converts the Cartographic instance to a Shapely geometry object.

        This method converts the Cartographic instance to an appropriate Shapely geometry object, allowing for
        geometric operations and manipulations commonly performed with Shapely.

        Parameters
        ----------
        clamp_to_Ground : :class:`bool`, optional
            If True, clamp positions to the ground. Default is False.
        link_markers : :class:`bool`, optional
            If True, link markers with a line. Default is False.

        Returns
        -------
        Shapely geometry
            A Shapely geometry object representing the Cartographic instance.

        Notes
        -----
        If `link_markers` is True and the list of coordinates is cyclic, a Shapely Polygon is returned otherwise a Shapely LineString.
        If `link_markers` is False, a Shapely MultiPoint is returned.

        Examples
        --------
        >>> position = Cartographic(longitude=10.0, latitude=20.0, height=30.0)
        >>> position.to_shapely()
        <POINT Z (10 20 30)>
        >>> positions = Cartographic(longitude=[10.0, 15.0, 20.0], latitude=[20.0, 25.0, 30.0], height=[30.0, 35.0, 40.0])
        >>> positions.to_shapely()
        <MULTIPOINT Z ((10 20 30), (15 25 35), (20 30 40))>
        >>> positions.to_shapely(clamp_to_Ground=True, link_markers=True)
        <LINESTRING (10 20, 15 25, 20 30)>
        """
        try:
            from shapely import Point, MultiPoint, LineString, Polygon
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Shapely is not installed. Please follow the instructions on https://shapely.readthedocs.io/en/stable/installation.html"
            )

        coords = self.__array__()[:, :2]  if clamp_to_Ground else self.__array__()
        coords = coords.squeeze().tolist()

        if self.is_collection():
            if link_markers:
                return Polygon([coords]) if np.all(coords[0] == coords[-1]) else LineString(coords)
            else:
                return MultiPoint(coords)
        else:
            return Point(coords)

    def save_html(self, filename, link_markers=False):
        """
        Saves Cartographic positions to an HTML file using Folium.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The name of the file to save the positions.
        link_markers : :class:`bool`, optional
            If True, link markers with a line. Default is False.

        Examples
        --------
        >>> position = Cartographic(longitude=2.230784, latitude=48.713028)
        >>> position.save_html("my_map.html")
        """
        try:
            import folium
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Folium is not installed. Please follow the instructions on https://python-visualization.github.io/folium/latest/getting_started.html"
            )
        
        lon0 = (self.longitude.max() + self.longitude.min()) / 2
        lat0 = (self.latitude.max() + self.latitude.min()) / 2
        m = folium.Map(location=(lat0, lon0))
        if link_markers:
            folium.PolyLine(np.flip(self.__array__()[:, :2], axis=0).tolist()).add_to(m)
        else:
            for position in self.__array__()[:, :2]:
                folium.Marker(
                    np.flip(position).tolist(),
                    popup=folium.Popup(
                        f"Longitude: {position[0]}</br>Latitude: {position[1]}"
                    ),
                ).add_to(m)
        m.save(Path(filename).with_suffix(".html"))

    def save_csv(self, filename):
        """
        Saves the Cartographic positions to a CSV file.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The name of the file to save the positions.

        Examples
        --------
        >>> positions = Cartographic(longitude=[10.0, 15.0], latitude=[20.0, 25.0], height=[30.0, 35.0])
        >>> positions.save_csv("positions.csv")
        """
        filename = Path(filename)
        np.savetxt(
            filename.with_suffix(".csv"),
            self.__array__(),
            fmt=['%.15f','%.15f','%.6f'],
            delimiter=';',
            newline='\n',
            comments='',
            encoding='utf8',
            header=f"""# {filename.stem}
# Fields descriptions:
# -------------------
#    o Positions as 3D geodetic coordinates in the WGS84 Geographic System (EPSG:4979):
#        - LON_WGS84_DEG [°]: The longitude in degrees.
#        - LAT_WGS84_DEG [°]: The latitude in degrees.
#        - HEIGHT_M [m]: The height in meters.

LON_WGS84_DEG;LAT_WGS84_DEG;HEIGHT_M"""
        )

    def to_pandas(self):
        """
        Convert Cartographic positions to a Pandas DataFrame.

        Returns
        -------
        :class:`pandas.DataFrame`
            A Pandas object initialized with the geodetic position(s).

        Examples
        --------
        >>> position = Cartographic(longitude=10.0, latitude=20.0, height=30.0)
        >>> position.to_pandas()
           longitude  latitude  height
        0       10.0      20.0    30.0
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Pandas is not installed. Please follow the instructions on https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html"
            )

        return pd.DataFrame(self.__array__(), columns=["longitude", "latitude", "height"])

    def to_array(self):
        """
        Convert Cartographic positions to a numpy array.

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array initialized with the point positions.

        Examples
        --------
        >>> position = Cartographic(longitude=10.0, latitude=20.0, height=30.0)
        >>> position.to_array()
        array([10., 20., 30.])
        """
        return self.__array__().squeeze()

    @staticmethod
    def dms_to_dd(d, m, s):
        """
        Converts Degrees - Minutes - Seconds representation of geodetic coordinates to decimal degrees.

        Parameters
        ----------
        d : :class:`numpy.ndarray`
            The Degrees value of the geodetic coordinates to be converted.
        m : :class:`numpy.ndarray`
            The Minutes value of the geodetic coordinates to be converted.
        s : :class:`numpy.ndarray`
            The Seconds value of the geodetic coordinates to be converted.

        Returns
        -------
        :class:`numpy.ndarray`
            The decimal degrees expression of the geodetic coordinates.

        Notes
        -----
        The orientation of latitude (North or South) and longitude (West or East) must be passed through the sign of the Degrees value.

        Examples
        --------
        >>> Cartographic.dms_to_dd([-5, 43], [39, 14], [17.114904,  7.709064]) # doctest: +ELLIPSIS
        array([-5.654..., 43.235...])
        """
        return np.sign(d) * (np.abs(d) + np.array(m) / 60 + np.array(s) / 3600)

    @staticmethod
    def dd_to_dms(dd):
        """
        Converts decimal degrees representation of geodetic coordinates to Degrees - Minutes - Seconds.

        Parameters
        ----------
        dd : :class:`numpy.ndarray`
            The decimal degrees expression of the geodetic coordinates to be converted.

        Returns
        -------
        d : :class:`numpy.ndarray`
            The Degrees value of the geodetic coordinates.
        m : :class:`numpy.ndarray`
            The Minutes value of the geodetic coordinates.
        s : :class:`numpy.ndarray`
            The Seconds value of the geodetic coordinates.

        Notes
        -----
        The orientation of latitude (North or South) and longitude (West or East) is returned through the sign of the Degrees value.

        Examples
        --------
        >>> d, m, s = Cartographic.dd_to_dms([-5.65475414, 43.23547474])
        >>> print(f"{d[0]}°", f"{m[0]}'", f'{s[0]}"') # doctest: +ELLIPSIS
        -5° 39' 17.114..."
        >>> print(f"{d[1]}°", f"{m[1]}'", f'{s[1]}"') # doctest: +ELLIPSIS
        43° 14' 7.709..."
        """
        dd = np.array(dd, ndmin=0)
        sign = np.sign(dd)
        dd = np.abs(dd)
        d = np.floor(dd)
        m = np.floor((dd - d) * 60)
        s = ((dd - d) * 60 - m) * 60
        return (sign * d).astype(np.int16), m.astype(np.int16), s


if __name__ == "__main__":
    import doctest
    doctest.testmod()
