"""
This module allows you to manipulate and transform terrestrial coordinates in the WGS84 geodetic system.
By default, distances are expressed in meters and angles in degrees.
"""
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
import pyproj

# Coordinate systems
wgs84_EGM96 = pyproj.crs.CRS.from_epsg(
    "4326+5773" # EPSG:9707 = WGS84 Geographic 2D coordinate system (GCS) + EGM96 height (= Gravity-related height)
)
wgs84_ECEF = pyproj.crs.CRS.from_epsg(
    "4978" # EPSG:4978 = WGS84 Geocentric 3D coordinate system (ECEF = Earth-centered, Earth-fixed coordinate system)
)
wgs84_GCS = pyproj.crs.CRS.from_epsg(
    "4979" # EPSG:4979 = WGS84 Geographic 3D coordinate system (GCS)
)

# Coordinate transformations
ecef2gcs = pyproj.Transformer.from_crs(
    crs_from=wgs84_ECEF, crs_to=wgs84_GCS
)
gcs2ecef = pyproj.Transformer.from_crs(
    crs_from=wgs84_GCS, crs_to=wgs84_ECEF
)
gcs2egm = pyproj.Transformer.from_crs(
    crs_from=wgs84_GCS, crs_to=wgs84_EGM96
)


def negativePiToPi(angle, degrees=True):
    """
    Converts angles to the range from -180 to 180 degrees.

    Parameters
    ----------
    angle : :class:`float` or array_like
        The input angle or a list/array of angles.
    degrees : bool, optional
        If True (default), takes input angles in degrees and returns the angle in degrees. If False, takes input radians and returns the angle in radians.

    Returns
    -------
    :class:`float` or :class:`numpy.ndarray`
        The converted angle or an array of converted angles (in degrees or radians).

    Examples
    --------
    >>> negativePiToPi(190)
    -170.0

    >>> negativePiToPi([-190, 190])
    array([ 170., -170.])
    """
    # Transforms arrays and scalars of type ndarray into lists or integers
    if isinstance(angle, np.ndarray):
        angle = angle.tolist()

    if isinstance(angle, list):
        return np.array([negativePiToPi(a) for a in angle])
    else:
        if degrees:
            if -180.0 <= angle <= 180.0:
                return angle
            else:
                return (angle + 180.0) % 360.0 - 180.0
        else:
            if -np.pi <= angle <= np.pi:
                return angle
            else:
                return (angle + np.pi) % (2 * np.pi) - np.pi


class Cartesian3(np.ndarray):
    """
    A Cartesian3 object represents the coordinates of a point or a vector in a 3D Cartesian coordinate system.
    This class is inspired by the `CesiumJS library <https://cesium.com/learn/cesiumjs/ref-doc/Cartesian3.html>`_.

    Parameters
    ----------
    x : :class:`float` or :class:`numpy.ndarray`
        The X component, in meters.
    y : :class:`float` or :class:`numpy.ndarray`
        The Y component, in meters.
    z : :class:`float` or :class:`numpy.ndarray`
        The Z component, in meters.
    origin : :class:`sargeom.coordinates.Cartographic`, optional
        The cartographic position describing the location of the local origin of the coordinate system.
        If the cartesian coordinate system used is not a local systems such as ENU, NED, and AER, this parameter is None.

    Raises
    ------
    :class:`ValueError`
        If the X, Y and Z components are not of equal size.
        If the X, Y and Z components are not 0- or 1-dimensional arrays.

    Returns
    -------
    :class:`sargeom.coordinates.Cartesian3`
        The 3D cartesian point.

    Examples
    --------
    Define a single XYZ Cartesian3 point:

    >>> Cartesian3(x=1.0, y=2.0, z=3.0)
    XYZ Cartesian3 point
    [1. 2. 3.]

    Define a set of XYZ Cartesian3 points:

    >>> Cartesian3(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0], z=[7.0, 8.0, 9.0])
    XYZ Cartesian3 points
    [[1. 4. 7.]
     [2. 5. 8.]
     [3. 6. 9.]]

    Slice a Cartesian3 instance:

    >>> A = Cartesian3(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0], z=[7.0, 8.0, 9.0])
    >>> A[1]
    XYZ Cartesian3 point
    [2. 5. 8.]
    >>> A[1:]
    XYZ Cartesian3 points
    [[2. 5. 8.]
     [3. 6. 9.]]

    Perform arithmetic operations on Cartesian3 points:

    >>> A = Cartesian3.UNIT_X()
    >>> B = Cartesian3.ONE()
    >>> A + B
    XYZ Cartesian3 point
    [2. 1. 1.]
    >>> A - B
    XYZ Cartesian3 point
    [ 0. -1. -1.]
    >>> A * B
    XYZ Cartesian3 point
    [1. 0. 0.]
    >>> A / B
    XYZ Cartesian3 point
    [1. 0. 0.]
    """

    def __new__(cls, x, y, z, origin=None):
        # Convert input to numpy arrays
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(z, np.ndarray):
            z = np.array(z)

        # Check if the input arrays have the same shape
        if x.shape != y.shape != z.shape:
            raise ValueError("The X, Y and Z components must be of equal size.")

        # Check if the input arrays are 0- or 1-dimensional
        if x.ndim == y.ndim == z.ndim == 0:
            obj = np.array([[x], [y], [z]]).T.view(cls)
        elif x.ndim == y.ndim == z.ndim == 1:
            obj = np.array([x, y, z]).T.view(cls)
        else:
            raise ValueError(
                "The X, Y and Z components must be 0- or 1-dimensional arrays."
            )

        # Set the local origin of the coordinate system
        obj._local_origin = origin

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        
        # Set the local origin of the coordinate system
        self._local_origin = getattr(obj, "_local_origin", None)

    def __repr__(self):
        """
        Returns a string representation of the XYZ Cartesian3 point(s).

        Returns
        -------
        :class:`str`
            A string representation of the XYZ Cartesian3 point(s)).
        """
        if self.is_collection():
            return f"XYZ {self.__class__.__name__} points\n{self.__array__().__str__()}"
        else:
            return f"XYZ {self.__class__.__name__} point\n{self.__array__().squeeze().__str__()}"

    def __getitem__(self, key):
        """
        Allows access to the Cartesian3 element(s) using the bracket notation.

        Parameters
        ----------
        key : :class:`int`, :class:`slice`, or :class:`tuple`
            The index or indices of the element(s) to access.

        Returns
        -------
        :class:`Cartesian3`
            The element(s) at the specified index or indices.
        """
        return self.from_array(self.__array__()[key], self._local_origin)

    @classmethod
    def from_array(cls, array, origin=None):
        """
        Initializes a Cartesian3 instance using a numpy array representing XYZ coordinates.

        Parameters
        ----------
        array : array_like
            A numpy array object representing a list of XYZ coordinates.
        origin : :class:`sargeom.coordinates.Cartographic`, optional
            The cartographic position describing the location of the local origin of the coordinate system.
            If the cartesian coordinate system used is not a local systems such as ENU, NED, and AER, this parameter is None.
            If not specified, the default local origin of the instance will be used.

        Raises
        ------
        :class:`ValueError`
            If the numpy array has not at least 1 row and only 3 columns.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            The Cartesian3 instance initialized by the input numpy array.

        Examples
        --------
        >>> array = np.array([1.0, 2.0, 3.0])
        >>> Cartesian3.from_array(array)
        XYZ Cartesian3 point
        [1. 2. 3.]

        >>> array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> Cartesian3.from_array(array)
        XYZ Cartesian3 points
        [[1. 2. 3.]
         [4. 5. 6.]]
        """
        # Convert input to numpy array
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        else:
            array = array.__array__()

        # Check if the input array has one dimension and three elements
        if array.ndim == 1 and array.shape[0] == 3:
            return cls(array[0], array[1], array[2], origin)
        
        # Check if the input array has two dimensions and three columns
        elif array.ndim == 2 and array.shape[1] == 3:
            return cls(array[:, 0], array[:, 1], array[:, 2], origin)
        
        # Raise an error if the input array does not meet the requirements
        else:
            raise ValueError(
                "The numpy array must have at least 1 row and only 3 columns."
            )

    @property
    def x(self):
        """
        The X component, in meters.

        Returns
        -------
        :class:`float` or :class:`numpy.ndarray`
            The X component.

        Examples
        --------
        >>> A = Cartesian3(x=10.0, y=20.0, z=30.0)
        >>> A.x
        array(10.)

        >>> A = Cartesian3(x=[10.0, 20.0, 30.0], y=[40.0, 50.0, 60.0], z=[70.0, 80.0, 90.0])
        >>> A.x
        array([10., 20., 30.])
        """
        return self.__array__()[:, 0].squeeze()

    @property
    def y(self):
        """
        The Y component, in meters.

        Returns
        -------
        :class:`float` or :class:`numpy.ndarray`
            The Y component.

        Examples
        --------
        >>> A = Cartesian3(x=10.0, y=20.0, z=30.0)
        >>> A.y
        array(20.)

        >>> A = Cartesian3(x=[10.0, 20.0, 30.0], y=[40.0, 50.0, 60.0], z=[70.0, 80.0, 90.0])
        >>> A.y
        array([40., 50., 60.])
        """
        return self.__array__()[:, 1].squeeze()

    @property
    def z(self):
        """
        The Z component, in meters.

        Returns
        -------
        :class:`float` or :class:`numpy.ndarray`
            The Z component.

        Examples
        --------
        >>> A = Cartesian3(x=10.0, y=20.0, z=30.0)
        >>> A.z
        array(30.)

        >>> A = Cartesian3(x=[10.0, 20.0, 30.0], y=[40.0, 50.0, 60.0], z=[70.0, 80.0, 90.0])
        >>> A.z
        array([70., 80., 90.])
        """
        return self.__array__()[:, 2].squeeze()

    @property
    def local_origin(self):
        """
        Returns the cartographic position of the origin of the local reference system, if the cartesian coordinate system used is one.
        To obtain its expression in the cartesian ECEF reference system, use the following
        method :meth:`sargeom.coordinates.Cartesian3.to_ecef` on the result.

        Raises
        ------
        :class:`ValueError`
            If the Cartesian coordinate system used is not a local system.

        Returns
        -------
        :class:`sargeom.coordinates.Cartographic`
            The cartographic position of the origin of the local reference system used.
        """
        if self.is_local():
            if self._local_origin is None:
                raise ValueError(
                    "The origin of the local Cartesian coordinate system is not defined."
                )
            else:
                return self._local_origin
        else:
            raise ValueError(
                "The Cartesian coordinate system used is not a local system."
            )

    @classmethod
    def UNIT_X(cls, N=(), origin=None):
        """
        A Cartesian3 instance initialized to (x=1.0, y=0.0, z=0.0).

        Parameters
        ----------
        N : :class:`int`, optional
            Number of points to initialize. The default is only 1.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            Instance initialized to (x=1.0, y=0.0, z=0.0).

        Examples
        --------
        >>> Cartesian3.UNIT_X()
        XYZ Cartesian3 point
        [1. 0. 0.]

        >>> Cartesian3.UNIT_X(3)
        XYZ Cartesian3 points
        [[1. 0. 0.]
         [1. 0. 0.]
         [1. 0. 0.]]
        """
        return cls(np.ones(N), np.zeros(N), np.zeros(N), origin)

    @classmethod
    def UNIT_Y(cls, N=(), origin=None):
        """
        A Cartesian3 instance initialized to (x=0.0, y=1.0, z=0.0).

        Parameters
        ----------
        N : :class:`int`, optional
            Number of points to initialize. The default is only 1.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            Instance initialized to (x=0.0, y=1.0, z=0.0).

        Examples
        --------
        >>> Cartesian3.UNIT_Y()
        XYZ Cartesian3 point
        [0. 1. 0.]

        >>> Cartesian3.UNIT_Y(3)
        XYZ Cartesian3 points
        [[0. 1. 0.]
         [0. 1. 0.]
         [0. 1. 0.]]
        """
        return cls(np.zeros(N), np.ones(N), np.zeros(N), origin)

    @classmethod
    def UNIT_Z(cls, N=(), origin=None):
        """
        A Cartesian3 instance initialized to (x=0.0, y=0.0, z=1.0).

        Parameters
        ----------
        N : :class:`int`, optional
            Number of points to initialize. The default is only 1.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            Instance initialized to (x=0.0, y=0.0, z=1.0).

        Examples
        --------
        >>> Cartesian3.UNIT_Z()
        XYZ Cartesian3 point
        [0. 0. 1.]

        >>> Cartesian3.UNIT_Z(3)
        XYZ Cartesian3 points
        [[0. 0. 1.]
         [0. 0. 1.]
         [0. 0. 1.]]
        """
        return cls(np.zeros(N), np.zeros(N), np.ones(N), origin)

    @classmethod
    def ONE(cls, N=(), origin=None):
        """
        A Cartesian3 instance initialized to (x=1.0, y=1.0, z=1.0).

        Parameters
        ----------
        N : :class:`int`, optional
            Number of points to initialize. The default is only 1.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            Instance initialized to (x=1.0, y=1.0, z=1.0).

        Examples
        --------
        >>> Cartesian3.ONE()
        XYZ Cartesian3 point
        [1. 1. 1.]

        >>> Cartesian3.ONE(3)
        XYZ Cartesian3 points
        [[1. 1. 1.]
         [1. 1. 1.]
         [1. 1. 1.]]
        """
        return cls(np.ones(N), np.ones(N), np.ones(N), origin)

    @classmethod
    def ZERO(cls, N=(), origin=None):
        """
        A Cartesian3 instance initialized to (x=0.0, y=0.0, z=0.0).

        Parameters
        ----------
        N : :class:`int`, optional
            Number of points to initialize. The default is only 1.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            Instance initialized to (x=0.0, y=0.0, z=0.0).

        Examples
        --------
        >>> Cartesian3.ZERO()
        XYZ Cartesian3 point
        [0. 0. 0.]

        >>> Cartesian3.ZERO(3)
        XYZ Cartesian3 points
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
        """
        return cls(np.zeros(N), np.zeros(N), np.zeros(N), origin)

    def is_collection(self):
        """
        Check if the Cartographic instance represents a set of cartesian points.

        Returns
        -------
        :class:`bool`
            `true` if the instance is a collections of points, `false` otherwise.
        
        Examples
        --------
        >>> A = Cartesian3(x=10.0, y=20.0, z=30.0)
        >>> A.is_collection()
        False

        >>> B = Cartesian3(x=[10.0, 20.0, 30.0], y=[40.0, 50.0, 60.0], z=[70.0, 80.0, 90.0])
        >>> B.is_collection()
        True
        """
        return self.shape[0] > 1

    def is_local(self):
        """
        Returns `true` if the cartesian coordinate system is local and the local origin is defined, `false` otherwise.

        Returns
        -------
        :class:`bool`
            `true` if the cartesian coordinate system is local and the local origin is defined, `false` otherwise.
        
        Examples
        --------
        >>> A = Cartesian3(x=10.0, y=20.0, z=30.0)
        >>> A.is_local()
        False
        """
        return (self._local_origin is not None) or ("Local" in self.__class__.__name__)

    def append(self, positions):
        """
        Create a new Cartesian3 instance with the appended positions.

        Parameters
        ----------
        positions : sequence of :class:`sargeom.coordinates.Cartesian3`
            The sequence of Cartesian3 instances to append.

        Raises
        ------
        :class:`ValueError`
            If the instance to append is not a Cartesian3 instance.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            The new Cartesian3 instance with the appended positions.

        Examples
        --------
        >>> A = Cartesian3(x=10.0, y=20.0, z=30.0)
        >>> B = Cartesian3(x=15.0, y=25.0, z=35.0)
        >>> A.append(B)
        XYZ Cartesian3 points
        [[10. 20. 30.]
         [15. 25. 35.]]
        """
        if np.all([isinstance(c, self.__class__) for c in positions]):
            return self.from_array(np.concatenate((self.__array__(), positions.__array__()), axis=0), self._local_origin)
        else:
            raise ValueError(
                f"The instance to append must be a {self.__class__.__name__} instance."
            )

    def __eq__(self, right):
        """
        Compares this cartesian point against the provided one componentwise and returns `True` if they are equal, `False` otherwise.

        Parameters
        ----------
        right : :class:`sargeom.coordinates.Cartesian3`
            The cartesian point to compare against.

        Returns
        -------
        :class:`bool`
            `True` if the cartesian points are equal, `False` otherwise.

        Examples
        --------
        >>> A = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> B = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> A == B
        True

        >>> A = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> B = Cartesian3(x=2.0, y=3.0, z=4.0)
        >>> A == B
        False
        """
        if self.is_local() or right.is_local():
            return bool(np.all([self.x == right.x, self.y == right.y, self.z == right.z, self._local_origin == right._local_origin]))
        else:
            return bool(np.all([self.x == right.x, self.y == right.y, self.z == right.z]))

    def magnitude(self):
        """
        Computes the magnitude (length) of the supplied cartesian vector.

        Returns
        -------
        :class:`float` or :class:`ndarray`
            The magnitude.

        Examples
        --------
        >>> A = Cartesian3(x=3.0, y=4.0, z=0.0)
        >>> A.magnitude()
        array(5.)
        """
        return np.linalg.norm(self.__array__(), axis=1).squeeze()

    def normalize(self):
        """
        Computes the normalized form of the supplied cartesian point.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            The normalized cartesian point.

        Notes
        -----
        The normalized cartesian point is obtained by dividing each component by the Euclidean norm of the point.

        Examples
        --------
        >>> A = Cartesian3(x=3.0, y=4.0, z=0.0)
        >>> A.normalize()
        XYZ Cartesian3 point
        [0.6 0.8 0. ]
        """
        return self / np.linalg.norm(self.__array__(), axis=1)[:, None]

    def proj_onto(self, vector):
        """
        Projects the vector described by the current instance onto the provided vector.

        Parameters
        ----------
        vector : :class:`sargeom.coordinates.Cartesian3`
            The vector to project onto.

        Raises
        ------
        :class:`ValueError`
            If the vector is not a single Cartesian3 instance.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            The projection vector.

        Notes
        -----
        The projection vector is obtained by computing the dot product of the current instance with the provided vector,
        dividing by the magnitude of the provided vector, and multiplying the result by the normalized provided vector.

        Examples
        --------
        >>> A = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> B = Cartesian3(x=2.0, y=3.0, z=4.0)
        >>> A.proj_onto(B) # doctest: +ELLIPSIS
        XYZ Cartesian3 point
        [ 7.427... 11.141... 14.855...]
        """
        # Check if the vector is a single Cartesian3 instance
        if vector.is_collection():
            raise ValueError("The vector must be a single Cartesian3 instance.")
        else:
            return vector.normalize() * vector.dot(self)[:, None]
    
    def reject_from(self, vector):
        """
        Rejects the vector described by the current instance from the provided vector.

        Parameters
        ----------
        vector : :class:`sargeom.coordinates.Cartesian3`
            The vector to reject from.

        Raises
        ------
        :class:`ValueError`
            If the vector is not a single Cartesian3 instance.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            The rejection vector.

        Notes
        -----
        The rejection vector is obtained by subtracting the projection vector from the current instance.

        Examples
        --------
        >>> A = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> B = Cartesian3(x=2.0, y=3.0, z=4.0)
        >>> A.reject_from(B) # doctest: +ELLIPSIS
        XYZ Cartesian3 point
        [ -6.427...  -9.141... -11.855...]
        """
        return self - self.proj_onto(vector)

    def interp(self, time_sampling, new_time_sampling):
        """
        Interpolates the set of Cartesian coordinates on the basis of the sampling times supplied as input.

        Parameters
        ----------
        time_sampling : :class:`numpy.ndarray`
            Sampling time at which coordinates are acquired.
        new_time_sampling : :class:`numpy.ndarray`
            Sampling time at which coordinates will be estimated.

        Raises
        ------
        :class:`ValueError`
            If this Cartesian3 instance is not a set of cartesian points.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            Estimated cartesian coordinates.

        Notes
        -----
        Interpolation is performed separately for each component (*X*, *Y*, and *Z*) using numpy's `interp` function.

        Examples
        --------
        >>> time = np.array([0.0, 1.0, 2.0])
        >>> positions = Cartesian3(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0], z=[7.0, 8.0, 9.0])
        >>> new_time = np.array([0.5, 1.5])
        >>> positions.interp(time, new_time)
        XYZ Cartesian3 points
        [[1.5 4.5 7.5]
         [2.5 5.5 8.5]]
        """
        if self.is_collection():
            x = np.interp(new_time_sampling, time_sampling, self.x)
            y = np.interp(new_time_sampling, time_sampling, self.y)
            z = np.interp(new_time_sampling, time_sampling, self.z)
            return self.__class__(x, y, z, self._local_origin)
        else:
            raise ValueError(
                "This Cartesian3 instance must be a set of cartesian points."
            )

    def to_pandas(self):
        """
        Converts cartesian point coordinates into a Pandas DataFrame.

        Returns
        -------
        :class:`pandas.DataFrame`
            A Pandas object initialized with the point coordinates.

        Examples
        --------
        >>> positions = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> positions.to_pandas()
             x    y    z
        0  1.0  2.0  3.0
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Pandas is not installed. Please follow the instructions on https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html"
            )

        return pd.DataFrame(self.__array__(), columns=["x", "y", "z"])

    def to_array(self):
        """
        Converts cartesian point coordinates into a numpy array.

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array initialized with the point coordinates.

        Examples
        --------
        >>> position = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> position.to_array()
        array([1., 2., 3.])
        """
        return self.__array__().squeeze()
    
    def save_csv(self, filename):
        """
        Saves the cartesian point coordinates to a CSV file.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The name of the file to save the coordinates.

        Examples
        --------
        >>> positions = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> positions.save_csv("positions.csv")
        """
        filename = Path(filename)
        np.savetxt(
            filename.with_suffix(".csv"),
            self.__array__(),
            fmt=3*['%.6f'],
            delimiter=';',
            newline='\n',
            comments='',
            encoding='utf8',
            header=f"""# {filename.stem}
# Fields descriptions:
# -------------------
#    o Positions as 3D cartesian coordinates:
#        - X_M [m]: The X component in meters.
#        - Y_M [m]: The Y component in meters.
#        - Z_M [m]: The Z component in meters.

X_M;Y_M;Z_M"""
        )

    def cross(self, right):
        """
        Computes the cross (outer) product with a provided cartesian point.

        Parameters
        ----------
        right : :class:`sargeom.coordinates.Cartesian3`
            The provided Cartesian point.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            The cross (outer) product with this cartesian point and the provided one.

        Examples
        --------
        >>> A = Cartesian3.UNIT_X()
        >>> B = Cartesian3.UNIT_Y()
        >>> A.cross(B)
        XYZ Cartesian3 point
        [0. 0. 1.]

        >>> A = Cartesian3(x=[1.0, 0.0, 0.0], y=[0.0, 1.0, 0.0], z=[0.0, 0.0, 1.0])
        >>> B = Cartesian3(x=[0.0, 1.0, 0.0], y=[0.0, 0.0, 1.0], z=[1.0, 0.0, 0.0])
        >>> A.cross(B)
        XYZ Cartesian3 points
        [[ 0. -1.  0.]
         [ 0.  0. -1.]
         [-1.  0.  0.]]
        """
        return self.from_array(np.cross(self.__array__(), right.__array__()), self._local_origin)

    def dot(self, right):
        """
        Computes the dot (scalar) product with a provided cartesian point.

        Parameters
        ----------
        right : :class:`sargeom.coordinates.Cartesian3`
            The provided Cartesian point.

        Returns
        -------
        :class:`numpy.ndarray`
            The dot (scalar) product with this cartesian point and the provided one.

        Examples
        --------
        >>> A = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> B = Cartesian3(x=4.0, y=5.0, z=6.0)
        >>> A.dot(B)
        array([32.])

        >>> A = Cartesian3(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0], z=[7.0, 8.0, 9.0])
        >>> B = Cartesian3(x=[10.0, 20.0, 30.0], y=[40.0, 50.0, 60.0], z=[70.0, 80.0, 90.0])
        >>> A.dot(B)
        array([ 660.,  930., 1260.])
        """
        return np.multiply(self.__array__(), right.__array__()).sum(axis=1)

    def centroid(self):
        """
        Computes the centroid of a set of cartesian points.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            The centroid of the set of cartesian points.

        Examples
        --------
        >>> positions = Cartesian3(
        ...     x=[1.0, 2.0, 3.0],
        ...     y=[4.0, 5.0, 6.0],
        ...     z=[7.0, 8.0, 9.0]
        ... )
        >>> positions.centroid()
        XYZ Cartesian3 point
        [2. 5. 8.]
        """
        return self.from_array(np.mean(self.__array__(), axis=0), self._local_origin)

    @staticmethod
    def distance(left, right):
        """
        Computes the distance between two cartesian points.

        Parameters
        ----------
        left : :class:`sargeom.coordinates.Cartesian3`
            The first Cartesian point.
        right : :class:`sargeom.coordinates.Cartesian3`
            The second Cartesian point.

        Returns
        -------
        :class:`numpy.ndarray`
           The distance between these two cartesian points.

        Examples
        --------
        >>> A = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> B = Cartesian3(x=4.0, y=5.0, z=6.0)
        >>> Cartesian3.distance(A, B) # doctest: +ELLIPSIS
        array([5.196...])

        >>> A = Cartesian3(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0], z=[7.0, 8.0, 9.0])
        >>> B = Cartesian3(x=[4.0, 5.0, 6.0], y=[7.0, 8.0, 9.0], z=[10.0, 11.0, 12.0])
        >>> Cartesian3.distance(A, B) # doctest: +ELLIPSIS
        array([5.196..., 5.196..., 5.196...])
        """
        return np.linalg.norm(left.__array__() - right.__array__(), axis=1)

    @staticmethod
    def middle(left, right):
        """
        Computes the midpoint between two cartesian points.

        Parameters
        ----------
        left : :class:`sargeom.coordinates.Cartesian3`
            The first cartesian point.
        right : :class:`sargeom.coordinates.Cartesian3`
            The second cartesian point.

        Returns
        -------
        :class:`sargeom.coordinates.Cartesian3`
            The midpoint between these two cartesian points.

        Examples
        --------
        >>> A = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> B = Cartesian3(x=4.0, y=5.0, z=6.0)
        >>> Cartesian3.middle(A, B)
        XYZ Cartesian3 point
        [2.5 3.5 4.5]

        >>> A = Cartesian3(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0], z=[7.0, 8.0, 9.0])
        >>> B = Cartesian3(x=[4.0, 5.0, 6.0], y=[7.0, 8.0, 9.0], z=[10.0, 11.0, 12.0])
        >>> Cartesian3.middle(A, B)
        XYZ Cartesian3 points
        [[ 2.5  5.5  8.5]
         [ 3.5  6.5  9.5]
         [ 4.5  7.5 10.5]]
        """
        return (left + right) / 2

    @staticmethod
    def angle_btw(left, right, degrees=True):
        """
        Returns the angle formed by two coordinate vectors (in degrees).

        Parameters
        ----------
        left : :class:`sargeom.coordinates.Cartesian3`
            The first cartesian point.
        right : :class:`sargeom.coordinates.Cartesian3`
            The second cartesian point.
        degrees : bool, optional
            If True (default), returns the angle in degrees. If False, returns the angle in radians.

        Returns
        -------
        :class:`numpy.ndarray`
            The angle formed by these two coordinate vectors (in degrees).

        Examples
        --------
        >>> A = Cartesian3(x=1.0, y=2.0, z=3.0)
        >>> B = Cartesian3(x=4.0, y=5.0, z=6.0)
        >>> Cartesian3.angle_btw(A, B) # doctest: +ELLIPSIS
        array([12.933...])

        >>> A = Cartesian3(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0], z=[7.0, 8.0, 9.0])
        >>> B = Cartesian3(x=[4.0, 5.0, 6.0], y=[7.0, 8.0, 9.0], z=[10.0, 11.0, 12.0])
        >>> Cartesian3.angle_btw(A, B) # doctest: +ELLIPSIS
        array([12.195...,  9.076...,  6.982...])
        """
        angle = np.arccos(left.normalize().dot(right.normalize()))

        if degrees:
            return np.rad2deg(angle)
        else:
            return angle


class CartesianECEF(Cartesian3):
    """
    A geocentric Earth-centered Earth-fixed (ECEF) system uses the Cartesian coordinates (*X*, *Y*, *Z*) to represent the 3D components of a position or a vector.
    For example, the ECEF coordinates of Parc des Buttes-Chaumont are (4198945 m, 174747 m, 4781887 m).

    - The positive X-axis intersects the surface of the ellipsoid at 0° latitude and 0° longitude, where the equator meets the prime meridian.
    - The positive Y-axis intersects the surface of the ellipsoid at 0° latitude and 90° longitude.
    - The positive Z-axis intersects the surface of the ellipsoid at 90° latitude and 0° longitude, the North Pole.

    Parameters
    ----------
    x : :class:`float` or :class:`numpy.ndarray`
        The X component, in meters.
    y : :class:`float` or :class:`numpy.ndarray`
        The Y component, in meters.
    z : :class:`float` or :class:`numpy.ndarray`
        The Z component, in meters.

    Notes
    -----
    The coordinate reference system (CRS) used is the GPS satellite navigation and positioning system : WGS84 Geocentric System `EPSG:4978 <https://epsg.org/crs_4978/WGS-84.html>`_.

    Raises
    ------
    :class:`ValueError`
        If the X, Y, and Z components are not of equal size.
        If the X, Y, and Z components are not 0- or 1-dimensional arrays.

    Returns
    -------
    :class:`sargeom.coordinates.CartesianECEF`
        The 3D cartesian point in the Earth-centered Earth-fixed (ECEF) system.

    Examples
    --------
    The ECEF coordinates of Parc des Buttes-Chaumont:

    >>> CartesianECEF(x=4198945, y=174747, z=4781887)
    XYZ CartesianECEF point
    [4198945  174747 4781887]
    """

    @staticmethod
    def crs():
        """
        Returns the WGS84 Geocentric System `EPSG:4978 <https://epsg.org/crs_4978/WGS-84.html>`_.

        Returns
        -------
        :class:`pyproj.crs.CRS`
            A pythonic coordinate reference system (CRS) manager.
        """
        return wgs84_ECEF

    def to_cartographic(self):
        """
        Converts geocentric Earth-centered Earth-fixed (ECEF) coordinates to geodetic coordinates.

        Returns
        -------
        :class:`sargeom.coordinates.Cartographic`
            The geodetic coordinates (*latitude*, *longitude*, *height*).

        Examples
        --------
        >>> ecef_coords = CartesianECEF(x=4198945, y=174747, z=4781887)
        >>> ecef_coords.to_cartographic() # doctest: +ELLIPSIS
        Lon.Lat.Height Cartographic position
        [  2.383...  48.879...  124.847...]
        """
        latitude, longitude, height = ecef2gcs.transform(self.x, self.y, self.z)
        return Cartographic(longitude, latitude, height)

    def to_ned(self, origin):
        """
        Transforms geocentric Earth-centered Earth-fixed (ECEF) coordinates to local North-East-Down (NED) system.

        Parameters
        ----------
        origin : :class:`sargeom.coordinates.Cartographic`
            The origin position in geodetic coordinates of the North-East-Down (NED) system.

        Raises
        ------
        :class:`ValueError`
            If the origin position is not a Cartographic instance.

        Returns
        -------
        :class:`sargeom.coordinates.CartesianLocalNED`
            The transformed coordinates in local North-East-Down (NED) system.

        Examples
        --------
        >>> ecef_coords = CartesianECEF(x=4198945, y=174747, z=4781887)
        >>> origin = Cartographic.ONERA_SDP()
        >>> ecef_coords.to_ned(origin) # doctest: +ELLIPSIS
        XYZ CartesianLocalNED point
        [ 587260.319... -200505.639...   30171.749...]
        """
        if isinstance(origin, Cartographic):
            return (self - origin.to_ecef()).to_nedv(origin)
        else:
            raise ValueError("The origin position must be a Cartographic instance.")

    def to_nedv(self, origin):
        """
        Transforms geocentric Earth-centered Earth-fixed (ECEF) vector coordinates to local North-East-Down (NED) system.

        Parameters
        ----------
        origin : :class:`sargeom.coordinates.Cartographic`
            The origin position in geodetic coordinates of the North-East-Down (NED) system.

        Raises
        ------
        :class:`ValueError`
            If the origin position is not a Cartographic instance.

        Returns
        -------
        :class:`sargeom.coordinates.CartesianLocalNED`
            The transformed vector coordinates in local North-East-Down (NED) system.

        Examples
        --------
        >>> ecef_vector = CartesianECEF(x=1.0, y=2.0, z=3.0)
        >>> origin = Cartographic.ONERA_SDP()
        >>> ecef_vector.to_nedv(origin) # doctest: +ELLIPSIS
        XYZ CartesianLocalNED point
        [ 1.361...  1.902... -2.919...]
        """
        if isinstance(origin, Cartographic):
            new_array = CartesianLocalNED.ZERO(origin=origin).rotation.apply(self.__array__())
            return CartesianLocalNED.from_array(new_array, origin)
        else:
            raise ValueError("The origin position must be a Cartographic instance.")

    def to_enu(self, origin):
        """
        Transforms geocentric Earth-centered Earth-fixed (ECEF) coordinates to local East-North-Up (ENU) system.

        Parameters
        ----------
        origin : :class:`sargeom.coordinates.Cartographic`
            The origin position in geodetic coordinates of the East-North-Up (ENU) system.

        Raises
        ------
        :class:`ValueError`
            If the origin position is not a Cartographic instance.

        Returns
        -------
        :class:`sargeom.coordinates.CartesianLocalENU`
            The transformed coordinates in local East-North-Up (ENU) system.

        Examples
        --------
        >>> ecef_coords = CartesianECEF(x=4198945, y=174747, z=4781887)
        >>> origin = Cartographic.ONERA_SDP()
        >>> ecef_coords.to_enu(origin) # doctest: +ELLIPSIS
        XYZ CartesianLocalENU point
        [-200505.639...  587260.319...  -30171.749...]
        """
        if isinstance(origin, Cartographic):
            return (self - origin.to_ecef()).to_enuv(origin)
        else:
            raise ValueError("The origin position must be a Cartographic instance.")

    def to_enuv(self, origin):
        """
        Transforms geocentric Earth-centered Earth-fixed (ECEF) vector coordinates to local East-North-Up (ENU) system.

        Parameters
        ----------
        origin : :class:`sargeom.coordinates.Cartographic`
            The origin position in geodetic coordinates of the East-North-Up (ENU) system.

        Raises
        ------
        :class:`ValueError`
            If the origin position is not a Cartographic instance.

        Returns
        -------
        :class:`sargeom.coordinates.CartesianLocalENU`
            The transformed vector coordinates in local East-North-Up (ENU) system.

        Examples
        --------
        >>> ecef_vector = CartesianECEF(x=1.0, y=2.0, z=3.0)
        >>> origin = Cartographic.ONERA_SDP()
        >>> ecef_vector.to_enuv(origin) # doctest: +ELLIPSIS
        XYZ CartesianLocalENU point
        [1.902... 1.361... 2.919...]
        """
        if isinstance(origin, Cartographic):
            new_array = CartesianLocalENU.ZERO(origin=origin).rotation.apply(self.__array__())
            return CartesianLocalENU.from_array(new_array, origin)
        else:
            raise ValueError("The origin position must be a Cartographic instance.")

    def save_csv(self, filename):
        """
        Saves the cartesian ECEF coordinates to a CSV file.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The name of the file to save the coordinates.

        Examples
        --------
        >>> positions = CartesianECEF(x=4198945, y=174747, z=4781887)
        >>> positions.save_csv("positions.csv")
        """
        filename = Path(filename)
        np.savetxt(
            filename.with_suffix(".csv"),
            self.__array__(),
            fmt=3*['%.6f'],
            delimiter=';',
            newline='\n',
            comments='',
            encoding='utf8',
            header=f"""# {filename.stem}
# Fields descriptions:
# -------------------
#    o Positions as 3D cartesian coordinates in the WGS84 Geocentric System (EPSG:4978):
#        - X_WGS84_M [m]: The X component in meters.
#        - Y_WGS84_M [m]: The Y component in meters.
#        - Z_WGS84_M [m]: The Z component in meters.

X_WGS84_M;Y_WGS84_M;Z_WGS84_M"""
        )


class CartesianLocalENU(Cartesian3):
    """
    A local East-North-Up (ENU) system uses the Cartesian coordinates (*xEast*, *yNorth*, *zUp*) to represent position relative to a local origin.
    The local origin is described by the geodetic coordinates (*lat0*, *lon0*, *h0*).
    Note that the origin does not necessarily lie on the surface of the ellipsoid.

    Parameters
    ----------
    xEast : :class:`float` or :class:`numpy.ndarray`
        The positive X-axis points east along the parallel of latitude containing *lat0*, in meters.
    yNorth : :class:`float` or :class:`numpy.ndarray`
        The positive Y-axis points north along the meridian of longitude containing *lon0*, in meters.
    zUp : :class:`float` or :class:`numpy.ndarray`
        The positive Z-axis points upward along the ellipsoid normal, in meters.
    origin : :class:`sargeom.coordinates.Cartographic`
        The origin position in geodetic coordinates of the East-North-Up (ENU) system.

    Attributes
    ----------
    rotation : :class:`scipy.spatial.transform.Rotation`
        The 3D Rotation SciPy instance for transforming geocentric ECEF coordinates of a vector to local ENU coordinates.

    Examples
    --------
    >>> CartesianLocalENU(10.0, 20.0, 30.0, origin=Cartographic.ONERA_SDP())
    XYZ CartesianLocalENU point
    [10. 20. 30.]
    """

    @property
    def rotation(self):
        """
        Get the 3D Rotation SciPy instance for transforming geocentric ECEF coordinates of a vector to local ENU coordinates.

        Returns
        -------
        :class:`scipy.spatial.transform.Rotation`
            The 3D Rotation SciPy instance.

        Examples
        --------
        >>> enu_coords = CartesianLocalENU(10.0, 20.0, 30.0, origin=Cartographic.ONERA_SDP())
        >>> enu_coords.rotation.as_matrix() # doctest: +ELLIPSIS
        array([[-0.089...,  0.996...,  0.        ],
               [-0.687..., -0.061...,  0.723...],
               [ 0.721...,  0.064...,  0.689...]])
        >>> enu_coords.rotation.as_quat() # doctest: +ELLIPSIS
        array([ 0.265...,  0.290...,  0.678..., -0.620...])
        """
        lat0, lon0 = np.deg2rad(
            [self._local_origin.latitude, self._local_origin.longitude]
        )
        return Rotation.from_matrix(
            [
                [-np.sin(lon0), np.cos(lon0), 0.0],
                [
                    -np.cos(lon0) * np.sin(lat0),
                    -np.sin(lon0) * np.sin(lat0),
                    np.cos(lat0),
                ],
                [
                    np.cos(lon0) * np.cos(lat0),
                    np.sin(lon0) * np.cos(lat0),
                    np.sin(lat0),
                ],
            ]
        )

    def to_ecef(self):
        """
        Converts local East-North-Up (ENU) coordinates to Earth-centered Earth-fixed (ECEF) coordinates.

        Returns
        -------
        :class:`sargeom.coordinates.CartesianECEF`
            The geocentric ECEF coordinates.

        Examples
        --------
        >>> enu_coords = CartesianLocalENU(10.0, 20.0, 30.0, origin=Cartographic.ONERA_SDP())
        >>> enu_coords.to_ecef() # doctest: +ELLIPSIS
        XYZ CartesianECEF point
        [4606335.623...  412550.865... 4377594.950...]
        """
        if self._local_origin is None:
            raise ValueError(
                "The origin of the local Cartesian coordinate system is not defined."
            )
        else:
            new_array = self.rotation.inv().apply(self.__array__())
            return CartesianECEF.from_array(new_array) + self._local_origin.to_ecef()

    def to_ned(self):
        """
        Converts local East-North-Up (ENU) coordinates to local North-East-Down (NED) coordinates.
        Both coordinate systems use the same local origin.

        Returns
        -------
        :class:`sargeom.coordinates.CartesianLocalNED`
            The local NED coordinates.

        Examples
        --------
        >>> enu_coords = CartesianLocalENU(10.0, 20.0, 30.0, origin=Cartographic.ONERA_SDP())
        >>> enu_coords.to_ned()
        XYZ CartesianLocalNED point
        [ 20.  10. -30.]
        """
        return CartesianLocalNED(self.y, self.x, -self.z, self._local_origin)

    def to_aer(self, degrees=True):
        """
        Transforms local East-North-Up (ENU) coordinates to local Azimuth-Elevation-Range (AER) spherical coordinates.
        Both coordinate systems use the same local origin.

        An azimuth-elevation-range (AER) system uses the spherical coordinates (*az*, *elev*, *range*) to represent position relative to a local origin.
        The local origin is described by the geodetic coordinates (*lat0*, *lon0*, *h0*).
        Azimuth, elevation, and slant range are dependent on the local ENU Cartesian system.

        Parameters
        ----------
        degrees : bool, optional
            If True (default), returns the angle in degrees. If False, returns the angle in radians.

        Returns
        -------
        azimuth : :class:`numpy.ndarray`
            The azimuth, the clockwise angle in the xEast-yNorth plane from the positive yNorth-axis to the projection of the object into the plane.
        elevation : :class:`numpy.ndarray`
            The elevation, the angle from the xEast-yNorth plane to the object.
        slant_range : :class:`numpy.ndarray`
            The slant range (in meters), the Euclidean distance between the object and the local origin.

        Examples
        --------
        >>> enu_coords = CartesianLocalENU(10.0, 20.0, 30.0, origin=Cartographic.ONERA_SDP())
        >>> azimuth, elevation, slant_range = enu_coords.to_aer()
        >>> print(f"{azimuth}°", f"{elevation}°", f"{slant_range}m") # doctest: +ELLIPSIS
        26.565...° 53.300...° 37.416...m
        """
        azimuth = np.arctan2(self.x, self.y)
        elevation = np.arctan2(self.z, np.sqrt(self.x**2 + self.y**2))
        slant_range = np.sqrt(self.x**2 + self.y**2 + self.z**2)

        if degrees:
            return np.rad2deg(azimuth), np.rad2deg(elevation), slant_range
        else:
            return azimuth, elevation, slant_range

    def save_csv(self, filename):
        """
        Saves the cartesian ENU coordinates to a CSV file.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The name of the file to save the coordinates.

        Examples
        --------
        >>> positions = CartesianLocalENU(x=10.0, y=20.0, z=30.0, origin=Cartographic.ONERA_SDP())
        >>> positions.save_csv("positions.csv")
        """
        filename = Path(filename)
        np.savetxt(
            filename.with_suffix(".csv"),
            self.__array__(),
            fmt=3*['%.6f'],
            delimiter=';',
            newline='\n',
            comments='',
            encoding='utf8',
            header=f"""# {filename.stem}
# Fields descriptions:
# -------------------
#    o Positions as 3D cartesian coordinates in the Local East-North-Up (ENU) system:
#        - X_ENU_M [m]: The X component in meters.
#        - Y_ENU_M [m]: The Y component in meters.
#        - Z_ENU_M [m]: The Z component in meters.
#
# Local origin of the ENU system:
# ------------------------------
#    o Position in WG84 Geocentric System (EPSG:4979):
#        - Latitude [°]: {self._local_origin.latitude}
#        - Longitude [°]: {self._local_origin.longitude}
#        - Height [m]: {self._local_origin.height}

X_ENU_M;Y_ENU_M;Z_ENU_M"""
        )


class CartesianLocalNED(Cartesian3):
    """
    A North-East-Down (NED) system uses the Cartesian coordinates (*xNorth*, *yEast*, *zDown*) to represent position relative to a local origin.
    The local origin is described by the geodetic coordinates (*lat0*, *lon0*, *h0*).
    Note that the origin does not necessarily lie on the surface of the ellipsoid.

    Parameters
    ----------
    xNorth : :class:`float` or :class:`numpy.ndarray`
        The positive X-axis points north along the meridian of longitude containing *lon0*, in meters.
    yEast : :class:`float` or :class:`numpy.ndarray`
        The positive Y-axis points east along the parallel of latitude containing *lat0*, in meters.
    zDown : :class:`float` or :class:`numpy.ndarray`
        The positive Z-axis points downward along the ellipsoid normal, in meters.
    origin : :class:`sargeom.coordinates.Cartographic`
        The origin position in geodetic coordinates of the North-East-Down (NED) system.

    Attributes
    ----------
    rotation : :class:`scipy.spatial.transform.Rotation`
        The 3D Rotation SciPy instance for transforming geocentric ECEF coordinates of a vector to local NED coordinates.

    Examples
    --------
    >>> CartesianLocalNED(10.0, 20.0, 30.0, origin=Cartographic.ONERA_SDP())
    XYZ CartesianLocalNED point
    [10. 20. 30.]
    """

    @property
    def rotation(self):
        """
        Get the 3D Rotation SciPy instance for transforming geocentric ECEF coordinates of a vector to local NED coordinates.

        Returns
        -------
        :class:`scipy.spatial.transform.Rotation`
            The 3D Rotation SciPy instance.

        Examples
        --------
        >>> ned_coords = CartesianLocalNED(10.0, 20.0, 30.0, origin=Cartographic.ONERA_SDP())
        >>> ned_coords.rotation.as_matrix() # doctest: +ELLIPSIS
        array([[-0.687..., -0.061...,  0.723...],
               [-0.089...,  0.996...,  0.        ],
               [-0.721..., -0.064..., -0.689...]])
        >>> ned_coords.rotation.as_quat() # doctest: +ELLIPSIS
        array([-0.041...,  0.918..., -0.017...,  0.393...])
        """
        lat0, lon0 = np.deg2rad(
            [self._local_origin.latitude, self._local_origin.longitude]
        )
        return Rotation.from_matrix(
            [
                [
                    -np.cos(lon0) * np.sin(lat0),
                    -np.sin(lon0) * np.sin(lat0),
                    np.cos(lat0),
                ],
                [
                    -np.sin(lon0),
                    np.cos(lon0),
                    0.0
                ],
                [
                    -np.cos(lon0) * np.cos(lat0),
                    -np.sin(lon0) * np.cos(lat0),
                    -np.sin(lat0),
                ],
            ]
        )

    def to_ecef(self):
        """
        Converts local North-East-Down (NED) coordinates to Earth-centered Earth-fixed (ECEF) coordinates.

        Returns
        -------
        :class:`sargeom.coordinates.CartesianECEF`
            The geocentric ECEF coordinates.

        Examples
        --------
        >>> ned_coords = CartesianLocalNED(10.0, 20.0, 30.0, origin=Cartographic.ONERA_SDP())
        >>> ned_coords.to_ecef() # doctest: +ELLIPSIS
        XYZ CartesianECEF point
        [4606298.339...  412557.566... 4377546.319...]
        """
        if self._local_origin is None:
            raise ValueError(
                "The origin of the local Cartesian coordinate system is not defined."
            )
        else:
            new_array = self.rotation.inv().apply(self.__array__())
            return CartesianECEF.from_array(new_array) + self._local_origin.to_ecef()

    def to_enu(self):
        """
        Converts local North-East-Down (NED) coordinates to local East-North-Up (ENU) coordinates.
        Both coordinate systems use the same local origin.

        Returns
        -------
        :class:`sargeom.coordinates.CartesianLocalENU`
            The local NED coordinates.

        Examples
        --------
        >>> ned_coords = CartesianLocalNED(10.0, 20.0, 30.0, origin=Cartographic.ONERA_SDP())
        >>> ned_coords.to_enu()
        XYZ CartesianLocalENU point
        [ 20.  10. -30.]
        """
        return CartesianLocalENU(self.y, self.x, -self.z, self._local_origin)

    def to_aer(self, degrees=True):
        """
        Transforms local North-East-Down (NED) coordinates to local Azimuth-Elevation-Range (AER) spherical coordinates.
        Both coordinate systems use the same local origin.

        An azimuth-elevation-range (AER) system uses the spherical coordinates (*az*, *elev*, *range*) to represent position relative to a local origin.
        The local origin is described by the geodetic coordinates (*lat0*, *lon0*, *h0*).
        Azimuth, elevation, and slant range are dependent on the local ENU Cartesian system.

        Parameters
        ----------
        degrees : bool, optional
            If True (default), returns the angle in degrees. If False, returns the angle in radians.

        Returns
        -------
        azimuth : :class:`numpy.ndarray`
            The azimuth, the clockwise angle in the xEast-yNorth plane from the positive yNorth-axis to the projection of the object into the plane.
        elevation : :class:`numpy.ndarray`
            The elevation, the angle from the xEast-yNorth plane to the object.
        slant_range : :class:`numpy.ndarray`
            The slant range (in meters), the Euclidean distance between the object and the local origin.

        Examples
        --------
        >>> ned_coords = CartesianLocalNED(10.0, 20.0, 30.0, origin=Cartographic.ONERA_SDP())
        >>> azimuth, elevation, slant_range = ned_coords.to_aer()
        >>> print(f"{azimuth}°", f"{elevation}°", f"{slant_range}m") # doctest: +ELLIPSIS
        63.434...° 53.300...° 37.416...m
        """
        azimuth = np.arctan2(self.y, self.x)
        elevation = np.arctan2(self.z, np.sqrt(self.x**2 + self.y**2))
        slant_range = np.sqrt(self.x**2 + self.y**2 + self.z**2)

        if degrees:
            return np.rad2deg(azimuth), np.rad2deg(elevation), slant_range
        else:
            return azimuth, elevation, slant_range

    def save_csv(self, filename):
        """
        Saves the cartesian NED coordinates to a CSV file.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The name of the file to save the coordinates.

        Examples
        --------
        >>> positions = CartesianLocalNED(x=10.0, y=20.0, z=30.0, origin=Cartographic.ONERA_SDP())
        >>> positions.save_csv("positions.csv")
        """
        filename = Path(filename)
        np.savetxt(
            filename.with_suffix(".csv"),
            self.__array__(),
            fmt=3*['%.6f'],
            delimiter=';',
            newline='\n',
            comments='',
            encoding='utf8',
            header=f"""# {filename.stem}
# Fields descriptions:
# -------------------
#    o Positions as 3D cartesian coordinates in the Local North-East-Down (NED) system:
#        - X_NED_M [m]: The X component in meters.
#        - Y_NED_M [m]: The Y component in meters.
#        - Z_NED_M [m]: The Z component in meters.
#
# Local origin of the NED system:
# ------------------------------
#    o Position in WG84 Geocentric System (EPSG:4979):
#        - Latitude [°]: {self._local_origin.latitude}
#        - Longitude [°]: {self._local_origin.longitude}
#        - Height [m]: {self._local_origin.height}

X_NED_M;Y_NED_M;Z_NED_M"""
        )


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
        if longitude.shape != latitude.shape != height.shape:
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

    @staticmethod
    def crs():
        """
        Returns the WGS84 Geocentric System `EPSG:4979 <https://epsg.org/crs_4979/WGS-84.html>`_.

        Returns
        -------
        :class:`pyproj.crs.CRS`
            A pythonic coordinate reference system (CRS) manager.
        """
        return wgs84_GCS

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
        origin : :class:`sargeom.coordinates.Cartographic`, optional
            The cartographic position describing the location of the local origin of the coordinate system.
            If the cartesian coordinate system used is not a local systems such as ENU, NED, and AER, this parameter is None.
            If not specified, the default local origin of the instance will be used.
        degrees : bool, optional
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

        Parameters
        ----------
        positions : :class:`sargeom.coordinates.Cartographic`
            The sequence of Cartographic instances to append.

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
        """
        if np.all([isinstance(c, Cartographic) for c in positions]):
            return self.from_array(np.concatenate((self.__array__(), positions.__array__()), axis=0))
        else:
            raise ValueError("The instance to append must be a Cartographic instance.")

    def __eq__(self, right):
        """
        Compares this Cartographic instance against the provided one componentwise and returns `True` if they are equal, `False` otherwise.
        """
        return np.all(
            [
                self.longitude == right.longitude,
                self.latitude == right.latitude,
                self.height == right.height,
            ]
        )

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
        x, y, z = gcs2ecef.transform(self.latitude, self.longitude, self.height)
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
        lat, lon, alt = gcs2egm.transform(
            self.__array__()[:, 1], # latitude array
            self.__array__()[:, 0], # longitude array
            self.__array__()[:, 2] # height array
        )
        for coords in zip(lon, lat, alt):
            pnt = kml.newpoint()
            pnt.name = "Lat {lat:.6f}, Lon: {lon:.6f}, Alt: {alt:.6f}".format(
                lat=coords[1], lon=coords[0], alt=coords[2]
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
            fmt=['%.12f','%.12f','%.6f'],
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