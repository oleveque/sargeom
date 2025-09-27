import numpy as np


class LambertConicConformal:
    """
    Lambert Conic Conformal projection class.
    
    This class implements the Lambert Conic Conformal projection, a conformal
    map projection that preserves angles and is commonly used for mapping
    regions in the middle latitudes with primarily east-west extent.
    
    Parameters
    ----------
    ellipsoid : :class:`sargeom.coordinates.Ellipsoid`
        Reference ellipsoid used for the projection.
    lon_origin_rad : :class:`float`
        Longitude of the origin of the Lambert Conic Conformal projection in radians.
    lat_origin_rad : :class:`float` 
        Latitude of the origin of the Lambert Conic Conformal projection in radians.
    scale : :class:`float`, optional
        Scale factor of the projection. Default is 1.0.
    x_offset_m : :class:`float`, optional
        Offset from the projection origin in the x-direction in meters. Default is 0.0.
    y_offset_m : :class:`float`, optional
        Offset from the projection origin in the y-direction in meters. Default is 0.0.
        
    Attributes
    ----------
    ellipsoid : :class:`sargeom.coordinates.Ellipsoid`
        The reference ellipsoid.
    lon_origin_rad : :class:`float`
        The longitude origin in radians.
    lat_origin_rad : :class:`float`
        The latitude origin in radians.
    scale : :class:`float`
        The scale factor.
    x_offset_m : :class:`float`
        The x-offset in meters.
    y_offset_m : :class:`float`
        The y-offset in meters.
    """
    def __init__(
            self,
            ellipsoid, lon_origin_rad, lat_origin_rad,
            scale=1.0, x_offset_m=0.0, y_offset_m=0.0
        ):
        R0 = (
            scale * ellipsoid.prime_vertical_curvature_radius(lat_origin_rad)
                / np.tan(lat_origin_rad)
        )
        self.ellipsoid = ellipsoid
        self.lon_origin_rad = lon_origin_rad
        self.lat_origin_rad = lat_origin_rad
        self.scale = scale
        self.x_offset_m = x_offset_m
        self.y_offset_m = y_offset_m + R0

        # Precompute constants
        self._sin_lat_origin = np.sin(self.lat_origin_rad)
        self._C = R0 * np.exp(
            self._sin_lat_origin * self.ellipsoid.isometric_latitude(self.lat_origin_rad)
        )

    def forward(self, lon_rad, lat_rad):
        """
        Forward transformation from geographic coordinates to projected coordinates.
        
        Transforms geographic coordinates (longitude, latitude) to projected coordinates 
        (x, y) in the Lambert Conic Conformal projection.

        Parameters
        ----------
        lon_rad : array_like
            The geographic longitude in radians.
        lat_rad : array_like
            The geographic latitude in radians.
            
        Returns
        -------
        x_m : :class:`numpy.ndarray`
            The projected x-coordinate in meters.
        y_m : :class:`numpy.ndarray`
            The projected y-coordinate in meters.
            
        Notes
        -----
        The transformation uses the standard Lambert Conic Conformal projection 
        formulas with the ellipsoid parameters and projection constants.
        """
        theta = self._sin_lat_origin * (lon_rad - self.lon_origin_rad)
        R = self._C * np.exp(-self._sin_lat_origin * self.ellipsoid.isometric_latitude(lat_rad))
        x_m = self.x_offset_m + R * np.sin(theta)
        y_m = self.y_offset_m - R * np.cos(theta)
        return x_m, y_m

    def inverse(self, x_m, y_m):
        """
        Inverse transformation from projected coordinates to geographic coordinates.
        
        Transforms projected coordinates (x, y) in the Lambert Conic Conformal 
        projection to geographic coordinates (longitude, latitude).

        Parameters
        ----------
        x_m : array_like
            The projected x-coordinate in meters.
        y_m : array_like
            The projected y-coordinate in meters.
            
        Returns
        -------
        lon_rad : :class:`numpy.ndarray`
            The geographic longitude in radians.
        lat_rad : :class:`numpy.ndarray`
            The geographic latitude in radians.
            
        Notes
        -----
        The transformation uses the inverse Lambert Conic Conformal projection 
        formulas with the ellipsoid parameters and projection constants.
        """
        if self._sin_lat_origin >= 0.0:
            dx = x_m - self.x_offset_m
            dy = self.y_offset_m - y_m
        else:
            dx = self.x_offset_m - x_m
            dy = y_m - self.y_offset_m
        # Longitude
        lon_rad = self.lon_origin_rad + np.arctan2(dx, dy) / self._sin_lat_origin
        # Latitude
        lat_rad = self.ellipsoid.inverse_isometric_latitude(
            -np.log(np.abs(np.hypot(dx, dy) / self._C)) / self._sin_lat_origin
        )
        return lon_rad, lat_rad


if __name__ == "__main__":
    import doctest
    doctest.testmod()