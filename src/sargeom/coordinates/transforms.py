import numpy as np
from sargeom.coordinates.ellipsoids import ELPS_CLARKE_1880, ELPS_PAM_WGS84

# Custom transformation parameters cartesian ECEF NTF to cartesian ECEF WGS84
_ANGLE_Z_NTF_TO_WGS84_RAD = np.deg2rad(0.554 / 3600.0)
_DX_NTF_TO_WGS84_M = -168.0
_DY_NTF_TO_WGS84_M = -72.0
_DZ_NTF_TO_WGS84_M = 318.5

class LambertConicConformal:
    """
    Lambert Conic Conformal projection class.
    
    This class implements the Lambert Conic Conformal projection, a conformal
    map projection that preserves angles and is commonly used for mapping
    regions in the middle latitudes with primarily east-west extent.
    
    Parameters
    ----------
    ellipsoid : :class:`sargom.coordinates.ellipsoids.Ellipsoid`
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
    ellipsoid : :class:`sargom.coordinates.ellipsoids.Ellipsoid`
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


def transform_trajectory_from_local_lambert_ntf_to_wgs84(header, records):
    """
    Transforms trajectory coordinates from local Lambert NTF projection to WGS84 geographic CRS
    with "custom" parameters.
    
    This function performs a multi-step transformation:
    1. From local Lambert NTF to geographic NTF
    2. From geographic NTF to cartesian ECEF NTF
    3. From cartesian ECEF NTF to cartesian ECEF WGS84 (custom)
    4. From cartesian ECEF WGS84 to geographic WGS84
    
    Parameters
    ----------
    header : array_like
        Header containing origin coordinates [lon_origin_ntf_rad, lat_origin_ntf_rad, height_origin_ntf_m].
    records : dict
        Dictionary containing trajectory records with keys 'longitude_rad', 'latitude_rad', 'height_m'.
        
    Returns
    -------
    records : dict
        Updated records with transformed coordinates in WGS84 geographic CRS.
    """
    # Unpack NTF trajectory origin coordinates
    lon_origin_ntf_rad = header[0]
    lat_origin_ntf_rad = header[1]
    height_origin_ntf_m = header[2]

    # Unpack position records (note: attitude angles are kept unchanged)
    x_loc_m = records['longitude_rad']
    y_loc_m = records['latitude_rad']
    height_ntf_m = records['height_m']

    # Lambert Conic Conformal projection initialization
    locLambertNTF = LambertConicConformal(
        ELPS_CLARKE_1880,
        lon_origin_ntf_rad,
        lat_origin_ntf_rad
    )

    # Step 1: from local Lambert NTF to geographic NTF
    lon_ntf_rad, lat_ntf_rad = locLambertNTF.inverse(x_loc_m, y_loc_m)

    # Step 2: Transform from geographic NTF to cartesian ECEF NTF
    x_ntf_m, y_ntf_m, z_ntf_m = ELPS_CLARKE_1880.to_ecef(
        lon_ntf_rad, lat_ntf_rad, height_ntf_m + height_origin_ntf_m
    )

    # Step 3: Transform from cartesian ECEF NTF to cartesian ECEF WGS84 (custom)
    x_wgs84_m = 1.0000002198 * x_ntf_m - _ANGLE_Z_NTF_TO_WGS84_RAD * y_ntf_m + _DX_NTF_TO_WGS84_M
    y_wgs84_m = 1.0000002198 * y_ntf_m + _ANGLE_Z_NTF_TO_WGS84_RAD * x_ntf_m + _DY_NTF_TO_WGS84_M
    z_wgs84_m = 1.0000002198 * z_ntf_m                                       + _DZ_NTF_TO_WGS84_M

    # Step 4: Transform from cartesian ECEF WGS84 to geographic WGS84
    lon_wgs84_rad, lat_wgs84_rad, height_wgs84_m = ELPS_PAM_WGS84.to_cartographic(
        x_wgs84_m, y_wgs84_m, z_wgs84_m
    )

    # Update records geographic coordinates
    records['longitude_rad'] = lon_wgs84_rad
    records['latitude_rad'] = lat_wgs84_rad
    records['height_m'] = height_wgs84_m

    return records


if __name__ == "__main__":
    import doctest
    doctest.testmod()