import numpy as np

from sargeom.coordinates.transforms import Ellipsoid, LambertConicConformal

# Custom Pamela Ellipsoid definitions
_PAM_NTF = Ellipsoid(a=6378249.2, b=6356515.0) # Clarke 1880 (IGN)
_PAM_WGS84 = Ellipsoid(a=6378137.0, b=6356752.3142) # Custom Pamela WGS84
# Custom transformation parameters cartesian ECEF NTF to cartesian ECEF WGS84
_ANGLE_Z_NTF_TO_WGS84_RAD = np.deg2rad(0.554 / 3600.0)
_DX_NTF_TO_WGS84_M = -168.0
_DY_NTF_TO_WGS84_M = -72.0
_DZ_NTF_TO_WGS84_M = 318.5

def transform_trajectory_from_local_lambert_ntf_to_wgs84(header, records):
    r"""
    Transforms trajectory coordinates from local Lambert NTF projection to WGS84 geographic CRS
    with "custom" parameters.
    
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
        _PAM_NTF,
        lon_origin_ntf_rad,
        lat_origin_ntf_rad
    )
    # Step 1: from local Lambert NTF to geographic NTF
    lon_ntf_rad, lat_ntf_rad = locLambertNTF.inverse(x_loc_m, y_loc_m)
    # Step 2: Transform from geographic NTF to cartesian ECEF NTF
    x_ntf_m, y_ntf_m, z_ntf_m = _PAM_NTF.to_cartesian_ecef(
        lon_ntf_rad, lat_ntf_rad, height_ntf_m + height_origin_ntf_m
    )
    # Step 3: Transform from cartesian ECEF NTF to cartesian ECEF WGS84 (custom)
    x_wgs84_m = 1.0000002198 * x_ntf_m - _ANGLE_Z_NTF_TO_WGS84_RAD * y_ntf_m + _DX_NTF_TO_WGS84_M
    y_wgs84_m = 1.0000002198 * y_ntf_m + _ANGLE_Z_NTF_TO_WGS84_RAD * x_ntf_m + _DY_NTF_TO_WGS84_M
    z_wgs84_m = 1.0000002198 * z_ntf_m                                       + _DZ_NTF_TO_WGS84_M
    # Step 4: Transform from cartesian ECEF WGS84 to geographic WGS84
    lon_wgs84_rad, lat_wgs84_rad, height_wgs84_m = _PAM_WGS84.to_cartographic(
        x_wgs84_m, y_wgs84_m, z_wgs84_m
    )
    # Update records geographic coordinates
    records['longitude_rad'] = lon_wgs84_rad
    records['latitude_rad'] = lat_wgs84_rad
    records['height_m'] = height_wgs84_m
    return records
