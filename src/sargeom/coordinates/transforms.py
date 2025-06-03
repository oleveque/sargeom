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
