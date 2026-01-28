"""
Coordinate systems module for SAR geometry calculations.

This module provides classes and functions for manipulating and transforming
terrestrial coordinates in the WGS84 geodetic system. It supports conversions
between geographic (latitude/longitude), geocentric (ECEF), and local
tangent plane (ENU/NED) coordinate systems.

By default, distances are expressed in meters and angles in degrees.
"""
from sargeom.coordinates.cartographic import Cartographic
from sargeom.coordinates.cartesian import Cartesian3, CartesianECEF, CartesianLocalENU, CartesianLocalNED
from sargeom.coordinates.ellipsoids import Ellipsoid, ELPS_WGS84, ELPS_CLARKE_1880
from sargeom.coordinates.transforms import LambertConicConformal
from sargeom.coordinates.utils import negativePiToPi

__all__ = [
    'Cartesian3',
    'CartesianECEF',
    'CartesianLocalENU',
    'CartesianLocalNED',
    'Cartographic',
    'Ellipsoid',
    'ELPS_WGS84',
    'ELPS_CLARKE_1880',
    'LambertConicConformal',
    'negativePiToPi',
]