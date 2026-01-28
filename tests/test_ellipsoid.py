"""
Test suite for the Ellipsoid class.

This module contains unit tests for the Ellipsoid class, covering
ellipsoid creation, coordinate conversions, and auxiliary latitude computations.
"""

import unittest
import numpy as np

from sargeom.coordinates.ellipsoids import Ellipsoid, ELPS_WGS84, ELPS_CLARKE_1880


class TestEllipsoidCreation(unittest.TestCase):
    """Test suite for Ellipsoid creation and initialization."""

    def test_creation_with_flattening(self):
        """Test ellipsoid creation with semi-major axis and flattening."""
        ellipsoid = Ellipsoid(semi_major_axis=6378137.0, flattening=1/298.257223563)
        self.assertAlmostEqual(ellipsoid._a, 6378137.0)
        self.assertAlmostEqual(ellipsoid._f, 1/298.257223563)
        self.assertIsNotNone(ellipsoid._b)

    def test_creation_with_semi_minor_axis(self):
        """Test ellipsoid creation with semi-major and semi-minor axes."""
        ellipsoid = Ellipsoid(semi_major_axis=6378249.2, semi_minor_axis=6356515.0)
        self.assertAlmostEqual(ellipsoid._a, 6378249.2)
        self.assertAlmostEqual(ellipsoid._b, 6356515.0)
        self.assertIsNotNone(ellipsoid._f)

    def test_creation_without_parameters_raises(self):
        """Test that ellipsoid creation without semi_minor_axis or flattening raises ValueError."""
        with self.assertRaises(ValueError):
            Ellipsoid(semi_major_axis=6378137.0)

    def test_wgs84_predefined(self):
        """Test predefined WGS84 ellipsoid."""
        self.assertAlmostEqual(ELPS_WGS84._a, 6378137.0)
        self.assertAlmostEqual(ELPS_WGS84._f, 1/298.257223563)

    def test_clarke_1880_predefined(self):
        """Test predefined Clarke 1880 ellipsoid."""
        self.assertAlmostEqual(ELPS_CLARKE_1880._a, 6378249.2)
        self.assertAlmostEqual(ELPS_CLARKE_1880._b, 6356515.0)


class TestEllipsoidCoordinateConversion(unittest.TestCase):
    """Test suite for Ellipsoid coordinate conversion methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.ellipsoid = ELPS_WGS84

    def test_to_ecef_single_point(self):
        """Test conversion from geodetic to ECEF for a single point."""
        # Paris coordinates
        lon_rad = np.deg2rad(2.3522)
        lat_rad = np.deg2rad(48.8566)
        height_m = 35.0

        x, y, z = self.ellipsoid.to_ecef(lon_rad, lat_rad, height_m)
        
        # Verify approximate ECEF coordinates for Paris
        self.assertAlmostEqual(x, 4200938.0, delta=100)
        self.assertAlmostEqual(y, 172561.0, delta=100)
        self.assertAlmostEqual(z, 4780015.0, delta=100)

    def test_to_ecef_array(self):
        """Test conversion from geodetic to ECEF for multiple points."""
        lon_rad = np.deg2rad([2.3522, -0.1276])
        lat_rad = np.deg2rad([48.8566, 51.5074])
        height_m = np.array([35.0, 11.0])

        x, y, z = self.ellipsoid.to_ecef(lon_rad, lat_rad, height_m)
        
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)
        self.assertEqual(len(z), 2)

    def test_to_cartographic_single_point(self):
        """Test conversion from ECEF to geodetic for a single point."""
        # ECEF coordinates
        x = 4201113.0
        y = 173048.0
        z = 4780015.0

        lon_rad, lat_rad, height_m = self.ellipsoid.to_cartographic(x, y, z)
        
        # Verify approximate geodetic coordinates
        self.assertAlmostEqual(np.rad2deg(lon_rad), 2.36, delta=0.1)
        self.assertAlmostEqual(np.rad2deg(lat_rad), 48.86, delta=0.1)

    def test_roundtrip_conversion(self):
        """Test round-trip conversion geodetic -> ECEF -> geodetic."""
        lon_rad_orig = np.deg2rad(5.0)
        lat_rad_orig = np.deg2rad(45.0)
        height_m_orig = 500.0

        # Convert to ECEF
        x, y, z = self.ellipsoid.to_ecef(lon_rad_orig, lat_rad_orig, height_m_orig)
        
        # Convert back to geodetic
        lon_rad, lat_rad, height_m = self.ellipsoid.to_cartographic(x, y, z)
        
        self.assertAlmostEqual(lon_rad, lon_rad_orig, places=10)
        self.assertAlmostEqual(lat_rad, lat_rad_orig, places=10)
        self.assertAlmostEqual(height_m, height_m_orig, places=5)


class TestEllipsoidCurvature(unittest.TestCase):
    """Test suite for Ellipsoid curvature computations."""

    def setUp(self):
        """Set up test fixtures."""
        self.ellipsoid = ELPS_WGS84

    def test_prime_vertical_curvature_radius_equator(self):
        """Test prime vertical curvature radius at the equator."""
        nu = self.ellipsoid.prime_vertical_curvature_radius(0.0)
        # At equator, nu should equal semi-major axis
        self.assertAlmostEqual(nu, self.ellipsoid._a, places=1)

    def test_prime_vertical_curvature_radius_pole(self):
        """Test prime vertical curvature radius at the pole."""
        nu = self.ellipsoid.prime_vertical_curvature_radius(np.pi / 2)
        # At pole, nu > a
        self.assertGreater(nu, self.ellipsoid._a)


class TestEllipsoidLatitudes(unittest.TestCase):
    """Test suite for Ellipsoid auxiliary latitude computations."""

    def setUp(self):
        """Set up test fixtures."""
        self.ellipsoid = ELPS_WGS84

    def test_isometric_latitude(self):
        """Test isometric latitude computation."""
        phi = np.deg2rad(45.0)
        psi = self.ellipsoid.isometric_latitude(phi)
        self.assertIsInstance(psi, float)

    def test_inverse_isometric_latitude(self):
        """Test inverse isometric latitude computation."""
        phi_orig = np.deg2rad(45.0)
        psi = self.ellipsoid.isometric_latitude(phi_orig)
        phi = self.ellipsoid.inverse_isometric_latitude(psi)
        self.assertAlmostEqual(phi, phi_orig, places=10)

    def test_conformal_latitude(self):
        """Test conformal latitude computation."""
        phi = np.deg2rad(45.0)
        chi = self.ellipsoid.conformal_latitude(phi)
        self.assertIsInstance(chi, float)
        # Conformal latitude should be slightly less than geodetic latitude
        self.assertLess(chi, phi)

    def test_inverse_conformal_latitude(self):
        """Test inverse conformal latitude computation."""
        phi_orig = np.deg2rad(45.0)
        chi = self.ellipsoid.conformal_latitude(phi_orig)
        phi = self.ellipsoid.inverse_conformal_latitude(chi)
        self.assertAlmostEqual(phi, phi_orig, places=10)


if __name__ == '__main__':
    unittest.main()
