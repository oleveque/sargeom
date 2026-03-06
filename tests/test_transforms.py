"""
Test suite for the LambertConicConformal projection class.

This module contains unit tests for the LambertConicConformal projection,
covering forward and inverse transformations.
"""

import unittest
import numpy as np

from sargeom.coordinates.transforms import LambertConicConformal
from sargeom.coordinates.ellipsoids import ELPS_WGS84, ELPS_CLARKE_1880


class TestLambertConicConformalCreation(unittest.TestCase):
    """Test suite for LambertConicConformal creation."""

    def test_creation_basic(self):
        """Test basic Lambert projection creation."""
        lcc = LambertConicConformal(
            ELPS_WGS84,
            lon_origin_rad=np.deg2rad(3.0),
            lat_origin_rad=np.deg2rad(46.5)
        )
        self.assertIsNotNone(lcc)
        self.assertEqual(lcc.scale, 1.0)
        self.assertEqual(lcc.x_offset_m, 0.0)

    def test_creation_with_offsets(self):
        """Test Lambert projection creation with offsets."""
        lcc = LambertConicConformal(
            ELPS_WGS84,
            lon_origin_rad=np.deg2rad(3.0),
            lat_origin_rad=np.deg2rad(46.5),
            scale=0.99987742,
            x_offset_m=600000.0,
            y_offset_m=200000.0
        )
        self.assertAlmostEqual(lcc.scale, 0.99987742)
        self.assertEqual(lcc.x_offset_m, 600000.0)


class TestLambertConicConformalTransformations(unittest.TestCase):
    """Test suite for LambertConicConformal transformations."""

    def setUp(self):
        """Set up test fixtures."""
        self.lcc = LambertConicConformal(
            ELPS_WGS84,
            lon_origin_rad=np.deg2rad(3.0),
            lat_origin_rad=np.deg2rad(46.5)
        )

    def test_forward_single_point(self):
        """Test forward transformation for a single point."""
        lon_rad = np.deg2rad(2.3522)
        lat_rad = np.deg2rad(48.8566)
        
        x, y = self.lcc.forward(lon_rad, lat_rad)
        
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)

    def test_forward_array(self):
        """Test forward transformation for multiple points."""
        lon_rad = np.deg2rad([2.3522, 3.0, 4.0])
        lat_rad = np.deg2rad([48.8566, 46.5, 45.0])
        
        x, y = self.lcc.forward(lon_rad, lat_rad)
        
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)

    def test_inverse_single_point(self):
        """Test inverse transformation for a single point."""
        # First do forward transformation
        lon_orig = np.deg2rad(2.3522)
        lat_orig = np.deg2rad(48.8566)
        x, y = self.lcc.forward(lon_orig, lat_orig)
        
        # Then inverse transformation
        lon_rad, lat_rad = self.lcc.inverse(x, y)
        
        self.assertAlmostEqual(lon_rad, lon_orig, places=8)
        self.assertAlmostEqual(lat_rad, lat_orig, places=8)

    def test_roundtrip_transformation(self):
        """Test round-trip forward -> inverse transformation."""
        lon_orig = np.deg2rad([2.0, 3.0, 4.0, 5.0])
        lat_orig = np.deg2rad([45.0, 46.0, 47.0, 48.0])
        
        # Forward
        x, y = self.lcc.forward(lon_orig, lat_orig)
        
        # Inverse
        lon_rad, lat_rad = self.lcc.inverse(x, y)
        
        np.testing.assert_array_almost_equal(lon_rad, lon_orig, decimal=8)
        np.testing.assert_array_almost_equal(lat_rad, lat_orig, decimal=8)

    def test_origin_maps_to_zero(self):
        """Test that the origin maps to (0, y_offset)."""
        x, y = self.lcc.forward(self.lcc.lon_origin_rad, self.lcc.lat_origin_rad)
        
        self.assertAlmostEqual(x, 0.0, places=5)


class TestLambertConicConformalWithClarke(unittest.TestCase):
    """Test suite for LambertConicConformal with Clarke 1880 ellipsoid."""

    def setUp(self):
        """Set up test fixtures for NTF projection."""
        # Approximate NTF Lambert I parameters
        self.lcc_ntf = LambertConicConformal(
            ELPS_CLARKE_1880,
            lon_origin_rad=np.deg2rad(2.337229166667),  # Paris meridian
            lat_origin_rad=np.deg2rad(49.5)  # Lambert I center
        )

    def test_forward_ntf(self):
        """Test forward transformation with Clarke 1880 ellipsoid."""
        lon_rad = np.deg2rad(2.3522)
        lat_rad = np.deg2rad(48.8566)
        
        x, y = self.lcc_ntf.forward(lon_rad, lat_rad)
        
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)

    def test_roundtrip_ntf(self):
        """Test round-trip transformation with Clarke 1880 ellipsoid."""
        lon_orig = np.deg2rad(2.3522)
        lat_orig = np.deg2rad(48.8566)
        
        x, y = self.lcc_ntf.forward(lon_orig, lat_orig)
        lon_rad, lat_rad = self.lcc_ntf.inverse(x, y)
        
        self.assertAlmostEqual(lon_rad, lon_orig, places=8)
        self.assertAlmostEqual(lat_rad, lat_orig, places=8)


if __name__ == '__main__':
    unittest.main()
