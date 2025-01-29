import unittest
import numpy as np
from sargeom.coordinates import CartesianLocalENU, Cartographic, CartesianECEF, CartesianLocalNED

class TestCartesianLocalENU(unittest.TestCase):

    def setUp(self):
        # Create instances for testing
        origin = Cartographic.ONERA_CP()
        self.enu_coords = CartesianLocalENU(10.0, 20.0, 30.0, origin)

    def test_is_local(self):
        # Test if the coordinate system is local
        self.assertTrue(self.enu_coords.is_local())

    def test_logical_operations(self):
        # Test equality of two ENU coordinates
        enu_coords_2 = CartesianLocalENU(10.0, 20.0, 30.0, Cartographic.ONERA_CP())
        self.assertTrue(self.enu_coords == enu_coords_2)
        
        # Test inequality of two ENU coordinates
        enu_coords_3 = CartesianLocalENU(10.0, 20.0, 31.0, Cartographic.ONERA_CP())
        self.assertFalse(self.enu_coords == enu_coords_3)
        
        # Test inequality of two ENU coordinates
        enu_coords_4 = CartesianLocalENU(10.0, 20.0, 30.0, Cartographic.ONERA_SDP())
        self.assertFalse(self.enu_coords == enu_coords_4)

    def test_rotation(self):
        # Test the rotation matrix
        rotation_matrix = self.enu_coords.rotation.as_matrix()
        expected_results = np.array([
            [-0.03892469,  0.99924215,  0.        ],
            [-0.75084472, -0.02924856,  0.65983083],
            [ 0.65933077,  0.02568371,  0.75141419]
        ])
        self.assertIsInstance(rotation_matrix, np.ndarray)
        np.testing.assert_array_almost_equal(rotation_matrix, expected_results)

    def test_to_ecef(self):
        # Test conversion to ECEF coordinates
        ecef_coords = self.enu_coords.to_ecef()
        self.assertFalse(ecef_coords.is_local())
        self.assertIsInstance(ecef_coords, CartesianECEF)
        self.assertAlmostEqual(float(ecef_coords.x), 4213276.57752126)
        self.assertAlmostEqual(float(ecef_coords.y), 164134.87330518753)
        self.assertAlmostEqual(float(ecef_coords.z), 4769597.260556103)

    def test_to_ned(self):
        # Test conversion to NED coordinates
        ned_coords = self.enu_coords.to_ned()
        self.assertIsInstance(ned_coords, CartesianLocalNED)
        self.assertEqual(ned_coords.x, 20.0)
        self.assertEqual(ned_coords.y, 10.0)
        self.assertEqual(ned_coords.z, -30.0)

    def test_to_aer(self):
        # Test conversion to AER coordinates
        azimuth, elevation, slant_range = self.enu_coords.to_aer()
        self.assertAlmostEqual(azimuth, 26.56505117707799)
        self.assertAlmostEqual(elevation, 53.30077479951012)
        self.assertAlmostEqual(slant_range, 37.416573867739416)

    def test_properties(self):
        # Test properties of the CartesianLocalENU instance
        self.assertEqual(self.enu_coords.x, 10.0)
        self.assertEqual(self.enu_coords.y, 20.0)
        self.assertEqual(self.enu_coords.z, 30.0)
        self.assertEqual(self.enu_coords.local_origin.longitude, 2.230784)
        self.assertEqual(self.enu_coords.local_origin.latitude, 48.713028)
        self.assertEqual(self.enu_coords.local_origin.height, 0.0)

if __name__ == '__main__':
    unittest.main()