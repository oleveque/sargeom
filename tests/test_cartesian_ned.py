import unittest
import numpy as np
from sargeom.coordinates import CartesianLocalENU, Cartographic, CartesianECEF, CartesianLocalNED

class TestCartesianLocalNED(unittest.TestCase):

    def setUp(self):
        # Create instances for testing
        origin = Cartographic.ONERA_CP()
        self.ned_coords = CartesianLocalNED(20.0, 10.0, -30.0, origin)

    def test_is_local(self):
        # Test if the coordinates are local
        self.assertTrue(self.ned_coords.is_local())

    def test_logical_operations(self):
        # Test equality of two NED coordinates
        ned_coords_2 = CartesianLocalNED(20.0, 10.0, -30.0, Cartographic.ONERA_CP())
        self.assertTrue(self.ned_coords == ned_coords_2)
        
        # Test inequality of two NED coordinates
        ned_coords_3 = CartesianLocalNED(20.0, 10.0, -31.0, Cartographic.ONERA_CP())
        self.assertFalse(self.ned_coords == ned_coords_3)
        
        # Test inequality of two NED coordinates
        ned_coords_4 = CartesianLocalNED(20.0, 10.0, -30.0, Cartographic.ONERA_SDP())
        self.assertFalse(self.ned_coords == ned_coords_4)

    def test_rotation(self):
        # Test the rotation matrix
        rotation_matrix = self.ned_coords.rotation.as_matrix()
        expected_results = np.array([
            [-7.50844725e-01, -2.92485640e-02,  6.59830827e-01],
            [-3.89246897e-02,  9.99242147e-01,  1.73472348e-18],
            [-6.59330773e-01, -2.56837102e-02, -7.51414186e-01]
        ])
        self.assertIsInstance(rotation_matrix, np.ndarray)
        np.testing.assert_array_almost_equal(rotation_matrix, expected_results)

    def test_to_ecef(self):
        # Test conversion to ECEF coordinates
        ecef_coords = self.ned_coords.to_ecef()
        self.assertFalse(ecef_coords.is_local())
        self.assertIsInstance(ecef_coords, CartesianECEF)
        self.assertAlmostEqual(float(ecef_coords.x), 4213276.57752126)
        self.assertAlmostEqual(float(ecef_coords.y), 164134.87330518753)
        self.assertAlmostEqual(float(ecef_coords.z), 4769597.260556103)

    def test_to_enu(self):
        # Test conversion to ENU coordinates
        enu_coords = self.ned_coords.to_enu()
        self.assertIsInstance(enu_coords, CartesianLocalENU)
        self.assertEqual(enu_coords.x, 10.0)
        self.assertEqual(enu_coords.y, 20.0)
        self.assertEqual(enu_coords.z, 30.0)

    def test_to_aer(self):
        # Test conversion to AER coordinates
        azimuth, elevation, slant_range = self.ned_coords.to_aer()
        self.assertAlmostEqual(azimuth, 26.56505117707799)
        self.assertAlmostEqual(elevation, -53.30077479951012)
        self.assertAlmostEqual(slant_range, 37.416573867739416)

    def test_append(self):
        # Test appending coordinates
        new_coords = CartesianLocalNED(30.0, 20.0, -40.0, self.ned_coords.local_origin)
        appended_coords = self.ned_coords.append(new_coords)
        self.assertEqual(appended_coords.shape[0], 2)
        self.assertEqual(appended_coords.x[1], 30.0)
        self.assertEqual(appended_coords.y[1], 20.0)
        self.assertEqual(appended_coords.z[1], -40.0)

if __name__ == '__main__':
    unittest.main()