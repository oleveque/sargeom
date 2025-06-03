import unittest
from sargeom.coordinates import CartesianECEF, Cartographic, CartesianLocalENU, CartesianLocalNED

class TestCartesianECEF(unittest.TestCase):

    def setUp(self):
        # Create instances for testing
        self.ecef_coords = CartesianECEF(4198945, 174747, 4781887)
    
    def test_is_local(self):
        # Test if the ECEF coordinates are local
        self.assertFalse(self.ecef_coords.is_local())

    def test_logical_operations(self):
        # Test equality of two ECEF coordinates
        ecef_coords_2 = CartesianECEF(4198945, 174747, 4781887)
        self.assertTrue(self.ecef_coords == ecef_coords_2)
        
        # Test inequality of two ECEF coordinates
        ecef_coords_3 = CartesianECEF(4198945, 174747, 4781888)
        self.assertFalse(self.ecef_coords == ecef_coords_3)

    def test_to_cartographic(self):
        # Test conversion from ECEF to Cartographic and back to ECEF
        cartographic_coords = self.ecef_coords.to_cartographic()
        self.assertIsInstance(cartographic_coords, Cartographic)
        
        result = cartographic_coords.to_ecef()
        self.assertAlmostEqual(result.x, 4198945)
        self.assertAlmostEqual(result.y, 174747)
        self.assertAlmostEqual(result.z, 4781887)

    def test_to_ned(self):
        # Test conversion from ECEF to NED and back to ECEF
        origin = Cartographic.ONERA_CP()
        ned_coords = self.ecef_coords.to_ned(origin)
        self.assertIsInstance(ned_coords, CartesianLocalNED)
        
        result = ned_coords.to_ecef()
        self.assertAlmostEqual(result.x, 4198945)
        self.assertAlmostEqual(result.y, 174747)
        self.assertAlmostEqual(result.z, 4781887)

    def test_to_nedv(self):
        # Test conversion from ECEF vector to NED vector
        origin = Cartographic.ONERA_CP()
        ned_vector = self.ecef_coords.to_nedv(origin)
        self.assertIsInstance(ned_vector, CartesianLocalNED)
        self.assertAlmostEqual(ned_vector.x, -2630.34644233)
        self.assertAlmostEqual(ned_vector.y, 11171.93647109)
        self.assertAlmostEqual(ned_vector.z, -6366159.53121787)

    def test_to_enu(self):
        # Test conversion from ECEF to ENU and back to ECEF
        origin = Cartographic.ONERA_CP()
        enu_coords = self.ecef_coords.to_enu(origin)
        self.assertIsInstance(enu_coords, CartesianLocalENU)
        
        result = enu_coords.to_ecef()
        self.assertAlmostEqual(result.x, 4198945)
        self.assertAlmostEqual(result.y, 174747)
        self.assertAlmostEqual(result.z, 4781887)

    def test_to_enuv(self):
        # Test conversion from ECEF vector to ENU vector
        ecef_vector = CartesianECEF(1.0, 2.0, 3.0)
        origin = Cartographic.ONERA_CP()
        enu_vector = ecef_vector.to_enuv(origin)
        self.assertIsInstance(enu_vector, CartesianLocalENU)
        self.assertAlmostEqual(enu_vector.x, 1.959559604538143)
        self.assertAlmostEqual(enu_vector.y, 1.17015063)
        self.assertAlmostEqual(enu_vector.z, 2.96494075)

if __name__ == '__main__':
    unittest.main()
