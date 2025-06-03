import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from sargeom.coordinates.utils import negativePiToPi

class TestUtils(unittest.TestCase):
    """Test suite for utility functions in sargeom.coordinates.utils."""

    def test_negativePiToPi_scalar_degrees(self):
        """Wrap scalar angles in degrees to range [-180, 180]."""
        self.assertEqual(negativePiToPi(190), -170.0)
        self.assertEqual(negativePiToPi(-190), 170.0)

    def test_negativePiToPi_scalar_radians(self):
        """Wrap scalar angles in radians to range [-pi, pi]."""
        assert_allclose(negativePiToPi(4.0, degrees=False), 4.0 - 2*np.pi, rtol=1e-7)
    
    def test_negativePiToPi_array_degrees(self):
        list_input = [-540, 540]
        expected_list = np.array([ -180.0, 180.0 ])
        assert_array_equal(negativePiToPi(list_input), expected_list)

        tuple_input = (-3*np.pi, 3*np.pi)
        expected_tuple = np.array([ -np.pi,  np.pi ])
        assert_array_equal(negativePiToPi(tuple_input, degrees=False), expected_tuple)


if __name__ == '__main__':
    unittest.main()
