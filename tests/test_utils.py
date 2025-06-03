import unittest
import numpy as np

from sargeom.coordinates.utils import negativePiToPi

class TestUtils(unittest.TestCase):
    
    def test_negativePiToPi_scalars(self):
        self.assertEqual(negativePiToPi(190), -170.0)
        self.assertEqual(negativePiToPi(-190), 170.0)
        self.assertAlmostEqual(negativePiToPi(4.0, degrees=False), 4.0 - 2*np.pi, places=7)
    
    def test_negativePiToPi_arrays(self):
        arr = [-190, 190, 360]
        result = negativePiToPi(arr)
        expected = np.array([ 170., -170.,   0.])
        np.testing.assert_array_equal(result, expected)
