import unittest
import numpy as np
import pandas as pd

from sargeom.coordinates.cartographic import Cartographic
from sargeom.coordinates.cartesian import CartesianECEF

class TestCartographic(unittest.TestCase):

    def setUp(self):
        # Create instances for testing
        self.cartographic_point = Cartographic(longitude=1.0, latitude=2.0, height=3.0)
        self.cartographic_collection = Cartographic(
            longitude=[1.0, 10.0],
            latitude=[2.0, 20.0],
            height=[3.0, 30.0]
        )

    def test_creation(self):
        with self.assertRaises(ValueError):
            Cartographic(longitude=[1,2], latitude=[3], height=[4,5])

        with self.assertRaises(ValueError):
            Cartographic(longitude=[[1,2],[3,4]], latitude=[10,20], height=[0,0])

        with self.assertRaises(ValueError):
            Cartographic.from_array(np.array([1.0, 2.0]))   # moins de 3 éléments
        
        with self.assertRaises(ValueError):
            Cartographic.from_array(np.array([[1,2,3,4]])) # 4 colonnes au lieu de 3

    def test_attribute_access(self):
        # Test attribute access for single point
        self.assertEqual(self.cartographic_point.longitude, 1.0)
        self.assertEqual(self.cartographic_point.latitude, 2.0)
        self.assertEqual(self.cartographic_point.height, 3.0)

    def test_creation_methods(self):
        # Test creation using predefined methods
        cp_point = Cartographic.ONERA_CP()
        self.assertEqual(cp_point.longitude, 2.230784)
        self.assertEqual(cp_point.latitude, 48.713028)
        self.assertEqual(cp_point.height, 0.0)
        
        sdp_point = Cartographic.ONERA_SDP()
        self.assertEqual(sdp_point.longitude, 5.117724)
        self.assertEqual(sdp_point.latitude, 43.619212)
        self.assertEqual(sdp_point.height, 0.0)
        
        zero_point = Cartographic.ZERO()
        self.assertEqual(zero_point.longitude, 0.0)
        self.assertEqual(zero_point.latitude, 0.0)
        self.assertEqual(zero_point.height, 0.0)

        # Test creation from array
        my_array = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        new_cartographic_collection = Cartographic.from_array(my_array)
        self.assertTrue(np.all(new_cartographic_collection.longitude == [1.0, 4.0, 7.0]))
        self.assertTrue(np.all(new_cartographic_collection.latitude == [2.0, 5.0, 8.0]))
        self.assertTrue(np.all(new_cartographic_collection.height == [3.0, 6.0, 9.0]))
        
        # Test appending new points
        new_instance = new_cartographic_collection.append(self.cartographic_point)
        self.assertTrue(np.all(new_instance.longitude == [1.0, 4.0, 7.0, 1.0]))
        self.assertTrue(np.all(new_instance.latitude == [2.0, 5.0, 8.0, 2.0]))
        self.assertTrue(np.all(new_instance.height == [3.0, 6.0, 9.0, 3.0]))

        with self.assertRaises(ValueError):
            Cartographic(1,2,3).append([42, 43, 44])
        
    def test_is_collection(self):
        # Test if the instance is a collection
        self.assertFalse(self.cartographic_point.is_collection())
        self.assertTrue(self.cartographic_collection.is_collection())

        pt = Cartographic(10.0, 20.0, 0.0)
        with self.assertRaises(ValueError):
            pt.bounding_box()

    def test_inner_operations(self):
        # Test string representation
        representation = repr(self.cartographic_point)
        self.assertIsInstance(representation, str)

        # Test length of single point and collection
        length_point = len(self.cartographic_point)
        self.assertEqual(length_point, 1)

        length_collection = len(self.cartographic_collection)
        self.assertEqual(length_collection, 2)

    def test_logical_operations(self):
        # Test equality of two collections
        my_array = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0]
        ])
        collection = Cartographic.from_array(my_array)
        self.assertTrue(self.cartographic_collection == collection)

    def test_slice_operations(self):
        # Test slicing
        self.assertTrue(np.all(self.cartographic_collection[0:2].to_array() == self.cartographic_collection.to_array()))

        # Test single index slicing
        sliced_collection = self.cartographic_collection[0]
        self.assertIsInstance(sliced_collection, Cartographic)
        self.assertEqual(sliced_collection.longitude, 1.0)
        self.assertEqual(sliced_collection.latitude, 2.0)
        self.assertEqual(sliced_collection.height, 3.0)
        
        # Test negative index slicing
        sliced_collection = self.cartographic_collection[-1]
        self.assertEqual(sliced_collection.longitude, 10.0)
        self.assertEqual(sliced_collection.latitude, 20.0)
        self.assertEqual(sliced_collection.height, 30.0)

    def test_export_functions(self):
        # Test conversion to numpy array
        my_array = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0]
        ])
        collection = self.cartographic_collection.to_array()
        self.assertIsInstance(collection, np.ndarray)
        self.assertTrue(np.all(my_array == collection))

        # Test conversion to pandas DataFrame
        my_dataframe = pd.DataFrame(my_array, columns=['longitude', 'latitude', 'height'])
        collection = self.cartographic_collection.to_pandas()
        self.assertIsInstance(collection, pd.DataFrame)
        self.assertTrue(np.all(my_dataframe == collection))

    def test_to_ecef(self):
        # Test conversion to ECEF coordinates
        ecef_coords = self.cartographic_point.to_ecef()
        self.assertIsInstance(ecef_coords, CartesianECEF)
        self.assertAlmostEqual(ecef_coords.x, 6373309.76229944, places=7)
        self.assertAlmostEqual(ecef_coords.y, 111246.53570858, places=7)
        self.assertAlmostEqual(ecef_coords.z, 221104.65000961, places=7)

    def test_bounding_box(self):
        # Test bounding box calculation
        east, west, north, south = self.cartographic_collection.bounding_box()
        self.assertEqual(east, 10.0)
        self.assertEqual(west, 1.0)
        self.assertEqual(north, 20.0)
        self.assertEqual(south, 2.0)

if __name__ == '__main__':
    unittest.main()