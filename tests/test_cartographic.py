import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from sargeom.coordinates.cartographic import Cartographic
from sargeom.coordinates.cartesian import CartesianECEF


class TestCartographic(unittest.TestCase):
    """Test suite for Cartographic coordinate conversions and utilities."""
    
    def setUp(self):
        # Instances for single point and collection
        self.cart_point = Cartographic(longitude=1.0, latitude=2.0, height=3.0)
        self.cart_collection = Cartographic(
            longitude=[1.0, 10.0],
            latitude=[2.0, 20.0],
            height=[3.0, 30.0]
        )

    def test_invalid_construction(self):
        """Construction should fail for mismatched list lengths or invalid shapes."""
        with self.assertRaises(ValueError):
            Cartographic(longitude=[1, 2], latitude=[3], height=[4, 5])

        with self.assertRaises(ValueError):
            Cartographic(longitude=[[1, 2], [3, 4]], latitude=[10, 20], height=[0, 0])

    def test_creation_and_accessors(self):
        """Verify properties of a single Cartographic instance and ZERO method."""
        self.assertEqual(self.cart_point.longitude, 1.0)
        self.assertEqual(self.cart_point.latitude, 2.0)
        self.assertEqual(self.cart_point.height, 3.0)

        # Test static ZERO method
        zero = Cartographic.ZERO()
        self.assertEqual(zero.longitude, 0.0)
        self.assertEqual(zero.latitude, 0.0)
        self.assertEqual(zero.height, 0.0)

        # Test static ONERA_CP
        cp_point = Cartographic.ONERA_CP()
        self.assertEqual(cp_point.longitude, 2.230784)
        self.assertEqual(cp_point.latitude, 48.713028)
        self.assertEqual(cp_point.height, 0.0)
        
        # Test static ONERA_SDP
        sdp_point = Cartographic.ONERA_SDP()
        self.assertEqual(sdp_point.longitude, 5.117724)
        self.assertEqual(sdp_point.latitude, 43.619212)
        self.assertEqual(sdp_point.height, 0.0)

    def test_from_array(self):
        """Construct a collection from a NumPy array and verify values."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],])
        collection = Cartographic.from_array(arr)
        assert_array_equal(collection.longitude, [1.0, 4.0, 7.0])
        assert_array_equal(collection.latitude, [2.0, 5.0, 8.0])
        assert_array_equal(collection.height, [3.0, 6.0, 9.0])

        # Test with invalid array shape
        with self.assertRaises(ValueError):
            Cartographic.from_array(np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            Cartographic.from_array(np.array([[1,2,3,4]]))

    def test_append(self):
        """Append a point to a collection and verify new lengths and contents."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],])
        collection = Cartographic.from_array(arr)
        new_instance = collection.append(self.cart_point)
        assert_array_equal(new_instance.longitude, [1.0, 4.0, 7.0, 1.0])
        assert_array_equal(new_instance.latitude, [2.0, 5.0, 8.0, 2.0])
        assert_array_equal(new_instance.height, [3.0, 6.0, 9.0, 3.0])

        # Appending invalid type should raise
        with self.assertRaises(ValueError):
            Cartographic(1,2,3).append([42, 43, 44])
        
    def test_is_collection(self):
        """Check is_collection property for point vs. collection."""
        self.assertFalse(self.cart_point.is_collection())
        self.assertTrue(self.cart_collection.is_collection())

    def test_inner_operations(self):
        # Test string representation
        representation = repr(self.cart_point)
        self.assertIsInstance(representation, str)

        # Test length of single point and collection
        length_point = len(self.cart_point)
        self.assertEqual(length_point, 1)

        length_collection = len(self.cart_collection)
        self.assertEqual(length_collection, 2)

    def test_logical_operations(self):
        # Test equality of two collections
        arr = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        collection = Cartographic.from_array(arr)
        self.assertTrue(self.cart_collection == collection)

    def test_slice_operations(self):
        """Test slicing returns correct sub-collection or single instance."""
        # Range slicing returns Cartographic with arrays
        slice_range = self.cart_collection[0:2]
        assert_array_equal(slice_range.to_array(), self.cart_collection.to_array())

        # Single index returns a point
        slice0 = self.cart_collection[0]
        self.assertIsInstance(slice0, Cartographic)
        self.assertEqual(slice0.longitude, 1.0)
        self.assertEqual(slice0.latitude, 2.0)
        self.assertEqual(slice0.height, 3.0)

        # Negative index
        slice_neg = self.cart_collection[-1]
        self.assertEqual(slice_neg.longitude, 10.0)
        self.assertEqual(slice_neg.latitude, 20.0)
        self.assertEqual(slice_neg.height, 30.0)

    def test_export_functions(self):
        """Verify exporting to NumPy array and pandas DataFrame."""
        expected_array = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        arr = self.cart_collection.to_array()
        self.assertIsInstance(arr, np.ndarray)
        assert_array_equal(arr, expected_array)

        expected_df = pd.DataFrame(expected_array, columns=['longitude', 'latitude', 'height'])
        df = self.cart_collection.to_pandas()
        self.assertIsInstance(df, pd.DataFrame)
        assert_frame_equal(df, expected_df)

    def test_to_ecef(self):
        """Test conversion from Cartographic to Cartesian ECEF coordinates."""
        ecef_coords = self.cart_point.to_ecef()
        self.assertIsInstance(ecef_coords, CartesianECEF)
        self.assertAlmostEqual(ecef_coords.x, 6373309.76229944, places=6)
        self.assertAlmostEqual(ecef_coords.y, 111246.53570858, places=6)
        self.assertAlmostEqual(ecef_coords.z, 221104.65000961, places=6)

    def test_bounding_box(self):
        """Calculate bounding box for a collection of points."""
        east, west, north, south = self.cart_collection.bounding_box()
        self.assertEqual(east, 10.0)
        self.assertEqual(west, 1.0)
        self.assertEqual(north, 20.0)
        self.assertEqual(south, 2.0)

        # Test bounding box for a single point
        pt = Cartographic(10.0, 20.0, 0.0)
        with self.assertRaises(ValueError):
            pt.bounding_box()


if __name__ == '__main__':
    unittest.main()