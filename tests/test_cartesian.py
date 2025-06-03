import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd

from sargeom.coordinates.cartesian import Cartesian3, CartesianECEF, CartesianLocalENU

class TestCartesian3(unittest.TestCase):

    def setUp(self):
        # Create instances for testing
        self.cartesian_point = Cartesian3(x=1.0, y=2.0, z=3.0)
        self.cartesian_collection = Cartesian3(
            x=[1.0, 10.0],
            y=[2.0, 20.0],
            z=[3.0, 30.0]
        )
    
    def test_creation(self):
        with self.assertRaises(ValueError):
            Cartesian3(x=[1,2], y=[3], z=[4,5])

        with self.assertRaises(ValueError):
            Cartesian3(x=[[1,2]], y=2.0, z=3.0)

        with self.assertRaises(ValueError):
            Cartesian3.from_array(np.array([1.0,2.0]))

        with self.assertRaises(ValueError):
            Cartesian3.from_array(np.ones((2,4)))

    def test_conversion_methods(self):
        enu = CartesianLocalENU(1,2,3, origin=None)
        with self.assertRaises(ValueError):
            enu.to_ecef()

        ecef = CartesianECEF(1,2,3)
        with self.assertRaises(ValueError):
            ecef.to_ned(origin="not a Cartographic")
        with self.assertRaises(ValueError):
            ecef.to_enuv(origin=12345)

    def test_attribute_access(self):
        # Test single point attribute access
        self.assertEqual(self.cartesian_point.x, 1.0)
        self.assertEqual(self.cartesian_point.y, 2.0)
        self.assertEqual(self.cartesian_point.z, 3.0)

    def test_creation_methods(self):
        # Test UNIT_X creation method
        unit_x = Cartesian3.UNIT_X()
        self.assertEqual(unit_x.x, 1.0)
        self.assertEqual(unit_x.y, 0.0)
        self.assertEqual(unit_x.z, 0.0)
        
        # Test UNIT_Y creation method
        unit_y = Cartesian3.UNIT_Y()
        self.assertEqual(unit_y.x, 0.0)
        self.assertEqual(unit_y.y, 1.0)
        self.assertEqual(unit_y.z, 0.0)
        
        # Test UNIT_Z creation method
        unit_z = Cartesian3.UNIT_Z()
        self.assertEqual(unit_z.x, 0.0)
        self.assertEqual(unit_z.y, 0.0)
        self.assertEqual(unit_z.z, 1.0)

        # Test ONE creation method
        one_vector = Cartesian3.ONE()
        self.assertEqual(one_vector.x, 1.0)
        self.assertEqual(one_vector.y, 1.0)
        self.assertEqual(one_vector.z, 1.0)
        
        # Test ZERO creation method
        zero_vector = Cartesian3.ZERO()
        self.assertEqual(zero_vector.x, 0.0)
        self.assertEqual(zero_vector.y, 0.0)
        self.assertEqual(zero_vector.z, 0.0)

        # Test from_array method
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        cart = Cartesian3.from_array(arr)
        npt.assert_array_equal(cart.x, [1.0, 4.0, 7.0])
        npt.assert_array_equal(cart.y, [2.0, 5.0, 8.0])
        npt.assert_array_equal(cart.z, [3.0, 6.0, 9.0])
        
        # Test append method
        new_instance = cart.append(self.cartesian_point)
        npt.assert_array_equal(new_instance.x, [1.0, 4.0, 7.0, 1.0])
        npt.assert_array_equal(new_instance.y, [2.0, 5.0, 8.0, 2.0])
        npt.assert_array_equal(new_instance.z, [3.0, 6.0, 9.0, 3.0])

    def test_is_collection(self):
        # Test is_collection method
        self.assertFalse(self.cartesian_point.is_collection())
        self.assertTrue(self.cartesian_collection.is_collection())
        
    def test_is_local(self):
        # Test is_local method
        self.assertFalse(self.cartesian_point.is_local())
        self.assertFalse(self.cartesian_collection.is_local())

    def test_inner_operations(self):
        # Test __repr__ method
        representation = repr(self.cartesian_point)
        self.assertIsInstance(representation, str)

        # Test __len__ method
        length_point = len(self.cartesian_point)
        self.assertEqual(length_point, 1)

        length_collection = len(self.cartesian_collection)
        self.assertEqual(length_collection, 2)

    def test_math_addition_operations(self):
        # Test addition with another Cartesian3 instance
        result1 = self.cartesian_point + Cartesian3.ONE()
        self.assertEqual(result1.x, 2.0)
        self.assertEqual(result1.y, 3.0)
        self.assertEqual(result1.z, 4.0)
        
        # Test addition with a scalar
        result2 = self.cartesian_point + 5.0
        self.assertEqual(result2.x, 6.0)
        self.assertEqual(result2.y, 7.0)
        self.assertEqual(result2.z, 8.0)
        
        # Test reverse addition with a scalar
        result3 = 5.0 + self.cartesian_point
        self.assertEqual(result3.x, 6.0)
        self.assertEqual(result3.y, 7.0)
        self.assertEqual(result3.z, 8.0)
        
    def test_math_substraction_operations(self):
        # Test negation
        neg_cartesian_point = -self.cartesian_point
        self.assertEqual(neg_cartesian_point.x, -1.0)
        self.assertEqual(neg_cartesian_point.y, -2.0)
        self.assertEqual(neg_cartesian_point.z, -3.0)
        
        # Test subtraction with another Cartesian3 instance
        result1 = self.cartesian_point - Cartesian3.ONE()
        self.assertEqual(result1.x, 0.0)
        self.assertEqual(result1.y, 1.0)
        self.assertEqual(result1.z, 2.0)
        
        # Test subtraction with a scalar
        result2 = self.cartesian_point - 5.0
        self.assertEqual(result2.x, -4.0)
        self.assertEqual(result2.y, -3.0)
        self.assertEqual(result2.z, -2.0)
    
    def test_math_division(self):
        # Test division with another Cartesian3 instance
        result = self.cartesian_point / self.cartesian_point
        self.assertEqual(result.x, 1.0)
        self.assertEqual(result.y, 1.0)
        self.assertEqual(result.z, 1.0)
    
    def test_math_multiplication(self):
        # Test multiplication with another Cartesian3 instance
        result = self.cartesian_point * self.cartesian_point
        self.assertEqual(result.x, 1.0)
        self.assertEqual(result.y, 4.0)
        self.assertEqual(result.z, 9.0)
        
    def test_math_cross(self):
        # Test cross product with another Cartesian3 instance
        result = self.cartesian_point.cross(Cartesian3.ONE())
        self.assertEqual(result.x, -1.0)
        self.assertEqual(result.y, 2.0)
        self.assertEqual(result.z, -1.0)

    def test_normalization(self):
        # Test normalization
        normalized_result = self.cartesian_point.normalize()
        self.assertEqual(np.linalg.norm(normalized_result.to_array()), 1.0)

    def test_rejection(self):
        # Test rejection from another Cartesian3 instance
        A = self.cartesian_collection
        B = self.cartesian_point
        rejection = A.reject_from(B)
        expected_rejection = A - A.proj_onto(B)
        self.assertTrue(np.allclose(rejection.to_array(), expected_rejection.to_array()))

    def test_projection(self):
        # Test projection onto another Cartesian3 instance
        A = self.cartesian_collection
        B = self.cartesian_point
        projection = A.proj_onto(B)
        expected_projection = A.dot(B)[:, None] * B / B.magnitude()
        self.assertTrue(np.allclose(projection.to_array(), expected_projection.to_array()))

    def test_logical_operations(self):
        # Test equality comparison
        my_array = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0]
        ])
        collection = Cartesian3.from_array(my_array)
        self.assertTrue(self.cartesian_collection == collection)

    def test_slice_operations(self):
        # Test slicing
        self.assertTrue(np.all(self.cartesian_collection[0:2].to_array() == self.cartesian_collection.to_array()))
        
        # Test single index slicing
        slice_result = self.cartesian_collection[0]
        self.assertEqual(slice_result.x, 1.0)
        self.assertEqual(slice_result.y, 2.0)
        self.assertEqual(slice_result.z, 3.0)
        
        # Test negative index slicing
        slice_result = self.cartesian_collection[-1]
        self.assertEqual(slice_result.x, 10.0)
        self.assertEqual(slice_result.y, 20.0)
        self.assertEqual(slice_result.z, 30.0)

    def test_center_of_mass(self):
        # Test center of mass calculation
        centroid = self.cartesian_collection.centroid()
        self.assertEqual(centroid.x, np.mean(self.cartesian_collection.x))
        self.assertEqual(centroid.y, np.mean(self.cartesian_collection.y))
        self.assertEqual(centroid.z, np.mean(self.cartesian_collection.z))

    def test_static_methods(self):
        # Test dot product
        A = self.cartesian_point
        B = Cartesian3(4.0, 5.0, 6.0)
        
        dot_product = A.dot(B)
        self.assertEqual(dot_product, 32.0)
        
        # Test distance calculation
        distance = Cartesian3.distance(A, B)
        self.assertAlmostEqual(distance, np.sqrt(27), places=7)
        
        # Test midpoint calculation
        middle = Cartesian3.middle(A, B)
        self.assertEqual(middle.x, 2.5)
        self.assertEqual(middle.y, 3.5)
        self.assertEqual(middle.z, 4.5)
        
        # Test angle calculation
        B = Cartesian3(-1.0, 2.0, -1.0)
        angle = Cartesian3.angle_btw(A, B)
        self.assertEqual(angle, 90.0)

    def test_interpolation(self):
        # Test interpolation
        time = np.array([0.0, 2.0])
        new_time = np.array([0.5])
        interpolated_coords = self.cartesian_collection.interp(time, new_time)
        self.assertEqual(len(interpolated_coords), len(new_time))

    def test_export_functions(self):
        # Test to_array method
        my_array = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0]
        ])
        collection = self.cartesian_collection.to_array()
        self.assertIsInstance(collection, np.ndarray)
        self.assertTrue(np.all(my_array == collection))

        # Test to_pandas method
        my_dataframe = pd.DataFrame(my_array, columns=['x', 'y', 'z'])
        collection = self.cartesian_collection.to_pandas()
        self.assertIsInstance(collection, pd.DataFrame)
        self.assertTrue(np.all(my_dataframe == collection))

    def test_magnitude(self):
        # Test magnitude calculation
        magnitude = self.cartesian_point.magnitude()
        self.assertAlmostEqual(magnitude, np.sqrt(14), places=7)

    def test_normalize(self):
        # Test normalization
        normalized = self.cartesian_point.normalize()
        self.assertAlmostEqual(np.linalg.norm(normalized.to_array()), 1.0, places=7)

    def test_cross_product(self):
        # Test cross product
        cross_product = self.cartesian_point.cross(Cartesian3(4.0, 5.0, 6.0))
        self.assertTrue(np.allclose(cross_product.to_array(), [-3.0, 6.0, -3.0]))

    def test_dot_product(self):
        # Test dot product
        dot_product = self.cartesian_point.dot(Cartesian3(4.0, 5.0, 6.0))
        self.assertEqual(dot_product, 32.0)

    def test_centroid(self):
        # Test centroid calculation
        centroid = self.cartesian_collection.centroid()
        self.assertTrue(np.allclose(centroid.to_array(), [5.5, 11.0, 16.5]))

    def test_distance(self):
        # Test distance calculation
        distance = Cartesian3.distance(self.cartesian_point, Cartesian3(4.0, 5.0, 6.0))
        self.assertAlmostEqual(distance, np.sqrt(27), places=7)

    def test_middle(self):
        # Test midpoint calculation
        middle = Cartesian3.middle(self.cartesian_point, Cartesian3(4.0, 5.0, 6.0))
        self.assertTrue(np.allclose(middle.to_array(), [2.5, 3.5, 4.5]))

    def test_angle_btw(self):
        # Test angle calculation
        angle = Cartesian3.angle_btw(self.cartesian_point, Cartesian3(-1.0, 2.0, -1.0))
        self.assertAlmostEqual(angle, 90.0, places=7)

if __name__ == '__main__':
    unittest.main()