import unittest
import numpy as np
import pandas as pd
from sargeom.coordinates import Cartesian3

class TestCartesian3(unittest.TestCase):

    def setUp(self):
        # Create instances for testing
        self.cartesian_point = Cartesian3(x=1.0, y=2.0, z=3.0)
        self.cartesian_collection = Cartesian3(
            x=[1.0, 10.0],
            y=[2.0, 20.0],
            z=[3.0, 30.0]
        )

    def test_attribute_access(self):
        # Test single point attribute access
        self.assertEqual(self.cartesian_point.x, 1.0)
        self.assertEqual(self.cartesian_point.y, 2.0)
        self.assertEqual(self.cartesian_point.z, 3.0)

        # Test collection attribute access
        x, y, z = self.cartesian_collection[1]
        self.assertEqual(x, 10.0)
        self.assertEqual(y, 20.0)
        self.assertEqual(z, 30.0)

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
        my_array = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        new_cartesian_collection = Cartesian3.from_array(my_array)
        self.assertTrue(np.all(new_cartesian_collection.x == [1.0, 4.0, 7.0]))
        self.assertTrue(np.all(new_cartesian_collection.y == [2.0, 5.0, 8.0]))
        self.assertTrue(np.all(new_cartesian_collection.z == [3.0, 6.0, 9.0]))
        
        # Test append method
        new_instance = new_cartesian_collection.append(self.cartesian_point)
        self.assertTrue(np.all(new_instance.x == [1.0, 4.0, 7.0, 1.0]))
        self.assertTrue(np.all(new_instance.y == [2.0, 5.0, 8.0, 2.0]))
        self.assertTrue(np.all(new_instance.z == [3.0, 6.0, 9.0, 3.0]))
        
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

    def test_logical_operations(self):
        # Test equality comparison
        my_array = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0]
        ])
        collection = Cartesian3.from_array(my_array)
        self.assertTrue(self.cartesian_collection == collection)

    def test_static_methods(self):
        # Test dot product
        A = self.cartesian_point
        B = Cartesian3(4.0, 5.0, 6.0)
        
        dot_product = Cartesian3.dot(A, B)
        self.assertEqual(dot_product, 32.0)
        
        # Test distance calculation
        distance = Cartesian3.distance(A, B)
        self.assertAlmostEqual(distance, np.sqrt(27))
        
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

if __name__ == '__main__':
    unittest.main()