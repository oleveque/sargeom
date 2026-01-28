"""
Test suite for the Trajectory class.

This module contains unit tests for the Trajectory class, covering
creation, manipulation, conversion, and I/O operations.
"""

import tempfile
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.spatial.transform import Rotation

from sargeom import Trajectory
from sargeom.trajectory import TRAJ_DTYPE
from sargeom.coordinates import Cartographic, CartesianECEF, CartesianLocalENU


class TestTrajectoryCreation(unittest.TestCase):
    """Test suite for Trajectory creation and initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.timestamps = np.array([0, 1, 2, 3])
        self.positions = Cartographic(
            longitude=[3.8777, 4.8391, 5.4524, 6.2345],
            latitude=[43.6135, 43.9422, 43.5309, 43.7891],
            height=[300.0, 400.0, 500.0, 600.0]
        )
        self.orientations = Rotation.from_euler(
            "ZYX",
            [[0, 0, 0], [90, 0, 0], [180, 0, 0], [270, 0, 0]],
            degrees=True
        )

    def test_creation_with_cartographic(self):
        """Test trajectory creation with Cartographic positions."""
        traj = Trajectory(self.timestamps, self.positions)
        self.assertEqual(len(traj), 4)
        self.assertIsInstance(traj.positions, CartesianECEF)

    def test_creation_with_ecef(self):
        """Test trajectory creation with CartesianECEF positions."""
        ecef_positions = self.positions.to_ecef()
        traj = Trajectory(self.timestamps, ecef_positions)
        self.assertEqual(len(traj), 4)

    def test_creation_with_orientations(self):
        """Test trajectory creation with orientations."""
        traj = Trajectory(self.timestamps, self.positions, self.orientations)
        self.assertTrue(traj.has_orientation())

    def test_creation_without_orientations(self):
        """Test trajectory creation without orientations."""
        traj = Trajectory(self.timestamps, self.positions)
        self.assertFalse(traj.has_orientation())

    def test_invalid_positions_type(self):
        """Test that invalid position types raise TypeError."""
        with self.assertRaises(TypeError):
            Trajectory(self.timestamps, [[1, 2, 3], [4, 5, 6]])

    def test_invalid_orientations_type(self):
        """Test that invalid orientation types raise TypeError."""
        with self.assertRaises(TypeError):
            Trajectory(self.timestamps, self.positions, orientations=[[0, 0, 0, 1]])


class TestTrajectoryProperties(unittest.TestCase):
    """Test suite for Trajectory properties."""

    def setUp(self):
        """Set up test fixtures."""
        self.timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        self.positions = Cartographic(
            longitude=[3.8777, 4.8391, 5.4524, 6.2345],
            latitude=[43.6135, 43.9422, 43.5309, 43.7891],
            height=[300.0, 400.0, 500.0, 600.0]
        )
        self.traj = Trajectory(self.timestamps, self.positions)

    def test_timestamps_property(self):
        """Test timestamps property returns correct values."""
        assert_array_equal(self.traj.timestamps, self.timestamps)

    def test_positions_property(self):
        """Test positions property returns CartesianECEF."""
        self.assertIsInstance(self.traj.positions, CartesianECEF)
        self.assertEqual(len(self.traj.positions), 4)

    def test_arc_lengths_property(self):
        """Test arc_lengths property computation."""
        arc_lengths = self.traj.arc_lengths
        self.assertEqual(len(arc_lengths), 3)
        self.assertTrue(all(arc_lengths > 0))

    def test_velocities_property(self):
        """Test velocities property computation."""
        velocities = self.traj.velocities
        self.assertEqual(len(velocities), 3)
        self.assertTrue(all(velocities > 0))

    def test_velocities_insufficient_points(self):
        """Test that velocities raises error with single point."""
        single_traj = Trajectory(
            timestamps=[0.0],
            positions=Cartographic(longitude=3.8, latitude=43.6, height=300.0)
        )
        with self.assertRaises(ValueError):
            _ = single_traj.velocities

    def test_total_arc_length(self):
        """Test total_arc_length computation."""
        total = self.traj.total_arc_length()
        self.assertGreater(total, 0)
        self.assertAlmostEqual(total, np.sum(self.traj.arc_lengths))

    def test_sampling_rate(self):
        """Test sampling_rate property."""
        self.assertAlmostEqual(self.traj.sampling_rate, 1.0)

    def test_sampling_rate_insufficient_points(self):
        """Test that sampling_rate raises error with single point."""
        single_traj = Trajectory(
            timestamps=[0.0],
            positions=Cartographic(longitude=3.8, latitude=43.6, height=300.0)
        )
        with self.assertRaises(ValueError):
            _ = single_traj.sampling_rate


class TestTrajectoryOperations(unittest.TestCase):
    """Test suite for Trajectory operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        self.positions = Cartographic(
            longitude=[3.8777, 4.8391, 5.4524, 6.2345],
            latitude=[43.6135, 43.9422, 43.5309, 43.7891],
            height=[300.0, 400.0, 500.0, 600.0]
        )
        self.traj = Trajectory(self.timestamps, self.positions)

    def test_len(self):
        """Test __len__ method."""
        self.assertEqual(len(self.traj), 4)

    def test_getitem_single_index(self):
        """Test __getitem__ with single index."""
        subset = self.traj[0]
        self.assertEqual(len(subset), 1)

    def test_getitem_slice(self):
        """Test __getitem__ with slice."""
        subset = self.traj[:2]
        self.assertEqual(len(subset), 2)

    def test_repr(self):
        """Test __repr__ method."""
        repr_str = repr(self.traj)
        self.assertIn("Trajectory samples", repr_str)

    def test_sort_ascending(self):
        """Test sort method in ascending order."""
        unsorted_traj = Trajectory(
            timestamps=[2, 0, 3, 1],
            positions=self.positions
        )
        unsorted_traj.sort()
        assert_array_equal(unsorted_traj.timestamps, [0, 1, 2, 3])

    def test_sort_descending(self):
        """Test sort method in descending order."""
        unsorted_traj = Trajectory(
            timestamps=[2, 0, 3, 1],
            positions=self.positions
        )
        unsorted_traj.sort(reverse=True)
        assert_array_equal(unsorted_traj.timestamps, [3, 2, 1, 0])

    def test_sort_inplace_false(self):
        """Test sort method with inplace=False."""
        unsorted_traj = Trajectory(
            timestamps=[2, 0, 3, 1],
            positions=self.positions
        )
        sorted_traj = unsorted_traj.sort(inplace=False)
        assert_array_equal(sorted_traj.timestamps, [0, 1, 2, 3])
        assert_array_equal(unsorted_traj.timestamps, [2, 0, 3, 1])


class TestTrajectoryInterpolation(unittest.TestCase):
    """Test suite for Trajectory interpolation methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        self.positions = Cartographic(
            longitude=[3.8777, 4.8391, 5.4524, 6.2345],
            latitude=[43.6135, 43.9422, 43.5309, 43.7891],
            height=[300.0, 400.0, 500.0, 600.0]
        )
        self.traj = Trajectory(self.timestamps, self.positions)

    def test_resample(self):
        """Test resample method."""
        resampled = self.traj.resample(2.0)
        self.assertGreater(len(resampled), len(self.traj))
        self.assertAlmostEqual(resampled.sampling_rate, 2.0, places=1)

    def test_resample_invalid_type(self):
        """Test resample with invalid type raises TypeError."""
        with self.assertRaises(TypeError):
            self.traj.resample("invalid")

    def test_resample_invalid_value(self):
        """Test resample with invalid value raises ValueError."""
        with self.assertRaises(ValueError):
            self.traj.resample(-1.0)

    def test_interp(self):
        """Test interp method."""
        new_timestamps = np.array([0.5, 1.5, 2.5])
        interp_traj = self.traj.interp(new_timestamps)
        self.assertEqual(len(interp_traj), 3)

    def test_interp_empty_timestamps(self):
        """Test interp with empty timestamps raises ValueError."""
        with self.assertRaises(ValueError):
            self.traj.interp([])

    def test_interp_out_of_range(self):
        """Test interp with out-of-range timestamps raises ValueError."""
        with self.assertRaises(ValueError):
            self.traj.interp([-1.0, 0.5])


class TestTrajectoryConcatenation(unittest.TestCase):
    """Test suite for Trajectory concatenation."""

    def test_concatenate_basic(self):
        """Test basic concatenation of two trajectories."""
        traj1 = Trajectory(
            timestamps=[0.0, 1.0],
            positions=CartesianECEF(x=[1000, 1100], y=[2000, 2100], z=[3000, 3100])
        )
        traj2 = Trajectory(
            timestamps=[2.0, 3.0],
            positions=CartesianECEF(x=[1200, 1300], y=[2200, 2300], z=[3200, 3300])
        )
        combined = Trajectory.concatenate([traj1, traj2])
        self.assertEqual(len(combined), 4)

    def test_concatenate_with_orientations(self):
        """Test concatenation preserves orientations when all have them."""
        orientations = Rotation.from_euler("ZYX", [[0, 0, 0], [10, 0, 0]], degrees=True)
        traj1 = Trajectory(
            timestamps=[0.0, 1.0],
            positions=CartesianECEF(x=[1000, 1100], y=[2000, 2100], z=[3000, 3100]),
            orientations=orientations
        )
        traj2 = Trajectory(
            timestamps=[2.0, 3.0],
            positions=CartesianECEF(x=[1200, 1300], y=[2200, 2300], z=[3200, 3300]),
            orientations=orientations
        )
        combined = Trajectory.concatenate([traj1, traj2])
        self.assertTrue(combined.has_orientation())

    def test_concatenate_mixed_orientations(self):
        """Test concatenation drops orientations when not all have them."""
        orientations = Rotation.from_euler("ZYX", [[0, 0, 0], [10, 0, 0]], degrees=True)
        traj1 = Trajectory(
            timestamps=[0.0, 1.0],
            positions=CartesianECEF(x=[1000, 1100], y=[2000, 2100], z=[3000, 3100]),
            orientations=orientations
        )
        traj2 = Trajectory(
            timestamps=[2.0, 3.0],
            positions=CartesianECEF(x=[1200, 1300], y=[2200, 2300], z=[3200, 3300])
        )
        combined = Trajectory.concatenate([traj1, traj2])
        self.assertFalse(combined.has_orientation())

    def test_concatenate_empty_raises(self):
        """Test concatenation of empty list raises ValueError."""
        with self.assertRaises(ValueError):
            Trajectory.concatenate([])

    def test_concatenate_invalid_type_raises(self):
        """Test concatenation of non-Trajectory raises ValueError."""
        with self.assertRaises(ValueError):
            Trajectory.concatenate([[1, 2, 3]])


class TestTrajectoryConversion(unittest.TestCase):
    """Test suite for Trajectory conversion methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        self.positions = Cartographic(
            longitude=[3.8777, 4.8391, 5.4524, 6.2345],
            latitude=[43.6135, 43.9422, 43.5309, 43.7891],
            height=[300.0, 400.0, 500.0, 600.0]
        )
        self.traj = Trajectory(self.timestamps, self.positions)

    def test_to_numpy(self):
        """Test to_numpy method."""
        data = self.traj.to_numpy()
        self.assertEqual(data.dtype, TRAJ_DTYPE)
        self.assertEqual(len(data), 4)

    def test_from_numpy(self):
        """Test from_numpy method."""
        data = np.array([
            (0., 3.8777, 43.6135, 300., 0., 0., 0.),
            (1., 4.8391, 43.9422, 400., 0., 0., 0.),
        ], dtype=TRAJ_DTYPE)
        traj = Trajectory.from_numpy(data)
        self.assertEqual(len(traj), 2)

    def test_from_numpy_invalid_type(self):
        """Test from_numpy with invalid type raises TypeError."""
        with self.assertRaises(TypeError):
            Trajectory.from_numpy([[1, 2, 3]])

    def test_from_numpy_invalid_dtype(self):
        """Test from_numpy with invalid dtype raises ValueError."""
        data = np.array([(0., 1., 2.)], dtype=[('a', float), ('b', float), ('c', float)])
        with self.assertRaises(ValueError):
            Trajectory.from_numpy(data)


class TestTrajectoryIO(unittest.TestCase):
    """Test suite for Trajectory I/O operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        self.positions = Cartographic(
            longitude=[3.8777, 4.8391, 5.4524, 6.2345],
            latitude=[43.6135, 43.9422, 43.5309, 43.7891],
            height=[300.0, 400.0, 500.0, 600.0]
        )
        self.traj = Trajectory(self.timestamps, self.positions)

    def test_save_csv_and_read_csv(self):
        """Test save_csv and read_csv round-trip."""
        with tempfile.NamedTemporaryFile(suffix='.traj.csv', delete=False) as f:
            temp_path = f.name
        
        filepath = self.traj.save_csv(temp_path)
        self.assertTrue(filepath.exists())
        
        loaded_traj = Trajectory.read_csv(filepath)
        self.assertEqual(len(loaded_traj), len(self.traj))
        
        # Clean up
        filepath.unlink()

    def test_save_npy(self):
        """Test save_npy method."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        filepath = self.traj.save_npy(temp_path)
        self.assertTrue(filepath.exists())
        
        # Verify file contents
        loaded_data = np.load(filepath)
        self.assertEqual(len(loaded_data), 4)
        
        # Clean up
        filepath.unlink()

    def test_read_csv_file_not_found(self):
        """Test read_csv with non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            Trajectory.read_csv("nonexistent.traj.csv")

    def test_read_csv_invalid_extension(self):
        """Test read_csv with invalid extension raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            with self.assertRaises(ValueError):
                Trajectory.read_csv(f.name)


class TestTrajectoryMakeStraightLine(unittest.TestCase):
    """Test suite for Trajectory.make_straight_line static method."""

    def test_make_straight_line_basic(self):
        """Test basic straight line creation."""
        start_pos = Cartographic(longitude=3.8777, latitude=43.6135, height=300.0)
        velocity = CartesianLocalENU(x=100.0, y=0.0, z=0.0, origin=start_pos)
        
        traj = Trajectory.make_straight_line(velocity, duration=10.0, num_samples=11)
        
        self.assertEqual(len(traj), 11)
        self.assertTrue(traj.has_orientation())
        # Total distance should be ~1000m for 100 m/s over 10s
        self.assertAlmostEqual(traj.total_arc_length(), 1000.0, places=0)

    def test_make_straight_line_invalid_type(self):
        """Test make_straight_line with invalid velocity type raises TypeError."""
        with self.assertRaises(TypeError):
            Trajectory.make_straight_line([100, 0, 0], duration=10.0)


if __name__ == '__main__':
    unittest.main()
