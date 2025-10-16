import re
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from sargeom.coordinates.cartesian import Cartesian3, CartesianECEF, Cartographic


TRAJ_DTYPE = [
    ('TIMESTAMP_S', '<f8'),
    ('LON_WGS84_DEG', '<f8'),
    ('LAT_WGS84_DEG', '<f8'),
    ('HEIGHT_WGS84_M', '<f8'),
    ('HEADING_DEG', '<f8'),
    ('ELEVATION_DEG', '<f8'),
    ('BANK_DEG', '<f8'),
]

PAMELA_TRAJ_DTYPE = [
    ('longitude_rad', '<f8'),
    ('latitude_rad', '<f8'),
    ('height_m', '<f8'),
    ('heading_rad', '<f4'),
    ('elevation_rad', '<f4'),
    ('bank_rad', '<f4'),
]

PAMELA_POS_DTYPE = [
    ('timestamp_s', '<f8'),
    ('latitude_deg', '<f8'),
    ('longitude_deg', '<f8'),
    ('height_m', '<f8'),
    ('velocity_north_m_s', '<f4'),
    ('velocity_east_m_s', '<f4'),
    ('velocity_up_m_s', '<f4'),
    ('bank_deg', '<f4'),
    ('elevation_deg', '<f4'),
    ('heading_deg', '<f4'),
    ('std_latitude_m', '<f4'),
    ('std_longitude_m', '<f4'),
    ('std_height_m', '<f4'),
]

# This class represents a trajectory of positions and orientations over time.
# Inspired by https://gereon-t.github.io/trajectopy/Documentation/Trajectory/
class Trajectory:
    """
    A Trajectory object represents a sequence of positions and orientations over time.

    It is defined by the following characteristics:

    - Timestamps are expressed in seconds. They may correspond to UTC, GPS Seconds of Week (SOW), Time of Day (TOD), or a custom time reference.
    - Positions are provided in either the WGS84 geographic coordinate system (`EPSG:4979 <https://epsg.org/crs_4979/WGS-84.html>`_) or the WGS84 geocentric coordinate system (`EPSG:4978 <https://epsg.org/crs_4978/WGS-84.html>`_).
    - Orientations are defined in the local North-East-Down (NED) Cartesian frame, relative to the associated position coordinates.

    Parameters
    ----------
    timestamps : array_like
        1D array of timestamps corresponding to each trajectory sample.
    positions : :class:`sargeom.coordinates.CartesianECEF` or :class:`sargeom.coordinates.Cartographic`
        Array of positions in either ECEF (x, y, z) or geographic (latitude, longitude, altitude) format.
    orientations : :class:`scipy.spatial.transform.Rotation`, optional
        Sequence of orientations as `Rotation objects`. Defined in the NED frame. Default is `None`.

    Raises
    ------
    :class:`TypeError`
        - If positions are not of type CartesianECEF or Cartographic.
        - If orientations are provided but not of type scipy.spatial.transform.Rotation.

    Examples
    --------
    Create a Trajectory instance using ECEF (Cartesian) coordinates:

    >>> timestamps = np.array([0, 1, 2, 3])
    >>> positions = CartesianECEF(
    ...     x=[4614831.06382533, 4583825.9258778, 4610933.91105407],
    ...     y=[312803.18870294, 388064.96749322, 440116.57314554],
    ...     z=[4377307.25608437, 4403747.15229078, 4370795.76589696]
    ... )
    >>> traj = Trajectory(timestamps, positions)

    Create a Trajectory instance using geographic (Cartographic) coordinates:

    >>> positions = Cartographic(
    ...     longitude=[3.8777, 4.8391, 5.4524, 6.2345],
    ...     latitude=[43.6135, 43.9422, 43.5309, 43.7891],
    ...     height=[300.0, 400.0, 500.0, 600.0]
    ... )
    >>> traj = Trajectory(timestamps, positions)

    Create a Trajectory instance with orientations:

    >>> from scipy.spatial.transform import Rotation
    >>> orientations = Rotation.from_euler("ZYX", [[0, 0, 0], [90, 0, 0], [180, 0, 0], [270, 0, 0]], degrees=True)
    >>> traj = Trajectory(timestamps, positions, orientations)
    """
    def __init__(self, timestamps, positions, orientations=None):
        self._timestamps = np.asarray(timestamps)
        if isinstance(positions, CartesianECEF):
            self._positions = positions
        elif isinstance(positions, Cartographic):
            self._positions = positions.to_ecef()
        else:
            raise TypeError("Positions must be of type CartesianECEF or Cartographic.")
        if orientations is not None:
            if isinstance(orientations, Rotation):
                self._orientations = orientations
            else:
                raise TypeError("Orientations must be of type scipy.spatial.transform.Rotation.")
        else:
            self._orientations = None

    def __len__(self):
        """
        Return the number of samples in the trajectory.

        Returns
        -------
        :class:`int`
            Number of trajectory samples.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> len(traj)
        4
        """
        return len(self._timestamps)

    def __getitem__(self, item):
        """
        Return a subset of the trajectory.

        Parameters
        ----------
        item : :class:`int` or :class:`slice`
            Index or slice to extract a subset of the trajectory.

        Returns
        -------
        :class:`Trajectory`
            A new Trajectory instance containing the subset.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj[:2]
        Trajectory samples (t, x, y, z, h, e, b)
        [(0., 3.8777, 43.6135, 300., 0., 0., 0.)
         (1., 4.8391, 43.9422, 400., 0., 0., 0.)]
        """
        return Trajectory(
            timestamps=self.timestamps[item],
            positions=self.positions[item],
            orientations=self.orientations[item] if self.has_orientation() else None
        )

    def __repr__(self):
        """
        Returns a string representation of the Trajectory instance.

        Returns
        -------
        :class:`str`
            A string representation of the Trajectory instance.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj
        Trajectory samples (t, x, y, z, h, e, b)
        [(0., 3.8777, 43.6135, 300., 0., 0., 0.)
         (1., 4.8391, 43.9422, 400., 0., 0., 0.)
         (2., 5.4524, 43.5309, 500., 0., 0., 0.)
         (3., 6.2345, 43.7891, 600., 0., 0., 0.)]
        """
        return f"Trajectory samples (t, x, y, z, h, e, b)\n{self.to_numpy()}"

    def sort(self, inplace=True, reverse=False):
        """
        Sort the trajectory by timestamps.

        Parameters
        ----------
        inplace : :class:`bool`, optional
            If True, sort the trajectory in place. Default is True.
        reverse : :class:`bool`, optional
            If True, sort in descending order. Default is False.

        Returns
        -------
        :class:`Trajectory` or None
            The sorted Trajectory instance if inplace is False, otherwise None.

        Examples
        --------
        Sort the trajectory in place:

        >>> traj = Trajectory(
        ...     timestamps=[2, 0, 1, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj.sort()
        Trajectory samples (t, x, y, z, h, e, b)
        [(0., 4.8391, 43.9422, 400., 0., 0., 0.)
         (1., 5.4524, 43.5309, 500., 0., 0., 0.)
         (2., 3.8777, 43.6135, 300., 0., 0., 0.)
         (3., 6.2345, 43.7891, 600., 0., 0., 0.)]

        Sort the trajectory and return a new instance:

        >>> sorted_traj = traj.sort(inplace=False)
        """
        indices = np.argsort(self._timestamps)
        if reverse:
            indices = indices[::-1]
        if inplace:
            self._timestamps = self._timestamps[indices]
            self._positions = self._positions[indices]
            if self.has_orientation():
                self._orientations = self._orientations[indices]
            return self
        else:
            return Trajectory(
                timestamps=self._timestamps[indices],
                positions=self._positions[indices],
                orientations=self._orientations[indices] if self.has_orientation() else None
            )

    @property
    def timestamps(self):
        """
        Timestamps of the trajectory samples, in seconds.

        Returns
        -------
        :class:`numpy.ndarray`
            1D array of timestamps (e.g., UTC, GPS SOW, TOD).

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj.timestamps
        array([0, 1, 2, 3])
        """
        return self._timestamps

    @property
    def positions(self):
        """
        Positions of the trajectory samples, in ECEF coordinates.

        Returns
        -------
        :class:`sargeom.coordinates.CartesianECEF`
            3D positions expressed in the WGS84 geocentric frame (EPSG:4978).

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj.positions
        XYZ CartesianECEF points
        [[4614831.06382533  312803.18870294 4377307.25608437]
         [4583825.9258778   388064.96749322 4403747.15229078]
         [4610933.91105407  440116.57314554 4370795.76589696]
         [4584879.02442076  500870.74890955 4391620.5067715 ]]
        """
        return self._positions

    @property
    def velocities(self):
        """
        Velocities between consecutive trajectory samples.

        Returns
        -------
        :class:`numpy.ndarray`
            1D array of velocity magnitudes, in meters per second.

        Raises
        ------
        :class:`ValueError`
            If there are fewer than two timestamps.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj.velocities
        array([85584.58995186, 67305.32205239, 69307.98527392])
        """
        if len(self._timestamps) < 2:
            raise ValueError("Not enough timestamps to compute velocities.")
        dt = np.diff(self._timestamps)
        return self.arc_lengths / dt

    def has_orientation(self):
        """
        Whether the trajectory has orientation data.

        Returns
        -------
        :class:`bool`
            True if orientations are defined, False otherwise.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj.has_orientation()
        False
        """
        return self._orientations is not None

    @property
    def orientations(self):
        """
        Orientations of the trajectory samples, in the NED frame.

        Returns
        -------
        :class:`scipy.spatial.transform.Rotation`
            Sequence of orientations as a `Rotation` object.

        Raises
        ------
        :class:`ValueError`
            If the trajectory has no orientations.

        Notes
        -----
        The orientations are defined in the local North-East-Down (NED) Cartesian frame, relative to the associated position coordinates.
        The orientations can be converted to quaternions, Euler angles, or other representations using the methods provided by the `Rotation` class.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation
        >>> attitude = Rotation.from_euler("ZYX", [[0, 0, 0], [90, 0, 0], [180, 0, 0], [270, 0, 0]], degrees=True)
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     ),
        ...     orientations=attitude
        ... )
        >>> traj.orientations.as_quat()
        array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
               [0.00000000e+00, 0.00000000e+00, 7.07106781e-01, 7.07106781e-01],
               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 6.12323400e-17],
               [0.00000000e+00, 0.00000000e+00, 7.07106781e-01, -7.07106781e-01]])
        """
        if not self.has_orientation():
            raise ValueError("This trajectory does not have orientations.")
        return self._orientations

    @property
    def arc_lengths(self):
        """
        Arc lengths between consecutive trajectory positions.

        Returns
        -------
        :class:`numpy.ndarray`
            1D array of distances in meters.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj.arc_lengths
        array([85584.58995186, 67305.32205239, 69307.98527392])
        """
        return Cartesian3.distance(self._positions[1:], self._positions[:-1])

    def total_arc_length(self):
        """
        Compute the total arc length of the trajectory.

        Returns
        -------
        :class:`float`
            Sum of distances between consecutive positions, in meters.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj.total_arc_length()
        np.float64(231835.03546103595)
        """
        return np.sum(self.arc_lengths)

    @property
    def sampling_rate(self):
        """
        Sampling rate of the trajectory.

        Returns
        -------
        :class:`float`
            Sampling frequency in Hz.

        Raises
        ------
        :class:`ValueError`
            If there are fewer than two timestamps.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj.sampling_rate
        np.float64(1.0)
        """
        if len(self._timestamps) < 2:
            raise ValueError("Not enough timestamps to compute sampling rate.")
        dt = np.diff(self._timestamps)
        return 1 / np.mean(dt)

    def resample(self, sampling_rate):
        """
        Resample the trajectory at a specified sampling rate.

        Parameters
        ----------
        sampling_rate : :class:`float`
            The desired sampling rate in Hz.

        Returns
        -------
        :class:`Trajectory`
            A new Trajectory instance with resampled data.

        Raises
        ------
        :class:`TypeError`
            If the sampling rate is not a number.
        :class:`ValueError`
            If the sampling rate is not positive.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> resampled_traj = traj.resample(2.0)
        """
        if not isinstance(sampling_rate, (int, float)):
            raise TypeError("Sampling rate must be a number.")
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive.")

        new_timestamps = np.arange(self._timestamps[0], self._timestamps[-1], 1 / sampling_rate)
        new_positions = self._positions.interp(self._timestamps, new_timestamps)
        
        if self.has_orientation():
            serl = Slerp(self._timestamps, self._orientations)
            return Trajectory(new_timestamps, new_positions, serl(new_timestamps))
        else:
            return Trajectory(new_timestamps, new_positions)

    def interp(self, new_timestamps):
        """
        Interpolate the trajectory to new timestamps.

        Parameters
        ----------
        new_timestamps : array_like
            Array of new timestamps to interpolate the trajectory to.

        Returns
        -------
        :class:`Trajectory`
            A new Trajectory instance with interpolated data.

        Raises
        ------
        :class:`ValueError`
            If the new timestamps are not within the range of existing timestamps.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> interp_traj = traj.interp([0.5, 1.5, 2.5])
        """
        if not isinstance(new_timestamps, np.ndarray):
            new_timestamps = np.asarray(new_timestamps)
        if len(new_timestamps) == 0:
            raise ValueError("Timestamps array cannot be empty.")
        if new_timestamps[0] < self._timestamps[0] or new_timestamps[-1] > self._timestamps[-1]:
            raise ValueError("New timestamps must be within the range of existing timestamps.")

        new_positions = self.positions.interp(self._timestamps, new_timestamps)
        if self.has_orientation():
            serl = Slerp(self._timestamps, self.orientations)
            return Trajectory(new_timestamps, new_positions, serl(new_timestamps))
        else:
            return Trajectory(new_timestamps, new_positions)

    @classmethod
    def concatenate(cls, trajectories):
        """
        Concatenate a sequence of Trajectory objects into a single object.
        
        Parameters
        ----------
        trajectories : sequence of :class:`sargeom.Trajectory`
            The trajectories to concatenate. Can be a list, tuple, or any iterable.

        Returns
        -------
        :class:`sargeom.Trajectory`
            A new Trajectory instance containing the concatenated data.

        Raises
        ------
        :class:`ValueError`
            - If the input list is empty.
            - If any item in the list is not a Trajectory instance.

        Notes
        -----
        - Timestamps and positions are always concatenated.
        - Orientations are concatenated only if ALL input trajectories have orientations.
        - If any trajectory lacks orientations, the result will have no orientations.
        - The order of concatenation follows the order of input trajectories.

        Examples
        --------
        Concatenate two trajectories with orientations:

        >>> import numpy as np
        >>> from scipy.spatial.transform import Rotation
        >>> timestamps_1 = np.array([0.0, 1.0])
        >>> positions_1 = CartesianECEF(x=[1000, 1100], y=[2000, 2100], z=[3000, 3100])
        >>> orientations_1 = Rotation.from_euler("ZYX", [[0, 0, 0], [10, 0, 0]], degrees=True)
        >>> traj_1 = Trajectory(timestamps_1, positions_1, orientations_1)
        >>> 
        >>> timestamps_2 = np.array([2.0, 3.0])
        >>> positions_2 = CartesianECEF(x=[1200, 1300], y=[2200, 2300], z=[3200, 3300])
        >>> orientations_2 = Rotation.from_euler("ZYX", [[20, 0, 0], [30, 0, 0]], degrees=True)
        >>> traj_2 = Trajectory(timestamps_2, positions_2, orientations_2)
        >>> 
        >>> combined = Trajectory.concatenate([traj_1, traj_2])
        >>> len(combined)
        4
        >>> combined.has_orientation()
        True

        Concatenate trajectories without orientations:

        >>> traj_no_orient_1 = Trajectory(timestamps_1, positions_1)
        >>> traj_no_orient_2 = Trajectory(timestamps_2, positions_2)
        >>> combined_no_orient = Trajectory.concatenate([traj_no_orient_1, traj_no_orient_2])
        >>> combined_no_orient.has_orientation()
        False

        Concatenate single trajectory with multiple points:

        >>> single_position = Cartographic(longitude=2.0, latitude=46.0, height=0.0)
        >>> single_traj = Trajectory(timestamps=[10.0], positions=single_position)
        >>> multi_positions = Cartographic(
        ...     longitude=[3.0, 4.0, 5.0], 
        ...     latitude=[47.0, 48.0, 49.0], 
        ...     height=[100.0, 200.0, 300.0]
        ... )
        >>> multi_traj = Trajectory(timestamps=[11.0, 12.0, 13.0], positions=multi_positions)
        >>> result = Trajectory.concatenate([single_traj, multi_traj])
        >>> len(result)
        4
        """
        # Convert to list if not already a sequence
        if not hasattr(trajectories, '__iter__'):
            raise TypeError("trajectories must be an iterable (list, tuple, etc.)")
        
        trajectories = list(trajectories)
        
        if not trajectories:
            raise ValueError("No trajectories to concatenate.")
        if not all(isinstance(traj, Trajectory) for traj in trajectories):
            raise ValueError("All items in the list must be Trajectory instances.")

        # Concatenate timestamps and positions
        timestamps = np.concatenate([traj._timestamps for traj in trajectories])
        positions = CartesianECEF.concatenate([traj._positions for traj in trajectories])

        # Concatenate orientations if they exist
        if all(traj.has_orientation() for traj in trajectories):
            orientations = Rotation.concatenate([traj._orientations for traj in trajectories])
        else:
            orientations = None

        return cls(timestamps, positions, orientations)

    def plot(self, **kwargs):
        # TODO: Implement plotting functionality
        # See: https://github.com/gereon-t/trajectopy/blob/main/trajectopy/core/plotting/mpl/trajectory.py
        raise NotImplementedError("Plotting functionality is not implemented yet.")

    @staticmethod
    def from_numpy(data):
        """
        Create a Trajectory instance from a numpy structured array.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            Input data as a numpy structured array using the :data:`TRAJ_DTYPE` type.

        Returns
        -------
        :class:`Trajectory`
            A new Trajectory instance.

        Raises
        ------
        :class:`TypeError`
            If the input data is not a numpy array.
        :class:`ValueError`
            If the input data does not have the correct dtype.

        Examples
        --------
        >>> data = np.array([
        ...     (0., 3.8777, 43.6135, 300., 0., 0., 0.),
        ...     (1., 4.8391, 43.9422, 400., 0., 0., 0.),
        ...     (2., 5.4524, 43.5309, 500., 0., 0., 0.),
        ...     (3., 6.2345, 43.7891, 600., 0., 0., 0.)
        ... ], dtype=TRAJ_DTYPE)
        >>> traj = Trajectory.from_numpy(data)
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array.")
        if data.dtype != TRAJ_DTYPE:
            raise ValueError(f"Input data must have dtype {TRAJ_DTYPE}.")
        
        timestamps = data['TIMESTAMP_S']
        positions = Cartographic(
            latitude=data['LAT_WGS84_DEG'],
            longitude=data['LON_WGS84_DEG'],
            height=data['HEIGHT_WGS84_M']
        ).to_ecef()
        
        if 'HEADING_DEG' in data.dtype.names:
            orientations = Rotation.from_euler(
                "ZYX",
                np.column_stack([data['HEADING_DEG'], data['ELEVATION_DEG'], data['BANK_DEG']]),
                degrees=True
            )
        else:
            orientations = None
        
        return Trajectory(timestamps, positions, orientations)

    def to_numpy(self):
        """
        Convert the Trajectory instance to a numpy structured array.

        Returns
        -------
        :class:`numpy.ndarray`
            The trajectory data as a numpy structured array using the :data:`TRAJ_DTYPE` type.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj.to_numpy()
        array([(0., 3.8777, 43.6135, 300., 0., 0., 0.),
               (1., 4.8391, 43.9422, 400., 0., 0., 0.),
               (2., 5.4524, 43.5309, 500., 0., 0., 0.),
               (3., 6.2345, 43.7891, 600., 0., 0., 0.)],
              dtype=[('TIMESTAMP_S', '<f8'), ('LON_WGS84_DEG', '<f8'), ('LAT_WGS84_DEG', '<f8'), ('HEIGHT_WGS84_M', '<f8'), ('HEADING_DEG', '<f8'), ('ELEVATION_DEG', '<f8'), ('BANK_DEG', '<f8')])
        """
        cartographic_positions = self._positions.to_cartographic()
        
        # Extract orientations if available
        if self.has_orientation():
            heading, elevation, bank = self._orientations.as_euler("ZYX", degrees=True).T
            heading %= 360  # Normalize heading angle to [0, 360]
        else:
            heading = elevation = bank = np.zeros(len(self))
        
        # Create the structured array
        data = np.empty(len(self), dtype=TRAJ_DTYPE)
        data['TIMESTAMP_S'] = self._timestamps
        data['LON_WGS84_DEG'] = cartographic_positions.longitude
        data['LAT_WGS84_DEG'] = cartographic_positions.latitude
        data['HEIGHT_WGS84_M'] = cartographic_positions.height
        data['HEADING_DEG'] = heading
        data['ELEVATION_DEG'] = elevation
        data['BANK_DEG'] = bank
        
        return data

    def to_pandas(self):
        """
        Convert the Trajectory instance to a pandas DataFrame.

        Returns
        -------
        :class:`pandas.DataFrame`
            The trajectory data as a pandas DataFrame.

        Raises
        ------
        :class:`ModuleNotFoundError`
            If pandas is not installed.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> traj.to_pandas()
                TIMESTAMP_S  LON_WGS84_DEG  LAT_WGS84_DEG  HEIGHT_WGS84_M  HEADING_DEG  ELEVATION_DEG  BANK_DEG
        0          0.0         3.8777        43.6135           300.0          0.0            0.0       0.0
        1          1.0         4.8391        43.9422           400.0          0.0            0.0       0.0
        2          2.0         5.4524        43.5309           500.0          0.0            0.0       0.0
        3          3.0         6.2345        43.7891           600.0          0.0            0.0       0.0
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Pandas is not installed. Please follow the instructions on https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html"
            )

        data = self.to_numpy()
        return pd.DataFrame(data)

    @classmethod
    def read_pivot(cls, filename, actor_name=None, actor_type=None):
        """
        Read a PIVOT .h5 file and create a Trajectory instance.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The filename or path to the .h5 file.
        actor_name : :class:`str`, optional
            The name of the specific actor to load. If None, loads the first actor found.
        actor_type : :class:`str`, optional
            The type of actor to load (e.g., 'TX_ANTENNA').
            May be one of: 'TX_PLATFORM', 'RX_PLATFORM', 'TX_ANTENNA', 'RX_ANTENNA', 'TARGET'.
            If None, loads any actor type.

        Returns
        -------
        :class:`Trajectory`
            A new Trajectory instance.

        Raises
        ------
        :class:`ImportError`
            If the pivot library is not installed.
        :class:`FileNotFoundError`
            If the file does not exist.
        :class:`ValueError`
            - If the file does not have a .h5 extension.
            - If no actors are found in the file.
            - If the specified actor is not found.
            - If the actor_type is not valid.
        :class:`KeyError`
            If required axis labels are missing from the actor data.

        Examples
        --------
        Load the first actor from a PIVOT file:

        >>> traj = Trajectory.read_pivot("trajectory.h5")

        Load a specific actor by name:

        >>> traj = Trajectory.read_pivot("trajectory.h5", actor_name="TrajectoryName_TrajectoryOrigin_TrajectoryType")

        Load an actor by type:

        >>> traj = Trajectory.read_pivot("trajectory.h5", actor_type="TX_ANTENNA")
        """
        try:
            from pivot.darpy import AxisLabelEnum
            from pivot.piactor import Actor, ActorTypeEnum
            from pivot.pivotutil import pivot_version
            print(f"Using pivot library version: {pivot_version()}")
        except ImportError:
            raise ImportError("The pivot library is not installed. Please install it to read PIVOT files.")

        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"File {filename} does not exist.")
        if not filename.suffix == '.h5':
            raise ValueError("File must have a .h5 extension.")

        # Validate actor_type parameter if provided
        if actor_type not in ['TX_PLATFORM', 'RX_PLATFORM', 'TX_ANTENNA', 'RX_ANTENNA', 'TARGET']:
            raise ValueError("'actor_type' must be one of: 'TX_PLATFORM', 'RX_PLATFORM', 'TX_ANTENNA', 'RX_ANTENNA', 'TARGET'")

        # Load all actors from the PIVOT file
        try:
            actors = Actor.load(filename)
        except Exception as e:
            raise ValueError(f"Failed to load PIVOT file: {e}")

        if not actors:
            raise ValueError("No actors found in the PIVOT file.")

        # Filter actors based on the provided criteria
        selected_actor = None

        if actor_name is not None:
            # Filter by actor name
            matching_actors = [act for act in actors if act.name == actor_name]
            if not matching_actors:
                available_names = [act.name for act in actors]
                raise ValueError(
                    f"Actor with name '{actor_name}' not found. "
                    f"Available actors: {available_names}"
                )
            selected_actor = matching_actors[0]
        elif actor_type is not None:
            # Filter by actor type
            target_type = ActorTypeEnum[actor_type]
            matching_actors = [act for act in actors if act.type is target_type]
            if not matching_actors:
                available_types = [act.type.name for act in actors]
                raise ValueError(
                    f"Actor with type '{actor_type}' not found. "
                    f"Available actor types: {available_types}"
                )
            selected_actor = matching_actors[0]
        else:
            # Use the first available actor
            selected_actor = actors[0]

        # Extract trajectory data from the selected actor

        # Extract timestamps
        time_axis = selected_actor.get_axis(AxisLabelEnum.TIME)
        timestamps = np.array(time_axis.values)

        # Extract ECEF positions
        pos_x_axis = selected_actor.get_axis(AxisLabelEnum.POS_X_ECEF)
        pos_y_axis = selected_actor.get_axis(AxisLabelEnum.POS_Y_ECEF)
        pos_z_axis = selected_actor.get_axis(AxisLabelEnum.POS_Z_ECEF)

        positions = CartesianECEF(
            x=np.array(pos_x_axis.values),
            y=np.array(pos_y_axis.values),
            z=np.array(pos_z_axis.values)
        )

        # Extract orientations

        # Get direction vectors for x and y axes in ECEF frame
        dir_x_x_axis = selected_actor.get_axis(AxisLabelEnum.DIR_x_X_ECEF)
        dir_x_y_axis = selected_actor.get_axis(AxisLabelEnum.DIR_x_Y_ECEF)
        dir_x_z_axis = selected_actor.get_axis(AxisLabelEnum.DIR_x_Z_ECEF)

        dir_y_x_axis = selected_actor.get_axis(AxisLabelEnum.DIR_y_X_ECEF)
        dir_y_y_axis = selected_actor.get_axis(AxisLabelEnum.DIR_y_Y_ECEF)
        dir_y_z_axis = selected_actor.get_axis(AxisLabelEnum.DIR_y_Z_ECEF)

        # Construct direction vectors
        x_axis_dir = np.column_stack([
            np.array(dir_x_x_axis.values),
            np.array(dir_x_y_axis.values),
            np.array(dir_x_z_axis.values)
        ])

        y_axis_dir = np.column_stack([
            np.array(dir_y_x_axis.values),
            np.array(dir_y_y_axis.values),
            np.array(dir_y_z_axis.values)
        ])

        # Convert ECEF direction vectors to NED orientations
        # First, convert positions to cartographic for local frame computation
        local_origins = positions.to_cartographic()

        # Compute rotation matrices from NED to ECEF for each position
        size = local_origins.shape[0]
        clon, slon = np.cos(local_origins.longitude), np.sin(local_origins.longitude)
        clat, slat = np.cos(local_origins.latitude), np.sin(local_origins.latitude)
        rot_ecef2ned = Rotation.from_matrix(
            np.array([
                [-clon * slat,          -slon, -clon * clat],
                [-slon * slat,           clon, -slon * clat],
                [        clat, np.zeros(size),        -slat]
            ]).T
        )

        # Transform ECEF direction vectors to NED frame
        x_axis_ned = rot_ecef2ned.apply(x_axis_dir)
        y_axis_ned = rot_ecef2ned.apply(y_axis_dir)

        # Convert from ENU-like to NED convention (negate Y component)
        y_axis_ned[:, 1] = -y_axis_ned[:, 1]

        # Construct rotation matrices from the direction vectors
        # Assuming x_axis is forward (North in NED), y_axis is right (East in NED)
        z_axis_ned = np.cross(x_axis_ned, y_axis_ned)  # Down in NED

        # Normalize the vectors to ensure orthogonality
        x_axis_ned = x_axis_ned / np.linalg.norm(x_axis_ned, axis=1, keepdims=True)
        y_axis_ned = y_axis_ned / np.linalg.norm(y_axis_ned, axis=1, keepdims=True)
        z_axis_ned = z_axis_ned / np.linalg.norm(z_axis_ned, axis=1, keepdims=True)

        # Construct rotation matrices
        rotation_matrices = np.stack([x_axis_ned, y_axis_ned, z_axis_ned], axis=2)
        orientations = Rotation.from_matrix(rotation_matrices)

        return cls(timestamps, positions, orientations)

    @classmethod
    def read_pamela_pos(cls, filename):
        """
        Read a PAMELA .pos file and create a Trajectory instance.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The filename or path to the .pos file.

        Returns
        -------
        :class:`Trajectory`
            A new Trajectory instance.

        Raises
        ------
        :class:`FileNotFoundError`
            If the file does not exist.
        :class:`ValueError`
            If the file does not have a .pos extension.
        """
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"File {filename} does not exist.")
        if not filename.suffix == '.pos':
            raise ValueError("File must have a .pos extension.")

        record = np.loadtxt(filename, dtype=PAMELA_POS_DTYPE)
        n = record.shape[0]

        data = np.empty(n, dtype=TRAJ_DTYPE)
        data['TIMESTAMP_S'] = record['timestamp_s']
        data['LON_WGS84_DEG'] = record['longitude_deg']
        data['LAT_WGS84_DEG'] = record['latitude_deg']
        data['HEIGHT_WGS84_M'] = record['height_m']
        data['HEADING_DEG'] = record['heading_deg']
        data['ELEVATION_DEG'] = record['elevation_deg']
        data['BANK_DEG'] = record['bank_deg']

        return cls.from_numpy(data)

    @classmethod
    def read_pamela_traj(cls, filename, sampling_time_s=None, crs='auto'):
        """
        Read a PAMELA .traj file and create a Trajectory instance.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The filename or path to the .traj file.
        sampling_time_s : :class:`float`, optional
            If provided, overrides the time step between trajectory points (in seconds).
        crs : :class:`str`, optional
            Coordinate reference system of the trajectory. Options are: 'auto' (default, WGS84 if new format, NTF if old format), 'WGS84', 'NTF'.

        Returns
        -------
        :class:`Trajectory`
            A new Trajectory instance.

        Raises
        ------
        :class:`FileNotFoundError`
            If the file does not exist.
        :class:`ValueError`
            If the file does not have a .traj extension.
        :class:`ValueError`
            If the old PAMELA file format is detected.
        :class:`ValueError`
            If the crs parameter is not one of the accepted values.
        """
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"File {filename} does not exist.")
        if not filename.suffix == '.traj':
            raise ValueError("File must have a .traj extension.")
        if crs not in ['auto', 'WGS84', 'NTF']:
            raise ValueError("crs must be 'auto', 'WGS84', or 'NTF'.")

        # Read header (11 doubles, little endian)
        # Format:
        #    0: origin longitude (rad)
        #    1: origin latitude (rad)
        #    2: origin height (m)
        #    3: nominal horizontal velocity (m/s)
        #    4: nominal course (rad)
        #    5: nominal slope (rad)
        #    6: nominal leeway (rad) - course minus heading
        #    7: mean elevation (rad)
        #    8: mean bank (rad)
        #    9: std position (m)
        #   10: std velocity (m/s)
        header_count = 11
        header = np.fromfile(filename, dtype='<f8', count=header_count)

        # Read all records in a single operation
        header_size = header_count * 8  # bytes for 11 doubles
        records = np.fromfile(filename, dtype=PAMELA_TRAJ_DTYPE, offset=header_size)

        # Determine coordinate reference system
        if crs == 'auto':
            if header[10] > -0.5:
                crs = 'NTF'
                print("Guessed origin CRS is 'NTF'")
                print(
                    f"Old PAMELA file format detected (Custom NTF in local Lambert projection) [flag = {header[10]}]!\n"
                    " ↳ Trajectory will be automatically converted to geographic WGS84 CRS format."
                )
            else:
                print("Guessed origin CRS is 'WGS84'")
                crs = 'WGS84'

        # Convert trajectory to WGS84 if needed
        if crs == 'NTF':
            from sargeom.coordinates.transforms import LambertConicConformal
            from sargeom.coordinates.ellipsoids import Ellipsoid, ELPS_CLARKE_1880

            # Unpack NTF trajectory origin coordinates
            lon_origin_ntf_rad = header[0]
            lat_origin_ntf_rad = header[1]
            height_origin_ntf_m = header[2]

            # Unpack position records (note: attitude angles are kept unchanged)
            x_loc_m = records['longitude_rad']
            y_loc_m = records['latitude_rad']
            height_ntf_m = records['height_m']

            # Lambert Conic Conformal projection initialization
            locLambertNTF = LambertConicConformal(
                ELPS_CLARKE_1880,
                lon_origin_ntf_rad,
                lat_origin_ntf_rad
            )

            # Step 1: from local Lambert NTF to geographic NTF
            lon_ntf_rad, lat_ntf_rad = locLambertNTF.inverse(x_loc_m, y_loc_m)

            # Step 2: Transform from geographic NTF to cartesian ECEF NTF
            x_ntf_m, y_ntf_m, z_ntf_m = ELPS_CLARKE_1880.to_ecef(
                lon_ntf_rad, lat_ntf_rad, height_ntf_m + height_origin_ntf_m
            )

            # Custom transformation parameters cartesian ECEF NTF to cartesian ECEF WGS84
            _ANGLE_Z_NTF_TO_WGS84_RAD = np.deg2rad(0.554 / 3600.0)
            _DX_NTF_TO_WGS84_M = -168.0
            _DY_NTF_TO_WGS84_M = -72.0
            _DZ_NTF_TO_WGS84_M = 318.5

            # Step 3: Transform from cartesian ECEF NTF to cartesian ECEF WGS84 (custom)
            x_wgs84_m = 1.0000002198 * x_ntf_m - _ANGLE_Z_NTF_TO_WGS84_RAD * y_ntf_m + _DX_NTF_TO_WGS84_M
            y_wgs84_m = 1.0000002198 * y_ntf_m + _ANGLE_Z_NTF_TO_WGS84_RAD * x_ntf_m + _DY_NTF_TO_WGS84_M
            z_wgs84_m = 1.0000002198 * z_ntf_m                                       + _DZ_NTF_TO_WGS84_M

            # Step 4: Transform from cartesian ECEF WGS84 to geographic WGS84
            ELPS_PAM_WGS84 = Ellipsoid(semi_major_axis=6378137.0, semi_minor_axis=6356752.3142) # Custom PamelaX11 WGS84
            lon_wgs84_rad, lat_wgs84_rad, height_wgs84_m = ELPS_PAM_WGS84.to_cartographic(
                x_wgs84_m, y_wgs84_m, z_wgs84_m
            )

            # Update records geographic coordinates
            records['longitude_rad'] = lon_wgs84_rad
            records['latitude_rad'] = lat_wgs84_rad
            records['height_m'] = height_wgs84_m

        # Check trajectory time sampling
        if sampling_time_s is not None:
            time_step = sampling_time_s
        else:
            time_step = header[9]
            if time_step <= 0.0:
                time_step = 1.0
                print(
                    "Sampling time step is non-positive or non defined !\n"
                    " ↳ This value is set to 1.0 second by default."
                )
        print(f"Sampling time step is set to {time_step}s. To modify the timestamp axis set a new one in the newly created Trajectory object.")

        # Create output structured array
        n = records.shape[0]
        data = np.empty(n, dtype=TRAJ_DTYPE)
        data['TIMESTAMP_S'] = np.arange(n) * time_step
        data['LON_WGS84_DEG'] = np.degrees(records['longitude_rad'])
        data['LAT_WGS84_DEG'] = np.degrees(records['latitude_rad'])
        data['HEIGHT_WGS84_M'] = records['height_m']
        data['HEADING_DEG'] = np.degrees(records['heading_rad'])
        data['ELEVATION_DEG'] = np.degrees(records['elevation_rad'])
        data['BANK_DEG'] = np.degrees(records['bank_rad'])

        return cls.from_numpy(data)

    @classmethod
    def read_csv(cls, filename):
        """
        Read a TRAJ CSV file and create a Trajectory instance.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The filename or path to the .traj.csv file.

        Returns
        -------
        :class:`Trajectory`
            A new Trajectory instance.

        Raises
        ------
        :class:`FileNotFoundError`
            If the file does not exist.
        :class:`ValueError`
            If the file does not have a .traj.csv extension.
        """
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"File {filename} does not exist.")
        if not filename.name.endswith('.traj.csv'):
            raise ValueError("File must have a .traj.csv extension.")

        data = np.genfromtxt(
            filename,
            delimiter=';',
            comments='#',
            names=None,
            dtype=TRAJ_DTYPE,
            encoding='utf8'
        )[1:]  # Skip the header line

        return cls.from_numpy(data)

    def save_csv(self, filename):
        """
        Save the Trajectory instance to a TRAJ CSV file.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The filename or path to save the .traj.csv file.

        Returns
        -------
        :class:`pathlib.Path`
            The path to the saved .traj.csv file.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> filename = traj.save_csv("output")
        >>> print(filename)
        output.traj.csv
        """
        filename = Path(filename)
        filename = filename.with_name(filename.stem).with_suffix(".traj.csv")

        np.savetxt(
            filename,
            self.to_numpy(),
            fmt=['%.12f', '%.15f', '%.15f', '%.6f', '%.6f', '%.6f', '%.6f'],
            delimiter=';',
            newline='\n',
            comments='',
            encoding='utf8',
            header=f"""# {filename.name}
# Fields descriptions:
# -------------------
#    o Time representation
#        - TIMESTAMP_S [s]: Trajectory timestamp in seconds. May be UTC/GPS Seconds Of Week (SOW) or Time Of Day (TOD) or custom
#    o Positions as WGS84 Geographic coordinates:
#        - LON_WGS84_DEG [°]: The longitude coordinate
#        - LAT_WGS84_DEG [°]: The latitude coordinate
#        - HEIGHT_WGS84_M [m]: The ellipsoïdal height coordinate
#    o Attitudes as Euler angles (heading, elevation, bank) in the local North-East-Down (NED) cartesian frame related to geographic position coordinates:
#        - HEADING_DEG [°]: The heading angle in degrees
#        - ELEVATION_DEG [°]: The elevation angle in degrees
#        - BANK_DEG [°]: The bank angle in degrees

TIMESTAMP_S;LON_WGS84_DEG;LAT_WGS84_DEG;HEIGHT_WGS84_M;HEADING_DEG;ELEVATION_DEG;BANK_DEG"""
        )
        return filename

    def save_pamela_pos(self, filename):
        """
        Save the Trajectory instance to a PAMELA .pos file.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The filename or path to save the .pos file.

        Returns
        -------
        :class:`pathlib.Path`
            The path to the saved .pos file.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> filename = traj.save_pamela_pos("output")
        >>> print(filename)
        output.pos
        """
        filename = Path(filename).with_suffix(".pos")
        
        data = self.to_numpy()
        n = data.shape[0]

        pos = np.zeros(n, dtype=PAMELA_POS_DTYPE)
        pos['timestamp_s'] = data['TIMESTAMP_S']
        pos['latitude_deg'] = data['LAT_WGS84_DEG']
        pos['longitude_deg'] = data['LON_WGS84_DEG']
        pos['height_m'] = data['HEIGHT_WGS84_M']
        pos['bank_deg'] = data['BANK_DEG']
        pos['elevation_deg'] = data['ELEVATION_DEG']
        pos['heading_deg'] = data['HEADING_DEG']

        np.savetxt(
            filename,
            pos,
            fmt=['%.3f', '%.10f', '%.10f', '%.5f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f'],
            delimiter='\t',
            newline='\n',
            comments='',
            encoding='utf8'
        )

        return filename

    def save_npy(self, filename):
        """
        Save the Trajectory instance to a numpy .npy file.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The filename or path to save the .npy file.

        Returns
        -------
        :class:`pathlib.Path`
            The path to the saved .npy file.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> filename = traj.save_npy("output")
        >>> print(filename)
        output.npy
        """
        filename = Path(filename).with_suffix(".npy")
        np.save(filename, self.to_numpy())

        return filename

    def save_pivot(self, filename, actor_type='TX_ANTENNA', data_owner='NA', data_type='TRUEVALUE', protection_tag='NON_PROTEGE'):
        """
        Save the Trajectory instance to a PIVOT .h5 file.

        Parameters
        ----------
        filename : :class:`str` or :class:`pathlib.Path`
            The filename or path to save the .h5 file.
        actor_type : :class:`str`, optional
            The type of actor to save (default: 'TX_ANTENNA').
            May be one of: 'TX_PLATFORM', 'RX_PLATFORM', 'TX_ANTENNA', 'RX_ANTENNA', 'TARGET'.
        data_owner : :class:`str`, optional
            The data owner to use (default: 'NA').
        data_type : :class:`str`, optional
            The data type to use (default: 'TRUEVALUE').
            May be one of: 'TRUEVALUE', 'SETVALUE', 'ESTIMATEDVALUE'.
        protection_tag : :class:`str`, optional
            The protection tag to use (default: 'NON_PROTEGE').

        Returns
        -------
        :class:`pathlib.Path`
            The path to the saved .h5 file.

        Raises
        ------
        :class:`ImportError`
            If the pivot library is not installed.
        :class:`ValueError`
            If the actor_type, data_type or protection_tag is not valid.
        :class:`NotImplementedError`
            If the trajectory has no orientation.

        Examples
        --------
        >>> traj = Trajectory(
        ...     timestamps=[0, 1, 2, 3],
        ...     positions=Cartographic(
        ...         longitude=[3.8777, 4.8391, 5.4524, 6.2345],
        ...         latitude=[43.6135, 43.9422, 43.5309, 43.7891],
        ...         height=[300.0, 400.0, 500.0, 600.0]
        ...     )
        ... )
        >>> filename = traj.save_pivot("output")
        >>> print(filename)
        output.h5
        """
        try:
            from pivot.darpy import Axis, AxisLabelEnum
            from pivot.piactor import Actor, ActorTypeEnum
            from pivot.pivotutil import pivot_version, Metadata, ProtectionTag
            print(f"Using pivot library version: {pivot_version()}")
        except ImportError:
            raise ImportError("The pivot library is not installed. Please install it to write PIVOT files.")

        filename = Path(filename).with_suffix('.h5')

        if actor_type not in ['TX_PLATFORM', 'RX_PLATFORM', 'TX_ANTENNA', 'RX_ANTENNA', 'TARGET']:
            raise ValueError("'actor_type' must be one of: 'TX_PLATFORM', 'RX_PLATFORM', 'TX_ANTENNA', 'RX_ANTENNA', 'TARGET'")
        
        if protection_tag not in ProtectionTag.__members__:
            raise ValueError(f"'protection_tag' must be one of {list(ProtectionTag.__members__.keys())}")

        if data_type not in ['TRUEVALUE', 'SETVALUE', 'ESTIMATEDVALUE']:
            raise ValueError("data_type must be one of: 'TRUEVALUE', 'SETVALUE', 'ESTIMATEDVALUE'")

        # Compute direction vectors in ECEF frame
        local_origins = self._positions.to_cartographic()
        size = local_origins.shape[0]
        clon, slon = np.cos(local_origins.longitude), np.sin(local_origins.longitude)
        clat, slat = np.cos(local_origins.latitude), np.sin(local_origins.latitude)
        rot_ned2ecef = Rotation.from_matrix(
            np.array([
                [-clon * slat, -slon * slat,           clat],
                [       -slon,         clon, np.zeros(size)],
                [-clon * clat, -slon * clat,          -slat]
            ]).T
        )

        if self.has_orientation():
            # Carrier "BODY" frame is in NED coordinates
            # x_axis_dir = (rot_ned2ecef * self._orientations).apply([1.0, 0.0, 0.0])  # X-axis direction (NED)
            # y_axis_dir = (rot_ned2ecef * self._orientations).apply([0.0, 1.0, 0.0])  # Y-axis direction (NED)
            x_axis_dir = (rot_ned2ecef * self._orientations).apply([1.0, 0.0, 0.0])  # X-axis direction (ENU)
            y_axis_dir = (rot_ned2ecef * self._orientations).apply([0.0, -1.0, 0.0])  # Y-axis direction (ENU)
        else:
            # TODO: if no orientation, compute the direction vector from the velocity vector
            raise NotImplementedError("Saving to PIVOT format without orientation is not implemented yet.")

        states = [
            Axis(AxisLabelEnum.TIME, self._timestamps.tolist()),
            Axis(AxisLabelEnum.POS_X_ECEF, self._positions.x.tolist()),
            Axis(AxisLabelEnum.POS_Y_ECEF, self._positions.y.tolist()),
            Axis(AxisLabelEnum.POS_Z_ECEF, self._positions.z.tolist()),
            Axis(AxisLabelEnum.DIR_x_X_ECEF, x_axis_dir[:, 0].tolist()),
            Axis(AxisLabelEnum.DIR_x_Y_ECEF, x_axis_dir[:, 1].tolist()),
            Axis(AxisLabelEnum.DIR_x_Z_ECEF, x_axis_dir[:, 2].tolist()),
            Axis(AxisLabelEnum.DIR_y_X_ECEF, y_axis_dir[:, 0].tolist()),
            Axis(AxisLabelEnum.DIR_y_Y_ECEF, y_axis_dir[:, 1].tolist()),
            Axis(AxisLabelEnum.DIR_y_Z_ECEF, y_axis_dir[:, 2].tolist())
        ]

        actor_dname = re.sub(r'[^a-zA-Z0-9]', '-', filename.stem)
        actor_downer = re.sub(r'[^a-zA-Z0-9]', '-', data_owner)

        tx_actor = Actor(
            ActorTypeEnum[actor_type],
            f"{actor_dname}_{actor_downer}_{data_type}_1",  # FIXME: remove _1 suffix when SCALIAN fixes Actor name bug
            states
        )

        meta = Metadata({ 'Rights': { 'dataOwner': data_owner, 'dataCoowner': 'NA', 'confid': ProtectionTag[protection_tag] } })

        tx_actor.save(filename, mode='override')
        meta.save(filename)

        return filename

    def save_kml(self, filename, **kwargs):
        # TODO: Implement saving to KML format
        raise NotImplementedError("Saving to KML format is not implemented yet.")