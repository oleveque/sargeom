# https://gereon-t.github.io/trajectopy/Documentation/Trajectory/
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
    ('latitude_deg', '<f8'),
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
    ('std_height_m', '<f4')
]

class Trajectory:
    """
    A Trajectory object represents a sequence of positions and orientations over time.

    It is defined by the following characteristics:

    - Timestamps are expressed in seconds. They may correspond to UTC, GPS Seconds of Week (SOW), Time of Day (TOD), or a custom time reference.
    - Positions are provided in either the WGS84 geographic coordinate system (EPSG:4979) or the WGS84 geocentric coordinate system (EPSG:4978).
    - Orientations are defined in the local North-East-Down (NED) Cartesian frame, relative to the associated position coordinates.

    Parameters
    ----------
    timestamps : :class:`numpy.ndarray`
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

    >>> timestamps = np.array([0, 1, 2])
    >>> positions = CartesianECEF(x=[0, 0, 0], y=[1, 1, 1], z=[2, 2, 2])
    >>> trajectory = Trajectory(timestamps, positions)

    Create a Trajectory instance using geographic (Cartographic) coordinates:

    >>> positions = Cartographic(longitude=[10, 20, 30], latitude=[40, 50, 60], height=[70, 80, 90])
    >>> trajectory = Trajectory(timestamps, positions)
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
        >>> len(trajectory)
        3
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
        >>> trajectory[0]
        Trajectory(timestamps=[0], positions=[[0, 0, 0]])

        >>> trajectory[:2]
        Trajectory(timestamps=[0, 1], positions=[[0, 0, 0], [1, 1, 1]])
        """
        return Trajectory(
            timestamps=self._timestamps[item],
            positions=self._positions[item],
            orientations=self._orientations[item] if self._orientations is not None else None
        )

    def __repr__(self):
        return f"Trajectory(timestamps={self._timestamps}, positions={self._positions}, orientations={self._orientations})"

    def sort(self, inplace=True, reverse=False):
        indices = np.argsort(self._timestamps)
        if reverse:
            indices = indices[::-1]
        if inplace:
            self._timestamps = self._timestamps[indices]
            self._positions = self._positions[indices]
            if self._orientations is not None:
                self._orientations = self._orientations[indices]
            return self
        else:
            return Trajectory(
                timestamps=self._timestamps[indices],
                positions=self._positions[indices],
                orientations=self._orientations[indices] if self._orientations is not None else None
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
        >>> trajectory.timestamps
        array([0, 1, 2])
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
        >>> trajectory.positions
        CartesianECEF([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
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
        >>> trajectory.velocities
        array([1.41421356, 1.41421356])
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
        >>> trajectory.has_orientation()
        True
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

        Examples
        --------
        >>> trajectory.orientations
        <scipy.spatial.transform._rotation.Rotation object at 0x...>
        """
        if self._orientations is None:
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
        >>> trajectory.arc_lengths
        array([1.73205081, 1.73205081])
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
        >>> trajectory.total_arc_length()
        3.46410162
        """
        return np.sum(self.arc_lengths)

    @property
    def sampling_rate(self):
        """
        Compute the sampling rate of the trajectory.

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
        >>> trajectory.sampling_rate
        1.0
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
        >>> resampled_trajectory = trajectory.resample(2.0)
        """
        if not isinstance(sampling_rate, (int, float)):
            raise TypeError("Sampling rate must be a number.")
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive.")

        new_timestamps = np.arange(self._timestamps[0], self._timestamps[-1], 1 / sampling_rate)
        new_positions = self._positions.interp(self._timestamps, new_timestamps)
        
        if self._orientations is None:
            return Trajectory(new_timestamps, new_positions)
        else:
            serl = Slerp(self._timestamps, self._orientations)
            return Trajectory(new_timestamps, new_positions, serl(new_timestamps))

    def interp(self, new_timestamps):
        if not isinstance(new_timestamps, np.ndarray):
            new_timestamps = np.asarray(new_timestamps)
        if len(new_timestamps) == 0:
            raise ValueError("Timestamps array cannot be empty.")
        if new_timestamps[0] < self._timestamps[0] or new_timestamps[-1] > self._timestamps[-1]:
            raise ValueError("New timestamps must be within the range of existing timestamps.")

        new_positions = self.positions.interp(self._timestamps, new_timestamps)
        if self.orientations is None:
            return Trajectory(new_timestamps, new_positions)
        else:
            serl = Slerp(self._timestamps, self.orientations)
            return Trajectory(new_timestamps, new_positions, serl(new_timestamps))

    def plot(self, **kwargs):
        # TODO: Implement plotting functionality
        # See: https://github.com/gereon-t/trajectopy/blob/main/trajectopy/core/plotting/mpl/trajectory.py
        raise NotImplementedError("Plotting functionality is not implemented yet.")

    def from_numpy(self, data):
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
                [data['HEADING_DEG'], data['ELEVATION_DEG'], data['BANK_DEG']],
                degrees=True
            )
        else:
            orientations = None
        
        return Trajectory(timestamps, positions, orientations)

    def to_numpy(self):
        cartographic_positions = self._positions.to_cartographic()
        if self._orientations is None:
            return np.array(list(zip(
                self._timestamps,
                cartographic_positions.longitude,
                cartographic_positions.latitude,
                cartographic_positions.height,
                np.empty(self.__len__()),  # HEADING_DEG
                np.empty(self.__len__()),  # ELEVATION_DEG
                np.empty(self.__len__())   # BANK_DEG
            )), dtype=TRAJ_DTYPE)
        else:
            [heading, elevation, bank] = self._orientations.as_euler("ZYX", degrees=True)
            return np.array(list(zip(
                self._timestamps,
                cartographic_positions.longitude,
                cartographic_positions.latitude,
                cartographic_positions.height,
                heading,
                elevation,
                bank
            )), dtype=TRAJ_DTYPE)

    def to_pandas(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Pandas is not installed. Please follow the instructions on https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html"
            )

        data = self.to_numpy()
        return pd.DataFrame(data)

    def read_pivot(self, filename):
        # TODO: Implement reading from a pivot file
        raise NotImplementedError("Reading from pivot files is not implemented yet.")

    def read_pamela_pos(self, filename):
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

        return Trajectory.from_numpy(data)

    def read_pamela_traj(self, filename):
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"File {filename} does not exist.")
        if not filename.suffix == '.traj':
            raise ValueError("File must have a .traj extension.")

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

        # Check format
        if header[10] > -0.5:
            raise ValueError("Old PAMELA file format detected, please check the file format!")

        time_step = header[9]
        header_size = header_count * 8  # bytes for 11 doubles

        # Read all records in a single operation
        records = np.fromfile(filename, dtype=PAMELA_TRAJ_DTYPE, offset=header_size)
        n = records.shape[0]

        # Create output structured array
        data = np.empty(n, dtype=TRAJ_DTYPE)
        data['TIMESTAMP_S'] = (np.arange(n) + 1) * time_step
        data['LON_WGS84_DEG'] = np.degrees(records['longitude_rad'])
        data['LAT_WGS84_DEG'] = np.degrees(records['latitude_rad'])
        data['HEIGHT_WGS84_M'] = records['height_m']
        data['HEADING_DEG'] = np.degrees(records['heading_rad'])
        data['ELEVATION_DEG'] = np.degrees(records['elevation_rad'])
        data['BANK_DEG'] = np.degrees(records['bank_rad'])

        return Trajectory.from_numpy(data)

    def read_csv(self, filename):
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"File {filename} does not exist.")
        if not filename.suffix == '.traj.csv':
            raise ValueError("File must have a .traj.csv extension.")

        data = np.genfromtxt(
            filename,
            delimiter=';',
            comments='#',
            names=None,
            dtype=TRAJ_DTYPE,
            encoding='utf8'
        )

        return Trajectory.from_numpy(data)

    def save_csv(self, filename):
        filename = Path(filename)
        data = self.to_numpy()
        np.savetxt(
            filename.with_suffix(".traj.csv"),
            data,
            fmt=['%.3f', '%.12f', '%.12f', '%.6f', '%.6f', '%.6f', '%.6f'],
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

    def save_pamela_pos(self, filename):
        filename = Path(filename)
        
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
            filename.with_suffix(".pos"),
            pos,
            fmt=['%.3f', '%.10f', '%.10f', '%.5f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f'],
            delimiter='\t',
            newline='\n',
            comments='',
            encoding='utf8'
        )

    def save_npy(self, filename):
        filename = Path(filename)
        data = self.to_numpy()
        np.save(filename.with_suffix('.npy'), data)

    def save_pivot(self, filename):
        # TODO: Implement saving to a pivot file
        raise NotImplementedError("Saving to pivot files is not implemented yet.")

    def save_kml(self, filename, **kwargs):
        # TODO: Implement saving to KML format
        raise NotImplementedError("Saving to KML format is not implemented yet.")