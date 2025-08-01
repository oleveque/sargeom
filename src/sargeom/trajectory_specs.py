# https://gereon-t.github.io/trajectopy/Documentation/Trajectory/
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from coordinates.cartesian import Cartesian3, CartesianECEF, Cartographic

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

class Trajectory:
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

    def __len__(self):
        return len(self._timestamps)

    def __getitem__(self, item):
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
        return self._timestamps

    @property
    def positions(self):
        return self._positions

    @property
    def velocities(self):
        if len(self._timestamps) < 2:
            raise ValueError("Not enough timestamps to compute velocities.")
        dt = np.diff(self._timestamps)
        return self.arc_lengths() / dt

    def has_orientation(self):
        return self._orientations is not None

    @property
    def orientations(self):
        if self._orientations is None:
            raise ValueError("This trajectory does not have orientations.")
        return self._orientations

    @property
    def arc_lengths(self):
        return Cartesian3.distance(self._positions[1:], self._positions[:-1])

    def total_arc_length(self):
        return np.sum(self.arc_lengths)

    @property
    def sampling_rate(self):
        if len(self._timestamps) < 2:
            raise ValueError("Not enough timestamps to compute sampling rate.")
        dt = np.diff(self._timestamps)
        return 1 / np.mean(dt)

    def resample(self, sampling_rate):
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
        # https://github.com/gereon-t/trajectopy/blob/main/trajectopy/core/plotting/mpl/trajectory.py
        pass

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
                np.nan(len(self._timestamps)),  # HEADING_DEG
                np.nan(len(self._timestamps)),  # ELEVATION_DEG
                np.nan(len(self._timestamps))   # BANK_DEG
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
        pass

    def read_pamela_traj(self, filename):
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"File {filename} does not exist.")
        if not filename.suffix == '.traj':
            raise ValueError("File must have a .traj extension.")

        # Read header (11 doubles, little endian)
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
        pass

    def save_pamela_traj(self, filename):
        pass

    def save_pivot(self, filename):
        # TODO: Implement saving to a pivot file
        raise NotImplementedError("Saving to pivot files is not implemented yet.")

    def save_kml(self, filename, **kwargs):
        pass