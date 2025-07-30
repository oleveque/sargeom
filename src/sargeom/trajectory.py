from coordinates.cartesian import CartesianECEF, Cartographic
from scipy.spatial.transform import Rotation, Slerp
from pathlib import Path
import numpy as np

class Trajectory:
    def __init__(self, timestamp, x, y, z, heading_angle=None, elevation_angle=None, bank_angle=None, degrees=True):
        self.timestamp = timestamp
        self.positions = CartesianECEF(x, y, z)
        if heading_angle is None or elevation_angle is None or bank_angle is None:
            self.orientations = None
        else:
            self.orientations = Rotation.from_euler(
                "ZYX",  # Intrinsic rotations
                [heading_angle, elevation_angle, bank_angle],
                degrees,
            )

    @classmethod
    def from_cartographic(cls, timestamp, latitude, longitude, altitude, heading_angle, elevation_angle, bank_angle, degrees=True):
        pos = Cartographic(latitude, longitude, altitude, degrees).to_ecef()
        return cls(timestamp, pos.x, pos.y, pos.z, heading_angle, elevation_angle, bank_angle, degrees)

    def interp(self, timestamp):
        pos = self.positions.interp(self.timestamp, timestamp)
        if self.orientations is None:
            return Trajectory(timestamp, pos.x, pos.y, pos.z)
        else:
            serl = Slerp(self.timestamp, self.orientations)
            angles = serl(timestamp).as_euler("ZYX", degrees=True)
            return Trajectory(timestamp, pos.x, pos.y, pos.z, *angles, degrees=True)
        
    def to_pandas(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Pandas is not installed. Please follow the instructions on https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html"
            )
        
        pos = self.positions.to_cartographic()
        if self.orientations is None:
            return pd.DataFrame({
                "TIMESTAMP_S": self.timestamp,
                "LON_WGS84_DEG": pos.longitude,
                "LAT_WGS84_DEG": pos.latitude,
                "HEIGHT_WGS84_M": pos.height
            })
        else:
            angles = self.orientations.as_euler("ZYX", degrees=True)
            return pd.DataFrame({
                "TIMESTAMP_S": self.timestamp,
                "LON_WGS84_DEG": pos.longitude,
                "LAT_WGS84_DEG": pos.latitude,
                "HEIGHT_WGS84_M": pos.height,
                "HEADING_DEG": angles[0],
                "ELEVATION_DEG": angles[1],
                "BANK_DEG": angles[2]
            })

    def save_csv(self, filename):
        filename = Path(filename)

        pos = self.positions.to_cartographic()
        if self.orientations is None:
            np.savetxt(
                filename.with_suffix(".traj.csv"),
                [self.timestamp, pos.longitude, pos.latitude, pos.height],
                fmt=['%.3f', '%.12f', '%.12f', '%.6f'],
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

TIMESTAMP_S;LON_WGS84_DEG;LAT_WGS84_DEG;HEIGHT_WGS84_M"""
            )
        else:
            angles = self.orientations.as_euler("ZYX", degrees=True)

            np.savetxt(
                filename.with_suffix(".traj.csv"),
                [self.timestamp, pos.longitude, pos.latitude, pos.height, *angles],
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

    def save_pivot(self, filename):
        try:
            from pivot.darpy import Axis, AxisLabelEnum
            from pivot.piactor import Actor, ActorTypeEnum
            from pivot.pivotutil import pivot_version, Metadata, ProtectionTag
            print("Pivot version:", pivot_version())
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Pivot modules are not installed. Please follow the instructions on http://125.40.2.23:3000/PIVOT/-/packages/pypi/pivot/2.3.0"
            )
        
        filename = Path(filename)

        # TODO: Implement pivot saving functionality
        raise NotImplementedError(
            "Pivot saving is not implemented yet. Please use save_csv instead."
        )