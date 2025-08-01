# https://gereon-t.github.io/trajectopy/Documentation/Trajectory/

class Trajectory:

    def __init__(self, timestamps, positions, orientations=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def __repr__(self):
        pass

    def sort(self, key="temporal", reverse=False):
        if key == "temporal":
            pass
        elif key == "spatial":
            pass
        else:
            raise ValueError("Invalid sort key. Use 'temporal' or 'spatial'.")
        return self

    @property
    def timestamps(self):
        pass

    @property
    def positions(self):
        pass

    @property
    def velocities(self):
        pass

    def has_orientation(self):
        pass

    @property
    def orientations(self):
        pass

    @property
    def arc_lengths(self):
        pass

    def total_arc_length(self):
        pass

    @property
    def sampling_rate(self):
        pass

    def resample(self, sampling_rate, method='linear'):
        pass

    def interpolate(self, timestamps, method='linear'):
        pass

    def plot(self, **kwargs):
        pass

    def to_numpy(self): # alias: to_array()
        pass

    def to_pandas(self, **kwargs):
        pass

    def from_pivot(self, filename):
        pass

    def from_pos_pamela(self, filename):
        pass

    def from_traj_pamela(self, filename):
        pass

    def from_csv(self, filename):
        pass

    def save_csv(self, filename=None):
        pass

    def save_pos_pamela(self, filename=None):
        pass

    def save_traj_pamela(self, filename=None):
        pass

    def save_pivot(self, filename=None):
        pass

    def save_kml(self, filename=None, **kwargs):
        pass