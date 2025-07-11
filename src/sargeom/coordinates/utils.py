import numpy as np


def negativePiToPi(angle, degrees=True):
    """
    Converts angles to the range from -180 to 180 degrees.

    Parameters
    ----------
    angle : :class:`float` or array_like
        The input angle or a list/array of angles.
    degrees : bool, optional
        If True (default), takes input angles in degrees and returns the angle in degrees. If False, takes input radians and returns the angle in radians.

    Returns
    -------
    :class:`float` or :class:`numpy.ndarray`
        The converted angle or an array of converted angles (in degrees or radians).

    Examples
    --------
    >>> negativePiToPi(190)
    -170.0

    >>> negativePiToPi([-190, 190])
    array([ 170., -170.])
    """
    # Transforms arrays and scalars of type ndarray into lists or integers
    if isinstance(angle, np.ndarray):
        angle = angle.tolist()

    if isinstance(angle, list) or isinstance(angle, tuple):
        return np.array([negativePiToPi(a, degrees) for a in angle])
    else:
        if degrees:
            if -180.0 <= angle <= 180.0:
                return angle
            else:
                return (angle + 180.0) % 360.0 - 180.0
        else:
            if -np.pi <= angle <= np.pi:
                return angle
            else:
                return (angle + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    import doctest
    doctest.testmod()
