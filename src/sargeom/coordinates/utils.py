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
    >>> negativePiToPi(190.0)
    -170.0

    >>> negativePiToPi([-190.0, 190.0])
    array([ 170., -170.])
    """
    angle = np.atleast_1d(angle).copy()
    if degrees:
        mask = (angle <= -180.0) | (angle >= 180.0)
        if mask.any():
            angle[mask] = (angle[mask] + 180.0) % 360.0 - 180.0
    else:
        mask = (angle <= -np.pi) | (angle >= np.pi)
        if mask.any():
            angle[mask] = (angle[mask] + np.pi) % (2 * np.pi) - np.pi
    # Format as Python `float` if it constains only one value
    if angle.size == 1:
        angle = angle.item()
    return angle

if __name__ == "__main__":
    import doctest
    doctest.testmod()
