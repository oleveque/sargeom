import numpy as np


def negativePiToPi(angle, degrees=True):
    """
    Normalize angles to the range [-180, 180] degrees or [-π, π] radians.

    This function wraps angles that fall outside the standard range back into
    the range [-180, 180] degrees (or [-π, π] radians when ``degrees=False``).

    Parameters
    ----------
    angle : :class:`float` or array_like
        The input angle or a list/array of angles to normalize.
    degrees : bool, optional
        If ``True`` (default), input and output angles are in degrees.
        If ``False``, input and output angles are in radians.

    Returns
    -------
    :class:`float` or :class:`numpy.ndarray`
        The normalized angle(s) in the range [-180, 180] degrees or [-π, π] radians.
        Returns a scalar if the input is a scalar, otherwise returns an array.

    Examples
    --------
    Normalize a single angle in degrees:

    >>> negativePiToPi(190.0)
    -170.0

    Normalize multiple angles:

    >>> negativePiToPi([-190.0, 190.0])
    array([ 170., -170.])

    Normalize angles in radians:

    >>> import numpy as np
    >>> negativePiToPi(4.0, degrees=False)  # doctest: +ELLIPSIS
    -2.283...
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
