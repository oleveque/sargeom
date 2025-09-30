import numpy as np


class Ellipsoid:
    """
    Represents a reference ellipsoid of revolution for geodetic calculations.

    The ellipsoid is defined by either:
    - Equatorial radius (semi-major axis) and polar radius (semi-minor axis), or
    - Equatorial radius (semi-major axis) and flattening factor (f).

    Parameters
    ----------
    semi_major_axis : :class:`float`
        Equatorial radius of the ellipsoid in meters.
    semi_minor_axis : :class:`float`, optional
        Polar radius of the ellipsoid in meters. Default is None.
    flattening : :class:`float`, optional
        Flattening factor of the ellipsoid. Default is None.
    """
    def __init__(self, semi_major_axis, semi_minor_axis=None, flattening=None):
        self._a, self._b, self._f = semi_major_axis, semi_minor_axis, flattening
        if self._b is None:
            if self._f is None:
                raise ValueError(
                    "Ellipsoid must be defined from its equatorial radius (a) in meters and"
                    " flattening factor (f) or polar radius (b) in meters."
                )
            else:
                self._b = (1 - self._f) * self._a
        else:
            self._f = (self._a - self._b) / self._a

        # Other used parameters defining the ellipsoid
        self._n = self._f / (2 - self._f)  # Third flattening factor
        self._e2 = self._f * (2 - self._f)  # Eccentricity squared
        self._e = np.sqrt(self._e2)  # Eccentricity

        n = [self._n**i for i in range(11)] # n^0 ... n^10
        # Coefficients d(n) of the series expansion of the inverse conformal latitude
        self._phichi = (
            # d(2)
            (2*n[1] - 2*n[2]/3 - 2*n[3] + 116*n[4]/45 + 26*n[5]/45 - 2854*n[6]/675
            + 16822*n[7]/4725 + 189416*n[8]/99225 - 1113026*n[9]/165375
            + 22150106*n[10]/4465125),
            # d(4)
            (7*n[2]/3 - 8*n[3]/5 - 227*n[4]/45 + 2704*n[5]/315 + 2323*n[6]/945
            - 31256*n[7]/1575 + 141514*n[8]/8505 + 10453448*n[9]/606375
            - 66355687*n[10]/1403325),
            # d(6)
            (56*n[3]/15 - 136*n[4]/35 - 1262*n[5]/105 + 73814*n[6]/2835
            + 98738*n[7]/14175 - 2363828*n[8]/31185 + 53146406*n[9]/779625
            + 1674405706*n[10]/18243225),
            # d(8)
            (4279*n[4]/630 - 332*n[5]/35 - 399572*n[6]/14175 + 11763988*n[7]/155925
            + 14416399*n[8]/935550 - 2647902052*n[9]/10135125
            + 23834033824*n[10]/91216125),
            # d(10)
            (4174*n[5]/315 - 144838*n[6]/6237 - 2046082*n[7]/31185
            + 258316372*n[8]/1216215 + 67926842*n[9]/2837835
            - 76998787574*n[10]/91216125),
            # d(12)
            (601676*n[6]/22275 - 115444544*n[7]/2027025 - 2155215124*n[8]/14189175
            + 41561762048*n[9]/70945875 + 625821359*n[10]/638512875),
            # d(14)
            (38341552*n[7]/675675 - 170079376*n[8]/1216215
            - 1182085822*n[9]/3378375 + 493459023622*n[10]/310134825),
            # d(16)
            (1383243703*n[8]/11351340 - 138163416988*n[9]/402026625
            - 1740830660174*n[10]/2170943775),
            # d(18)
            106974149462*n[9]/402026625 - 24899113566814*n[10]/29462808375,
            # d(20)
            175201343549*n[10]/297604125
        )

    def prime_vertical_curvature_radius(self, phi):
        """
        Computes the prime vertical curvature radius of a point at latitude φ.

        The prime vertical curvature radius is computed using the formula:

        .. math:: \nu(\phi)=\dfrac{a}{\sqrt{1-e^2\sin^2\phi}}

        where :math:`a` is the semi-major axis and :math:`e` is the eccentricity
        of the ellipsoid.

        Parameters
        ----------
        phi : array_like
            The geodetic latitude in radians.

        Returns
        -------
        nu : :class:`numpy.ndarray`
            Prime vertical curvature radius in meters.

        Notes
        -----
        The prime vertical curvature radius is the radius of curvature in the 
        plane of the prime vertical, which is perpendicular to the meridian 
        and contains the normal to the ellipsoid.
        """
        nu = self._a / np.sqrt(1 - self._e2 * np.sin(phi)**2)
        return nu

    def isometric_latitude(self, phi):
        """
        Computes the isometric latitude (parameter of the Mercator projection).

        The isometric latitude is computed using the formula:

        .. math::
            \psi(\phi)=\mathrm{arctanh}\big(\sin(\phi)\big)-
            e\mathrm{arctanh}\big(e\sin(\phi)\big)

        It can also be computed from the conformal latitude:

        .. math::
            \psi(\phi)=\mathrm{arctanh}\big[\sin\big(\chi(\phi)\big)\big]

        Parameters
        ----------
        phi : array_like
            The geodetic latitude in radians.
 
        Returns
        -------
        psi : :class:`numpy.ndarray`
            The isometric latitude in radians.

        Notes
        -----
        The isometric latitude is used as a parameter in the Mercator projection
        and other conformal map projections. It represents the latitude on an
        auxiliary sphere that preserves angles.

        See Also
        --------
        inverse_isometric_latitude : Inverse function
        conformal_latitude : Related conformal latitude function
        """
        sphi = np.sin(phi)
        psi = np.arctanh(sphi) - self._e * np.arctanh(self._e * sphi)
        return psi

    def inverse_isometric_latitude(self, psi):
        """
        Computes the inverse isometric latitude.

        The calculation is performed from the inverse function of the
        conformal latitude:

        .. math::
            \phi(\psi)=\chi^{-1}\big[\arcsin\big(\tanh(\psi)\big)\big]

        Parameters
        ----------
        psi : array_like
            The isometric latitude in radians.

        Returns
        -------
        phi : :class:`numpy.ndarray`
            The geodetic latitude in radians.

        Notes
        -----
        This function computes the geodetic latitude from the isometric latitude
        by first converting to conformal latitude and then using the inverse
        conformal latitude function.

        See Also
        --------
        isometric_latitude, inverse_conformal_latitude
        """
        chi = np.arcsin(np.tanh(psi))
        phi = self.inverse_conformal_latitude(chi)
        return phi

    def conformal_latitude(self, phi):
        """
        Computes the conformal latitude from the geodetic latitude.

        The conformal latitude is computed using the formula:

        .. math::
            \chi(\phi)=\arcsin\big[\tanh\big(\mathrm{arctanh}(\sin(\phi))-
            e\mathrm{arctanh}(e\sin(\phi))\big)\big]

        Parameters
        ----------
        phi : array_like
            The geodetic latitude in radians.
            
        Returns
        -------
        chi : :class:`numpy.ndarray`
            The conformal latitude in radians.

        Notes
        -----
        The conformal latitude preserves angles during the projection
        of the ellipsoid onto the auxiliary sphere. It is used in conformal
        map projections such as the Mercator and Lambert Conformal Conic
        projections.

        See Also
        --------
        inverse_conformal_latitude
        """
        sphi = np.sin(phi)
        chi = np.arcsin(np.tanh(
            np.arctanh(sphi) - self._e * np.arctanh(self._e * sphi)
        ))
        return chi

    def inverse_conformal_latitude(self, chi):
        """
        Computes the inverse conformal latitude.

        The calculation is performed from a harmonic expansion up to
        order 10 in terms of the third flattening factor :math:`n`
        using the Lagrange inversion formula:

        .. math::
            \phi(\chi)\simeq\chi+\sum\limits_{p=1}^{10}d_{2p}(n)\sin(2p\chi)

        Parameters
        ----------
        chi : array_like
            The conformal latitude in radians.

        Returns
        -------
        phi : :class:`numpy.ndarray`
            The geodetic latitude in radians.

        Notes
        -----
        This implementation uses a series expansion up to the 10th order
        for high accuracy. The coefficients d(n) are precomputed during
        ellipsoid initialization.

        See Also
        --------
        conformal_latitude
        """
        phi = np.copy(chi)
        for n, dn in enumerate(self._phichi):
            phi += dn * np.sin(2 * (n + 1) * chi)
        return phi
    
    def to_ecef(self, lamb, phi, height_m=0):
        """
        Converts geodetic coordinates to geocentric cartesian ECEF coordinates.

        Converts geodetic coordinates :math:`(\lambda,\phi, H)` of a
        point to geocentric cartesian ECEF coordinates :math:`(X,Y,Z)`.

        The conversion is made through the relationships:

        .. math::
            \left\{\begin{array}{rcl}
            X & = & \big(\nu(\phi) + H\big)\cos(\phi)\cos(\lambda) \\
            Y & = & \big(\nu(\phi) + H\big)\cos(\phi)\sin(\lambda) \\
            Z & = & \big((1-e^2)\nu(\phi) + H\big)\sin(\phi)
            \end{array}\right.

        where :math:`\nu(\phi)` is the prime vertical curvature radius
        of the point with latitude :math:`\phi`.
        
        Parameters
        ----------
        lamb : array_like
            The geographic longitude of the point in radians.
        phi : array_like
            The geographic latitude of the point in radians.
        height_m : array_like, optional
            The geodetic height of the point in meters. Default is 0.
        
        Returns
        -------
        X : :class:`numpy.ndarray`
            The X cartesian ECEF coordinate in meters.
        Y : :class:`numpy.ndarray`
            The Y cartesian ECEF coordinate in meters.
        Z : :class:`numpy.ndarray`
            The Z cartesian ECEF coordinate in meters.

        See Also
        --------
        prime_vertical_curvature_radius, to_cartographic
        """
        nu = self.prime_vertical_curvature_radius(phi) # At the ellipsoid level
        nuhcosphi = (nu + height_m) * np.cos(phi)
        X = nuhcosphi * np.cos(lamb)
        Y = nuhcosphi * np.sin(lamb)
        Z = ((1 - self._e2) * nu + height_m) * np.sin(phi)
        return X, Y, Z
    
    def to_cartographic(self, X, Y, Z):
        """
        Converts geocentric cartesian ECEF coordinates to geodetic coordinates.

        Converts geocentric cartesian ECEF coordinates :math:`(X,Y,Z)`
        of a point to geodetic coordinates :math:`(\lambda,\phi, H)`.

        Parameters
        ----------
        X : array_like
            The X cartesian ECEF coordinate in meters.
        Y : array_like
            The Y cartesian ECEF coordinate in meters.
        Z : array_like
            The Z cartesian ECEF coordinate in meters.

        Returns
        -------
        lamb : :class:`numpy.ndarray`
            The geographic longitude in radians.
        phi : :class:`numpy.ndarray`
            The geographic latitude in radians.
        height_m : :class:`numpy.ndarray`
            The geodetic height in meters.

        Notes
        -----
        Conversion is made using the algorithm proposed by Vermeille (2002) but
        with limited application to the case where the evolute sign test is not performed,
        that is this algorithm is valid for points not "too deep" towards the center of
        the Earth (height > -6314 km for WGS84 ellipsoid), which is the case for our applications.

        References
        ----------
        .. [1] Vermeille, H. Direct transformation from geocentric coordinates to geodetic coordinates.
               Journal of Geodesy 76, 451–454 (2002). https://doi.org/10.1007/s00190-002-0273-6

        See Also
        --------
        to_ecef, prime_vertical_curvature_radius
        """
        R = np.hypot(X, Y)
        p = R**2 / self._a**2
        q = (1 - self._e2) * Z**2 / self._a**2
        r = (p + q - self._e2**2) / 6
        cbrt = np.cbrt(np.sqrt(8 * r**3 + self._e2**2 * p * q) + np.sqrt(p * q) * self._e2)
        cbrt *= cbrt
        u = r + 0.5 * cbrt + 2.0 * r * r / cbrt
        v = np.sqrt(u**2 + self._e2**2 * q)
        w = 0.5 * self._e2 * (u + v - q) / v
        k = (u + v) / (np.sqrt(w**2 + u + v) + w)
        D = k * R / (k + self._e2)
        hypotDZ = np.hypot(D, Z)

        # Calculation of h and phi
        h = (k + self._e2 - 1) * hypotDZ / k
        phi = 2 * np.arctan(Z / (hypotDZ + D))

        # Calculation of lambda
        lamb = np.arctan2(Y, X)

        return lamb, phi, h

# Official ellipsoid definitions
ELPS_WGS84 = Ellipsoid(semi_major_axis=6378137.0, flattening=1/298.257223563) # WGS 84
ELPS_CLARKE_1880 = Ellipsoid(semi_major_axis=6378249.2, semi_minor_axis=6356515.0) # Clarke 1880 (IGN)


if __name__ == "__main__":
    import doctest
    doctest.testmod()