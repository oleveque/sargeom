# import pyproj
import numpy as np

# # Coordinate systems
# wgs84_EGM96 = pyproj.crs.CRS.from_epsg(
#     "4326+5773" # EPSG:9707 = WGS84 Geographic 2D coordinate system (GCS) + EGM96 height (= Gravity-related height)
# )
# wgs84_ECEF = pyproj.crs.CRS.from_epsg(
#     "4978" # EPSG:4978 = WGS84 Geocentric 3D coordinate system (ECEF = Earth-centered, Earth-fixed coordinate system)
# )
# wgs84_GCS = pyproj.crs.CRS.from_epsg(
#     "4979" # EPSG:4979 = WGS84 Geographic 3D coordinate system (GCS)
# )

# # Coordinate transformations
# ecef2gcs = pyproj.Transformer.from_crs(
#     crs_from=wgs84_ECEF, crs_to=wgs84_GCS
# )
# gcs2ecef = pyproj.Transformer.from_crs(
#     crs_from=wgs84_GCS, crs_to=wgs84_ECEF
# )
# gcs2egm = pyproj.Transformer.from_crs(
#     crs_from=wgs84_GCS, crs_to=wgs84_EGM96
# )


class Ellipsoid:
    r"""Classe définissant l'ellipsoïde de révolution de référence à utiliser
    pour les calculs géodésiques. L'ellipsoïde de révolution est généralement
    défini à partir du couple :

    * :math:`(a,b)` :math:`\rightarrow` (rayon équatorial, rayon polaire) ou
    * :math:`(a,f)` :math:`\rightarrow` (rayon équatorial,
      facteur d'applatissement)
    
    L'ensemble des paramètres définissant l'ellipsoïde de révolution et leurs
    relations sont définis dans les attributs de cette classe.
    
    Parameters
    ----------
    a : float, optional
        Equatorial radius of the ellipsoid in meters.
        Default to equatorial radius of the WGS84 ellipsoid, i.e. a=6378137.0.
    b : float, optional
        Polar radius of the ellipsoid in meters. Default to `None`.
    f : float, optional
        Flattening factor of the ellipsoid.
        Default to flattening factor of the WGS84 ellipsoid, i.e. f=1/298.257223563.
    """
    def __init__(self, a=6378137.0, b=None, f=1/298.257223563):
        self._a, self._b, self._f = a, b, f
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
        self._n = self._f / (2 - self._f)         # Third flattening factor
        self._e2 = self._f * (2 - self._f)        # Excentricity squared
        self._e = np.sqrt(self._e2)               # Excentricity
        self._phichi = self._init_phichi_coeffs() # d(n)

    def _init_phichi_coeffs(self):
        """Coefficients of the series expansion of the inverse conformal latitude up to the 10th order."""
        n = [self._n**i for i in range(11)] # n^0 ... n^10
        # Coefficients d(n) of the series expansion of the inverse conformal latitude
        phichi = (
            # d2
            (2*n[1] - 2*n[2]/3 - 2*n[3] + 116*n[4]/45 + 26*n[5]/45 - 2854*n[6]/675
            + 16822*n[7]/4725 + 189416*n[8]/99225 - 1113026*n[9]/165375
            + 22150106*n[10]/4465125),
            # d4
            (7*n[2]/3 - 8*n[3]/5 - 227*n[4]/45 + 2704*n[5]/315 + 2323*n[6]/945
            - 31256*n[7]/1575 + 141514*n[8]/8505 + 10453448*n[9]/606375
            - 66355687*n[10]/1403325),
            # d6
            (56*n[3]/15 - 136*n[4]/35 - 1262*n[5]/105 + 73814*n[6]/2835
            + 98738*n[7]/14175 - 2363828*n[8]/31185 + 53146406*n[9]/779625
            + 1674405706*n[10]/18243225),
            # d8
            (4279*n[4]/630 - 332*n[5]/35 - 399572*n[6]/14175 + 11763988*n[7]/155925
            + 14416399*n[8]/935550 - 2647902052*n[9]/10135125
            + 23834033824*n[10]/91216125),
            # d10
            (4174*n[5]/315 - 144838*n[6]/6237 - 2046082*n[7]/31185
            + 258316372*n[8]/1216215 + 67926842*n[9]/2837835
            - 76998787574*n[10]/91216125),
            # d12
            (601676*n[6]/22275 - 115444544*n[7]/2027025 - 2155215124*n[8]/14189175
            + 41561762048*n[9]/70945875 + 625821359*n[10]/638512875),
            # d14
            (38341552*n[7]/675675 - 170079376*n[8]/1216215
            - 1182085822*n[9]/3378375 + 493459023622*n[10]/310134825),
            # d16
            (1383243703*n[8]/11351340 - 138163416988*n[9]/402026625
            - 1740830660174*n[10]/2170943775),
            # d18
            106974149462*n[9]/402026625 - 24899113566814*n[10]/29462808375,
            # d20
            175201343549*n[10]/297604125
        )
        return phichi

    def prime_vertical_curvature_radius(self, phi):
        r"""Calcul le rayon de courbure de la verticale principale
        d'un point de latitude :math:`\phi` :

        .. math:: \nu(\phi)=\dfrac{a}{\sqrt{1-e^2\sin^2\phi}}

        Parameters
        ----------
        phi : array_like
            La latitude géodésique.
        degrees : bool, optional
            L'unité de la latitude. Par défaut à True, la latitude fournie
            est en degrés.
        
        Returns
        -------
        nu : numpy.ndarray
            Rayon de courbure de la verticale principale en mètres.

        """
        nu = self._a / np.sqrt(1 - self._e2 * np.sin(phi)**2)
        return nu

    def isometric_latitude(self, phi):
        r"""Computes the isometric latitude (parameter of the Mercator projection):

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
        psi : numpy.ndarray
            The isometric latitude in radians.

        See Also
        --------
        inverse_isometric_latitude

        """
        sphi = np.sin(phi)
        psi = np.arctanh(sphi) - self._e * np.arctanh(self._e * sphi)
        return psi

    def inverse_isometric_latitude(self, psi):
        r"""Computes the inverse isometric latitude.

        The calculation is performed from the inverse function of the
        conformal latitude:

        .. math::
            \phi(\psi)=\chi^{-1}\big[\arcsin\big(\tanh(\psi)\big)\big]

        Parameters
        ----------
        psi :  array_like
            The isometric latitude in radians.
            
        Returns
        -------
        phi : numpy.ndarray
            The geodetic latitude in radians.

        See also
        --------
        isometric_latitude, inverse_conformal_latitude

        """
        chi = np.arcsin(np.tanh(psi))
        phi = self.inverse_conformal_latitude(chi)
        return phi

    def conformal_latitude(self, phi):
        r"""Computes the conformal latitude from the geodetic latitude:

        .. math::
            \chi(\phi)=\arcsin\big[\tanh\big(\mathrm{arctanh}(\sin(\phi))-
            e\mathrm{arctanh}(e\sin(\phi))\big)\big]

        The conformal latitude preserves angles during the projection
        of the ellipsoid onto the auxiliary sphere.
        
        Parameters
        ----------
        phi : array_like
            The geodetic latitude in radians.
            
        Returns
        -------
        chi : numpy.ndarray
            The conformal latitude in radians.

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
        r"""Computes the inverse conformal latitude.

        The calculation is performed from a harmonic expansion up to
        order 10 in terms of the third flattening factor :math:`n`
        using the Lagrange inversion formula:

        .. math::
            \phi(\chi)\simeq\chi+\sum\limits_{p=1}^{10}d_{2p}(n)\sin(2p\chi)
        
        Parameters
        ----------
        chi :  array_like
            The conformal latitude in radians.
            
        Returns
        -------
        phi : numpy.ndarray
            The geodetic latitude in radians.

        See also
        --------
        conformal_latitude

        """
        phi = np.copy(chi)
        for n, dn in enumerate(self._phichi):
            phi += dn * np.sin(2 * (n + 1) * chi)
        return phi
    
    def to_cartesian_ecef(self, lamb, phi, height_m=0):
        r"""Converts geodetic coordinates :math:`(\lambda,\phi, H)` of a
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
        lamb, phi : array_like
            The geographic longitude and latitude of the point in radians.
        height_m : array_like, optional
            The geodetic height of the point in meters. Default to `0`.
        
        Returns
        -------
        X, Y, Z : numpy.ndarray
            The cartesian ECEF coordinates of the point in meters.

        See Also
        --------
        prime_vertical_curvature_radius, to_cartographic

        """
        nu = self.prime_vertical_curvature_radius(phi) # Au niveau de l'ellipsoïde
        nuhcosphi = (nu + height_m) * np.cos(phi)
        X = nuhcosphi * np.cos(lamb)
        Y = nuhcosphi * np.sin(lamb)
        Z = ((1 - self._e2) * nu + height_m) * np.sin(phi)
        return X, Y, Z
    
    def to_cartographic(self, X, Y, Z):
        r"""Converts geocentric cartesian ECEF coordinates :math:`(X,Y,Z)`
        of a point to geodetic coordinates :math:`(\lambda,\phi, H)`.
        
        Conversion is made using the algorithm proposed by Vermeille (2002) but
        with limited application to the case where the evolute sign test is not performed,
        that is this algorithm is valid for points not "too deep" towards the center of
        the Earth (height > -6314 km for WGS84 ellipsoid), which is the case for our applications.

        "Vermeille, H. Direct transformation from geocentric coordinates to geodetic coordinates.
        Journal of Geodesy 76, 451–454 (2002). https://doi.org/10.1007/s00190-002-0273-6"

        Parameters
        ----------
        X, Y, Z : array_like
            The cartesian ECEF coordinates of the point in meters.
        
        Returns
        -------
        lamb, phi : numpy.ndarray
            The geographic longitude and latitude of the point in radians.
        height_m : numpy.ndarray
            The geodetic height of the point in meters. Default to `0`.

        See Also
        --------
        prime_vertical_curvature_radius, to_cartographic
        
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
        # Calcul de h et phi
        h = (k + self._e2 - 1) * hypotDZ / k
        phi = 2 * np.arctan(Z / (hypotDZ + D))
        # Calcul de lambda
        lamb = np.arctan2(Y, X)
        return lamb, phi, h

WGS84 = Ellipsoid() # WGS84 ellipsoid by default

class LambertConicConformal:
    r"""Lambert Conic Conformal projection class.
    
    Parameters
    ----------
    ellipsoid : Ellipsoid
        Reference ellipsoid used for the projection.
    lon_origin_rad, lat_origin_rad : float, optional
        Longitude and latitude of the origin of the Lambert Conic Conformal projection in radians.
    scale: float
        Scale factor of the projection. Default to `1.0`.
    x_offset_m, y_offset_m : float
        Offset from the projection origin in meters. Default to `0.0`.
    """
    def __init__(
            self,
            ellipsoid, lon_origin_rad, lat_origin_rad,
            scale=1.0, x_offset_m=0.0, y_offset_m=0.0
        ):
        R0 = (
            scale * ellipsoid.prime_vertical_curvature_radius(lat_origin_rad)
                / np.tan(lat_origin_rad)
        )
        self.ellipsoid = ellipsoid
        self.lon_origin_rad = lon_origin_rad
        self.lat_origin_rad = lat_origin_rad
        self.scale = scale
        self.x_offset_m = x_offset_m
        self.y_offset_m = y_offset_m + R0
        # Precompute constants
        self._sin_lat_origin = np.sin(self.lat_origin_rad)
        self._C = R0 * np.exp(
            self._sin_lat_origin * self.ellipsoid.isometric_latitude(self.lat_origin_rad)
        )

    def forward(self, lon_rad, lat_rad):
        r"""
        Forward transformation from geographic coordinates (lon_rad, lat_rad)
        to projected coordinates (x, y) in the Lambert Conic Conformal projection.

        """
        theta = self._sin_lat_origin * (lon_rad - self.lon_origin_rad)
        R = self._C * np.exp(-self._sin_lat_origin * self.ellipsoid.isometric_latitude(lat_rad))
        x_m = self.x_offset_m + R * np.sin(theta)
        y_m = self.y_offset_m - R * np.cos(theta)
        return x_m, y_m

    def inverse(self, x_m, y_m):
        r"""
        Inverse transformation from projected coordinates (x, y) in the Lambert
        Conic Conformal projection to geographic coordinates (lon_rad, lat_rad).

        """
        if self._sin_lat_origin >= 0.0:
            dx = x_m - self.x_offset_m
            dy = self.y_offset_m - y_m
        else:
            dx = self.x_offset_m - x_m
            dy = y_m - self.y_offset_m
        # Longitude
        lon_rad = self.lon_origin_rad + np.arctan2(dx, dy) / self._sin_lat_origin
        # Latitude
        lat_rad = self.ellipsoid.inverse_isometric_latitude(
            -np.log(np.abs(np.hypot(dx, dy) / self._C)) / self._sin_lat_origin
        )
        return lon_rad, lat_rad
