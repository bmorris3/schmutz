# Licensed under the MIT License - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import numpy as np
from scipy.integrate import quad
import astropy.units as u
from astropy.coordinates import UnitSphericalRepresentation, CartesianRepresentation
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product

from .sun import draw_random_sunspot_latitudes, draw_random_sunspot_radii

__all__ = ['Star', 'Spot']

trappist1_posteriors_path = os.path.join(os.path.dirname(__file__), os.pardir,
                                         'data', 'trappist1',
                                         'posteriors_bright_spot.txt')
                                         #'trappist1_spotmodel_posteriors.txt')
                                         #'trappist1_spotmodel_posteriors_onehemisphere.txt')
#'trappist1_spotmodel_posteriors.txt')

k296_posteriors_path = os.path.join(os.path.dirname(__file__), os.pardir,
                                    'data', 'k296',
                                    'k296_spotmodel_posteriors.txt')

#np.random.seed(42)


class Spot(object):
    """
    Properties of a starspot.
    """
    def __init__(self, x=None, y=None, z=None, r=None, stellar_radius=1,
                 contrast=None):
        """
        Parameters
        ----------
        x : float
            X position [stellar radii]
        y : float
            Y position [stellar radii]
        z : float
            Z position [stellar radii], default is ``z=sqrt(r_star^2 - x^2 - y^2)``.
        r : float
            Spot radius [stellar radii], default is ``r=1``.
        stellar_radius : float
            Radius of the star, in the same units as ``x,y,z,r``. Default is 1.
        """
        if z is None:
            z = np.sqrt(stellar_radius**2 - x**2 - y**2)
        self.r = r
        self.cartesian = CartesianRepresentation(x=x, y=y, z=z)
        self.contrast = contrast

    @classmethod
    def from_latlon(cls, latitude, longitude, radius, contrast=None):
        """
        Construct a spot from latitude, longitude coordinates

        Parameters
        ----------
        latitude : float
            Spot latitude [deg]
        longitude : float
            Spot longitude [deg]
        radius : float
            Spot radius [stellar radii]
        """

        cartesian = latlon_to_cartesian(latitude, longitude)

        return cls(x=cartesian.x.value, y=cartesian.y.value,
                   z=cartesian.z.value, r=radius, contrast=contrast)

    @classmethod
    def from_sunspot_distribution(cls, mean_latitude=15, radius_multiplier=1,
                                  contrast=0.7):
        """
        Parameters
        ----------
        mean_latitude : float
            Define the mean absolute latitude of the two symmetric active
            latitudes, where ``mean_latitude > 0``.
        """
        lat = draw_random_sunspot_latitudes(n=1, mean_latitude=mean_latitude)[0]
        lon = 2*np.pi * np.random.rand() * u.rad
        radius = draw_random_sunspot_radii(n=1)[0]

        cartesian = latlon_to_cartesian(lat, lon)

        return cls(x=cartesian.x.value, y=cartesian.y.value,
                   z=cartesian.z.value, r=radius*radius_multiplier,
                   contrast=contrast)

    def __repr__(self):
        return ("<Spot: x={0}, y={1}, z={2}, r={3}>"
                .format(self.x, self.y, self.z, self.r))


def latlon_to_cartesian(latitude, longitude, stellar_inclination=90*u.deg):
    """
    Convert coordinates in latitude/longitude for a star with a given
    stellar inclination into cartesian coordinates.

    The X-Y plane is the sky plane: x is aligned with the stellar equator, y is
    aligned with the stellar rotation axis.

    Parameters
    ----------
    latitude : float or `~astropy.units.Quantity`
        Spot latitude. Will assume unit=deg if none is specified.
    longitude : float or `~astropy.units.Quantity`
        Spot longitude. Will assume unit=deg if none is specified.
    stellar_inclination : float
        Stellar inclination angle, measured away from the line of sight,
        in [deg]. Default is 90 deg.

    Returns
    -------
    cartesian : `~astropy.coordinates.CartesianRepresentation`
        Cartesian representation in the frame described above.
    """

    if not hasattr(longitude, 'unit') and not hasattr(latitude, 'unit'):
        longitude *= u.deg
        latitude *= u.deg

    c = UnitSphericalRepresentation(longitude, latitude)
    cartesian = c.to_cartesian()

    rotate_about_z = rotation_matrix(90*u.deg, axis='z')
    rotate_is = rotation_matrix(stellar_inclination, axis='y')
    transform_matrix = matrix_product(rotate_about_z, rotate_is)
    cartesian = cartesian.transform(transform_matrix)
    return cartesian


class Star(object):
    """
    Object defining a star.
    """
    def __init__(self, spots=None, u1=0.4987, u2=0.1772, r=1,
                 radius_threshold=0.1, rotation_period=25*u.day):
        """
        The star is assumed to have stellar inclination 90 deg (equator-on).

        Parameters
        ----------
        u1 : float (optional)
            Quadratic limb-darkening parameter, linear term
        u2 : float (optional)
            Quadratic limb-darkening parameter, quadratic term
        r : float (optional)
            Stellar radius (default is unity)
        radius_threshold : float (optional)
            If all spots are smaller than this radius, use the analytic solution
            to compute the stellar centroid, otherwise use the numerical
            solution.
        spots : list (optional)
            List of spots on this star.
        rotation_period : `~astropy.units.Quantity`
            Stellar rotation period [default = 25 d].
        contrast : float (optional)
            Spot contrast relative to photosphere. Default is ``c=0.7``
        """
        if spots is None:
            spots = []
        self.spots = spots

        self.spots_cartesian = CartesianRepresentation(x=[spot.cartesian.x for spot in spots],
                                                       y=[spot.cartesian.y for spot in spots],
                                                       z=[spot.cartesian.z for spot in spots])
        self.spots_r = np.array([spot.r for spot in spots])
        self.spot_contrasts = np.array([spot.contrast for spot in spots])
        self.x = 0
        self.y = 0
        self.r = r
        self.u1 = u1
        self.u2 = u2
        self.radius_threshold = radius_threshold
        self.rotations_applied = 0 * u.deg
        self.rotation_period = rotation_period
        self.inclination = 90*u.deg
        self.unspotted_flux = (2 * np.pi *
                               quad(lambda r: r * self.limb_darkening_normed(r),
                                    0, self.r)[0])

    def plot(self, n=3000, ax=None):
        """
        Plot a 2D projected schematic of the star and its spots.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axis object to draw the plot on
        n : int
            Number of pixels per side in the image.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Matplotlib axis object, with the new plot on it.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        image = self._compute_image(n=n)

        ax.imshow(image, origin='lower', interpolation='nearest',
                  cmap=plt.cm.Greys_r, extent=[-1, 1, -1, 1],
                  vmin=0, vmax=1)
        ax.set_aspect('equal')

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xlabel('x [$R_\star$]', fontsize=14)
        ax.set_ylabel('y [$R_\star$]', fontsize=14)
        return ax

    @classmethod
    def with_trappist1_spot_distribution(cls):
        samples = np.loadtxt(trappist1_posteriors_path)
        sample_index = np.random.randint(0, samples.shape[0])

        lat0, lon0, rad0, lat1, lon1, rad1, lat2, lon2, rad2, contrast, kep_offset = samples[sample_index, :]

        spots = [Spot.from_latlon(lat0, lon0, rad0, contrast),
                 Spot.from_latlon(lat1, lon1, rad1, contrast),
                 Spot.from_latlon(lat2, lon2, rad2, contrast)]

        return cls(spots=spots, rotation_period=3.3*u.day)

    @classmethod
    def with_k296_spot_distribution(cls):
        samples = np.loadtxt(k296_posteriors_path)
        sample_index = np.random.randint(0, samples.shape[0])

        lat0, lon0, rad0, lat1, lon1, rad1, contrast, kep_offset = samples[sample_index, :]

        spots = [Spot.from_latlon(lat0, lon0, rad0, contrast),
                 Spot.from_latlon(lat1, lon1, rad1, contrast)]

        return cls(spots=spots, rotation_period=36.085629155859351*u.day)

    def spotted_area(self, times, t0=0):
        """
        Compute flux at ``times`` as the star rotates.

        Parameters
        ----------
        times: `~numpy.ndarray`
            Times
        t0 : float
            Reference epoch.

        Returns
        -------
        area : `~numpy.ndarray`
            Area covered by spots at ``times`` [Hem]
        """
        p_rot_d = self.rotation_period.to(u.d).value
        rotational_phases = (((times - t0) % p_rot_d) / p_rot_d) * 2*np.pi*u.rad

        # Rotate the star about its axis assuming stellar inclination 90 deg
        transform_matrix = rotation_matrix(rotational_phases[:, np.newaxis],
                                           axis='y')
        old_cartesian = self.spots_cartesian
        new_cartesian = old_cartesian.transform(transform_matrix)

        # Use numpy array broadcasting to vectorize computations with spot radii
        broadcast_radii = (np.ones_like(rotational_phases.value)[:, np.newaxis]
                           * self.spots_r)

        # Only include spot flux if it's on the observer facing side
        visible = (new_cartesian.z > 0).astype(int)

        # Compute radial position of spot
        r_spots = np.hypot(new_cartesian.x, new_cartesian.y)  #np.sqrt(new_cartesian.x**2 + new_cartesian.y**2)

        # Compute approximate spot area, given foreshortening in 3D
        spot_areas = (np.pi * broadcast_radii**2 *
                      np.sqrt(1 - (r_spots/self.r)**2))
        area = np.sum(spot_areas * visible, axis=1) / (2 * np.pi * self.r**2)
        if hasattr(area, 'unit'):
            area = area.value
        return area

    def flux(self, times, t0=0):
        """
        Compute flux at ``times`` as the star rotates.

        Parameters
        ----------
        times: `~numpy.ndarray`
            Times
        t0 : float
            Reference epoch.

        Returns
        -------
        flux : `~numpy.ndarray`
            Fluxes at ``times``
        """
        p_rot_d = self.rotation_period.to(u.d).value
        rotational_phases = (((times - t0) % p_rot_d) / p_rot_d) * 2*np.pi*u.rad

        # Rotate the star about its axis assuming stellar inclination 90 deg
        transform_matrix = rotation_matrix(rotational_phases[:, np.newaxis],
                                           axis='y')
        old_cartesian = self.spots_cartesian
        new_cartesian = old_cartesian.transform(transform_matrix)

        # Use numpy array broadcasting to vectorize computations with spot radii
        broadcast_radii = (np.ones_like(rotational_phases.value)[:, np.newaxis]
                           * self.spots_r)

        # Only include spot flux if it's on the observer facing side
        visible = (new_cartesian.z > 0).astype(int)

        # Compute radial position of spot
        r_spots = np.sqrt(new_cartesian.x**2 + new_cartesian.y**2)

        # Compute approximate spot area, given foreshortening in 3D
        spot_areas = (np.pi * broadcast_radii**2 *
                      np.sqrt(1 - (r_spots/self.r)**2))

        # For a given spot contrast and limb darkening, compute missing flux
        spot_flux = (-1 * spot_areas * self.limb_darkening_normed(r_spots) *
                     (1 - self.spot_contrasts)) * visible

        return self.unspotted_flux + np.sum(spot_flux, axis=1)

    def fractional_flux(self, times, t0=0):
        """
        Compute stellar flux as a fraction of the unspotted stellar flux at
        ``times`` as the star rotates.

        Parameters
        ----------
        times: `~numpy.ndarray`
            Times
        t0 : float
            Reference epoch.

        Returns
        -------
        flux : `~numpy.ndarray`
            Fluxes at ``times``
        """
        return self.flux(times, t0=t0)/self.unspotted_flux
    #
    # def flux_weighted_area(self, times, t0=0):
    #     """
    #     Compute flux at ``times`` as the star rotates.
    #
    #     Parameters
    #     ----------
    #     times: `~numpy.ndarray`
    #         Times
    #     t0 : float
    #         Reference epoch.
    #
    #     Returns
    #     -------
    #     flux : `~numpy.ndarray`
    #         Fluxes at ``times``
    #     """
    #     p_rot_d = self.rotation_period.to(u.d).value
    #     rotational_phases = (((times - t0) % p_rot_d) / p_rot_d) * 2*np.pi*u.rad
    #
    #     # Rotate the star about its axis assuming stellar inclination 90 deg
    #     transform_matrix = rotation_matrix(rotational_phases[:, np.newaxis],
    #                                        axis='y')
    #     old_cartesian = self.spots_cartesian
    #     new_cartesian = old_cartesian.transform(transform_matrix)
    #
    #     # Use numpy array broadcasting to vectorize computations with spot radii
    #     broadcast_radii = (np.ones_like(rotational_phases.value)[:, np.newaxis]
    #                        * self.spots_r)
    #
    #     # Only include spot flux if it's on the observer facing side
    #     visible = (new_cartesian.z > 0).astype(int)
    #
    #     # Compute radial position of spot
    #     r_spots = np.sqrt(new_cartesian.x**2 + new_cartesian.y**2)
    #
    #     # Compute approximate spot area, given foreshortening in 3D
    #     spot_areas = (np.pi * broadcast_radii**2 *
    #                   np.sqrt(1 - (r_spots/self.r)**2))
    #
    #     area_spotted = np.sum(spot_areas * visible, axis=1) / (2 * np.pi * self.r**2)
    #
    #     if hasattr(area_spotted, 'unit'):
    #         area_spotted = area_spotted.value
    #
    #     # For a given spot contrast and limb darkening, compute missing flux
    #     flux_spots = np.sum((spot_areas * self.limb_darkening_normed(r_spots) *
    #                         (1 - self.contrast)) * visible, axis=1)
    #
    #     missing_photosphere = (-1 * spot_areas *
    #                            self.limb_darkening_normed(r_spots)) * visible
    #
    #     flux_photosphere = self.unspotted_flux + np.sum(missing_photosphere,
    #                                                     axis=1)
    #
    #     photosphere_area = 1 - area_spotted
    #
    #     # flux_weighted_spot_area = (flux_photosphere * photosphere_area +
    #     #                            flux_spots * area_spotted)
    #
    #     flux_weighted_spot_area = (flux_spots * area_spotted /
    #                                (flux_photosphere * photosphere_area))
    #
    #     if hasattr(flux_weighted_spot_area, 'unit'):
    #         flux_weighted_spot_area = flux_weighted_spot_area.value
    #
    #     return flux_weighted_spot_area

    def _compute_image(self, n=3000, delete_arrays_after_use=True):
        """
        Compute the stellar centroid using a numerical approximation.

        Parameters
        ----------
        n : int
            Generate a simulated image of the star with ``n`` by ``n`` pixels.

        Returns
        -------
        x_centroid : float
            Photocenter in the x dimension, in units of stellar radii
        y_centroid : float
            Photocenter in the y dimension, in units of stellar radii
        """
        image = np.zeros((n, n))
        x = np.linspace(-self.r, self.r, n)
        y = np.linspace(-self.r, self.r, n)
        x, y = np.meshgrid(x, y)

        # Limb darkening
        irradiance = self.limb_darkening_normed(np.sqrt(x**2 + y**2))

        on_star = x**2 + y**2 <= self.r**2

        image[on_star] = irradiance[on_star]
        on_spot = None

        for cartesian, r, c in zip(self.spots_cartesian, self.spots_r,
                                   self.spot_contrasts):
            if cartesian.z > 0:
                r_spot = np.sqrt(cartesian.x**2 + cartesian.y**2)
                foreshorten_semiminor_axis = np.sqrt(1 - (r_spot/self.r)**2)

                a = r  # Semi-major axis
                b = r * foreshorten_semiminor_axis  # Semi-minor axis
                A = np.pi/2 + np.arctan2(cartesian.y.value, cartesian.x.value)  # Semi-major axis rotation
                on_spot = (((x - cartesian.x) * np.cos(A) +
                            (y - cartesian.y) * np.sin(A))**2 / a**2 +
                           ((x - cartesian.x) * np.sin(A) -
                            (y - cartesian.y) * np.cos(A))**2 / b**2 <= self.r**2)

                image[on_spot & on_star] *= c

        if delete_arrays_after_use:
            del on_star
            if on_spot is not None:
                del on_spot
            del x
            del y
            del irradiance

        return image

    def limb_darkening(self, r):
        """
        Compute the intensity at radius ``r`` for quadratic limb-darkening law
        with parameters ``Star.u1, Star.u2``.

        Parameters
        ----------
        r : float or `~numpy.ndarray`
            Stellar surface position in radial coords on (0, 1)

        Returns
        -------
        intensity : float
            Intensity in un-normalized units
        """
        mu = np.sqrt(1 - r**2)
        u1 = self.u1
        u2 = self.u2
        return (1 - u1 * (1 - mu) - u2 * (1 - mu)**2) / (1 - u1/3 - u2/6) / np.pi

    def limb_darkening_normed(self, r):
        """
        Compute the normalized intensity at radius ``r`` for quadratic
        limb-darkening law with parameters ``Star.u1, Star.u2``.

        Parameters
        ----------
        r : float or `~numpy.ndarray`
            Stellar surface position in radial coords on (0, 1)

        Returns
        -------
        intensity : float
            Intensity relative to the intensity at the center of the disk.
        """
        return self.limb_darkening(r) / self.limb_darkening(0)

    def rotate(self, angle):
        """
        Rotate the star, by moving the spots.

        Parameters
        ----------
        angle : `~astropy.units.Quantity`

        """
        transform_matrix = rotation_matrix(angle, axis='y')

        old_cartesian = self.spots_cartesian
        new_cartesian = old_cartesian.transform(transform_matrix)
        self.spots_cartesian = new_cartesian
        self.rotations_applied += angle

    def derotate(self):
        self.rotate(-self.rotations_applied)
        self.rotations_applied = 0
