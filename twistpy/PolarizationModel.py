"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2022
:license:
    None
"""
import sys

import numpy as np


class PolarizationModel:
    r""" Six-component pure-state polarization model.

    This class computes the six-component pure-state polarization vector for the specified wave-type
    according to Equations 40 in Sollberger et al. (2018) [1], corrected in [2]. Polarization models can either be computed for recordings
    at the free surface or inside the medium.

    |  [1] https://doi.org/10.1093/gji/ggx542
    |  [2] https://doi.org/10.1093/gji/ggy205

    Parameters
    ----------
    wave_type : :obj:`str`
        Wave type

        |  'P':     P-wave
        |  'SV':    SV-wave
        |  'SH':    SH-wave
        |  'L':     Love-wave
        |  'R':     Rayleigh-wave
    vp : :py:class:`numpy.ndarray`
        P-wave velocity (m/s) at the receiver location
    vs : :obj:`numpy.ndarray`
        S-wave velocity (m/s) at the receiver location
    vl : :obj:`numpy.ndarray`
        Love-wave velocity (m/s) at the receiver location
    vr : :obj:`numpy.ndarray`
        Rayleigh-wave velocity (m/s) at the receiver location
    theta : :obj:`numpy.ndarray`
        Inclination angle (degree), only for body waves
    xi : :obj:`numpy.ndarray`
        Ellipticity angle (degree), for Rayleigh waves
    scaling_velocity : :obj:`float`
        Scaling velocity (m/s) applied to the translational components
        to convert translations to dimensionless units, Default: 1 (m/s)
    free_surface : :obj:`bool`
        True (default): the wave is recorded at the free surface, False: the wave is recorded inside
        the medium
    """

    # :param wave_type: Wave-type \n
    #     'P': P-wave \n
    #     'SV': SV-wave \n
    #     'SH': SH-wave \n
    #     'L': Love-wave \n
    #     'R': Rayleigh-wave
    # :type wave_type: str
    # :param vp: P-wave velocity (m/s) at the receiver location
    # :type vp: class: 'numpy.ndarray'
    # :param vs: S-wave velocity (m/s) at the receiver location
    # :type vs: class: 'numpy.ndarray'
    # :param vl: Love-wave velocity (m/s)
    # :type vl: class: 'numpy.ndarray'
    # :param vr: Rayleigh-wave velocity (m/s)
    # :type vr: class: 'numpy.ndarray'
    # :param theta: Inclination (degree), only for body waves
    # :type theta: class: 'numpy.ndarray'
    # :param xi: Ellipticity angle (rad) for Rayleigh waves
    # :type xi: class: 'numpy.ndarray'
    # :param scaling_velocity: scaling velocity (m/s) to make translations dimensionless, Default: 1 (m/s)
    # :type scaling_velocity: float
    # :param free_surface: True (default): the wave is recorded at the free surface, False: the wave is recorded inside
    #     the medium
    # :type free_surface: bool
    # :attribute polarization: 6-C polarization vector (automatically computed from the input parameters)
    # :type polarization: class: 'numpy.ndarray'
    # :property polarization_vector: Computes the polarization vector for the current instance of the class 'Wave'
    def __init__(self, wave_type: str, scaling_velocity: float = 1, free_surface: bool = True,
                 vp: np.ndarray = None,
                 vs: np.ndarray = None, vl: np.ndarray = None, vr: np.ndarray = None, theta: np.ndarray = None,
                 phi: np.ndarray = None,
                 xi: np.ndarray = None) -> None:
        self.free_surface, self.wave_type = free_surface, wave_type
        self.vp, self.vs, self.vl, self.vr, self.theta, self.phi, self.xi = vp, vs, vl, vr, theta, phi, xi
        self.scaling_velocity = scaling_velocity
        self.polarization = self.polarization_vector

    @property
    def polarization_vector(self) -> np.ndarray:
        """
        Pure state six-component polarization vector
        """

        if self.wave_type == 'P':
            # Exclude non-physical vp to vs ratios
            self.vp[self.vp / self.vs > 4] = np.nan
            self.vp[self.vp / self.vs < 1.66] = np.nan
            self.vs[self.vp / self.vs > 4] = np.nan
            self.vs[self.vp / self.vs < 1.66] = np.nan

            if self.free_surface:
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)

                v = (self.vp ** 2 - 2. * self.vs ** 2) / (2. * (self.vp ** 2 - self.vs ** 2))  # Poisson's ratio
                kappa = (2. * (1 - v) / (1 - 2 * v)) ** (1 / 2.)
                theta_s = np.arcsin((1 / kappa) * np.sin(theta_rad))  # angle of reflected S-wave

                # amplitude of reflected P-wave
                alpha_pp = (np.sin(2 * theta_rad) * np.sin(2 * theta_s) - kappa ** 2 * (np.cos(2 * theta_s)) ** 2) / \
                           (np.sin(2 * theta_rad) * np.sin(2 * theta_s) + kappa ** 2 * (np.cos(2 * theta_s)) ** 2)

                # amplitude of reflected S-wave
                alpha_ps = (2 * kappa * np.sin(2 * theta_rad) * np.cos(2 * theta_s)) / \
                           (np.sin(2 * theta_rad) * np.sin(2 * theta_s) + kappa ** 2 * (np.cos(2 * theta_s)) ** 2)

                v_x = -(np.sin(theta_rad) * np.cos(phi_rad)
                        + alpha_pp * np.sin(theta_rad) * np.cos(phi_rad)
                        + alpha_ps * np.cos(theta_s) * np.cos(phi_rad)) / self.scaling_velocity
                v_y = (np.sin(theta_rad) * np.sin(phi_rad)
                       + alpha_pp * np.sin(theta_rad) * np.sin(phi_rad)
                       + alpha_ps * np.cos(theta_s) * np.sin(phi_rad)) / self.scaling_velocity
                v_z = (np.cos(theta_rad)
                       - alpha_pp * np.cos(theta_rad)
                       + alpha_ps * np.sin(theta_s)) / self.scaling_velocity
                w_x = (1 / 2.) * alpha_ps * np.sin(phi_rad) / self.vs
                w_y = (1 / 2.) * alpha_ps * np.cos(phi_rad) / self.vs
                w_z = 0. * w_x

                polarization = np.asarray([v_x, v_y, v_z, w_x, w_y, w_z]).astype('complex').squeeze()
            elif not self.free_surface:
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)
                v_x = - (1. / self.scaling_velocity) * np.sin(theta_rad) * np.cos(phi_rad)
                v_y = (1. / self.scaling_velocity) * np.sin(theta_rad) * np.sin(phi_rad)
                v_z = (1. / self.scaling_velocity) * np.cos(theta_rad)
                w_x = 0. * v_x
                w_y = 0. * v_x
                w_z = 0. * v_x
                polarization = np.asarray([v_x, v_y, v_z, w_x, w_y, w_z]).astype('complex').squeeze()

        elif self.wave_type == 'SV':
            # Exclude non-physical vp to vs ratios
            self.vp[self.vp / self.vs > 4] = np.nan
            self.vp[self.vp / self.vs < 1.66] = np.nan
            self.vs[self.vp / self.vs > 4] = np.nan
            self.vs[self.vp / self.vs < 1.66] = np.nan
            if self.free_surface:
                theta_rad = np.radians(self.theta)
                theta_rad[theta_rad == np.pi / 4] = np.nan
                phi_rad = np.radians(self.phi)
                v = (self.vp ** 2 - 2. * self.vs ** 2) / (2. * (self.vp ** 2 - self.vs ** 2))  # poisson's ratio
                kappa = (2 * (1 - v) / (1 - 2 * v)) ** (1 / 2.)
                theta_crit = np.arcsin(1 / kappa)

                # Check whether the current incidence angle is at or above the critical angle
                theta_p = np.zeros_like(theta_rad, dtype='complex')
                alpha_sp = np.zeros_like(theta_rad, dtype='complex')
                alpha_ss = np.zeros_like(theta_rad, dtype='complex')

                # Special case, critical angle
                index = theta_rad == theta_crit
                theta_p[index] = np.pi / 2.
                alpha_sp[index] = (4. * (kappa[index] ** 2 - 1)) / (kappa[index] * (2 - kappa[index] ** 2))
                alpha_ss[index] = -1

                # Above critical angle (complex polarizaton)
                index = theta_crit < theta_rad
                theta_p[index] = np.arcsin(np.sin(theta_rad[index]) * self.vp[index] /
                                           self.vs[index]).astype('complex')
                alpha_ss[index] = (4 * (np.sin(theta_rad[index]) ** 2 - kappa[index] ** (-2)) * np.sin(
                    2 * theta_rad[index]) ** 2 *
                                   np.sin(theta_rad[index]) ** 2
                                   - np.cos(theta_rad[index]) ** 4 + 4 * 1j * (
                                           np.sin(theta_rad[index]) ** 2 - kappa[index] ** -2) ** (
                                           1 / 2.) *
                                   np.sin(2 * theta_rad[index]) * np.sin(theta_rad[index]) * (
                                       np.cos(2 * theta_rad[index])) ** 2) \
                                  / (np.cos(2 * theta_rad[index]) ** 4 + 4 * (
                        np.sin(theta_rad[index]) ** 2 - kappa[index] ** -2) *
                                     np.sin(2 * theta_rad[index]) ** 2 * np.sin(theta_rad[index]) ** 2)
                alpha_sp[index] = (2 * kappa[index] ** -1 * np.sin(2 * theta_rad[index]) * np.cos(2 * theta_rad[index])
                                   * (np.cos(2 * theta_rad[index]) ** 2 - 2
                                      * 1j * (np.sin(theta_rad[index]) ** 2 - kappa[index] ** (-2)) ** (1 / 2.)
                                      * np.sin(2 * theta_rad[index]) * np.sin(theta_rad[index]))) / \
                                  (np.cos(2 * theta_rad[index]) ** 4 + 4 * (
                                          np.sin(theta_rad[index]) ** 2 - kappa[index] ** -2)
                                   * np.sin(2 * theta_rad[index]) ** 2 * np.sin(theta_rad[index]) ** 2)
                # Sub-critical
                index = theta_rad < theta_crit
                theta_p[index] = np.arcsin(np.sin(theta_rad[index]) * self.vp[index] / self.vs[index])
                alpha_ss[index] = (np.sin(2 * theta_rad[index]) * np.sin(2 * theta_p[index]) - kappa[index] ** 2 * (
                    np.cos(2 * theta_p[index])) ** 2) \
                                  / (np.sin(2 * theta_rad[index]) * np.sin(2 * theta_p[index]) + kappa[index] ** 2 * (
                    np.cos(2 * theta_rad[index])) ** 2)
                alpha_sp[index] = -(kappa[index] * np.sin(4 * theta_rad[index])) \
                                  / (np.sin(2 * theta_rad[index]) * np.sin(2 * theta_p[index])
                                     + kappa[index] ** 2 * (np.cos(2 * theta_rad[index])) ** 2)

                v_x = (np.cos(theta_rad) * np.cos(phi_rad)
                       - alpha_ss * np.cos(theta_rad) * np.cos(phi_rad)
                       - alpha_sp * np.sin(theta_p) * np.cos(phi_rad)) / self.scaling_velocity
                v_y = - (np.cos(theta_rad) * np.sin(phi_rad)
                         - alpha_ss * np.cos(theta_rad) * np.sin(phi_rad)
                         - alpha_sp * np.sin(theta_p) * np.sin(phi_rad)) / self.scaling_velocity
                v_z = (np.sin(theta_rad)
                       + alpha_ss * np.sin(theta_rad)
                       - alpha_sp * np.cos(theta_p)) / self.scaling_velocity

                w_x = (1 / 2.) * (1 + alpha_ss) * np.sin(phi_rad) / self.vs
                w_y = (1 / 2.) * (1 + alpha_ss) * np.cos(phi_rad) / self.vs
                w_z = 0. * w_x
                polarization = np.asarray([v_x, v_y, v_z, w_x, w_y, w_z]).astype('complex').squeeze()

            elif not self.free_surface:
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)
                v_x = (1. / self.scaling_velocity) * np.cos(theta_rad) * np.cos(phi_rad)
                v_y = -(1. / self.scaling_velocity) * np.cos(theta_rad) * np.sin(phi_rad)
                v_z = (1. / self.scaling_velocity) * np.sin(theta_rad)
                w_x = (2 * self.vs) ** -1 * np.sin(phi_rad)
                w_y = (2 * self.vs) ** -1 * np.cos(phi_rad)
                w_z = 0. * w_x
                polarization = np.asarray([v_x, v_y, v_z, w_x, w_y, w_z]).astype('complex').squeeze()

        elif self.wave_type == 'SH':
            if self.free_surface:
                phi_rad = np.radians(self.phi)
                theta_rad = np.radians(self.theta)
                v_x = 2. / self.scaling_velocity * np.sin(phi_rad)
                v_y = 2. / self.scaling_velocity * np.cos(phi_rad)
                v_z = 0. * v_x
                w_x = 0. * v_x
                w_y = 0. * v_x
                w_z = - 1. / self.vs * np.sin(theta_rad)
                polarization = np.asarray([v_x, v_y, v_z, w_x, w_y, w_z]).astype('complex').squeeze()
            else:
                phi_rad = np.radians(self.phi)
                theta_rad = np.radians(self.theta)
                v_x = (1. / self.scaling_velocity) * np.sin(phi_rad)
                v_y = (1. / self.scaling_velocity) * np.cos(phi_rad)
                v_z = 0. * v_x
                w_x = - (2 * self.vs) ** -1 * np.cos(theta_rad) * np.cos(phi_rad)
                w_y = (2 * self.vs) ** -1 * np.cos(theta_rad) * np.sin(phi_rad)
                w_z = (2 * self.vs) ** -1 * np.sin(theta_rad)
                polarization = np.asarray([v_x, v_y, v_z, w_x, w_y, w_z]).astype('complex').squeeze()

        elif self.wave_type == 'R':
            phi_rad = np.radians(self.phi)
            xi_rad = np.radians(self.xi)
            v_x = -1j * 1. / self.scaling_velocity * np.sin(xi_rad) * np.cos(phi_rad)
            v_y = 1j * 1. / self.scaling_velocity * np.sin(xi_rad) * np.sin(phi_rad)
            v_z = 1. / self.scaling_velocity * np.cos(xi_rad)

            w_x = 1. / self.vr * np.sin(phi_rad) * np.cos(xi_rad)
            w_y = 1. / self.vr * np.cos(phi_rad) * np.cos(xi_rad)
            w_z = 0. * v_x
            polarization = np.asarray([v_x, v_y, v_z, w_x, w_y, w_z]).astype('complex').squeeze()
        elif self.wave_type == 'L':
            phi_rad = np.radians(self.phi)
            v_x = 1 / self.scaling_velocity * np.sin(phi_rad)
            v_y = 1 / self.scaling_velocity * np.cos(phi_rad)
            v_z = 0. * v_x

            w_x = 0. * v_x
            w_y = 0. * v_x
            w_z = - 1. / (2 * self.vl)
            polarization = np.asarray([v_x, v_y, v_z, w_x, w_y, w_z]).astype('complex').squeeze()

        elif self.wave_type not in ['L', 'R', 'P', 'SV', 'SH']:
            sys.exit(f"Invalid wave type '{self.wave_type}' specified!")

        polarization = np.divide(polarization, np.linalg.norm(polarization, axis=0))
        return polarization
