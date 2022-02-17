"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2022
:license:
    None
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *
import numpy as np

from gc import get_referents
from gc import collect
import sys
from machine_learning import SupportVectorMachine
from obspy import Trace
import pickle
import multiprocessing as mp
from scipy.optimize import differential_evolution
from obspy.signal.tf_misfit import cwt
from obspy.signal.util import next_pow_2
from scipy.signal import spectrogram, hanning, convolve, hilbert
from obspy import Stream, Trace
from datetime import datetime
import matplotlib.pyplot as plt
import tables as tb
from matplotlib import colors
from scipy.interpolate import interp2d
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import median_filter

sys.path.insert(0, "./SVC_models")


class DispersionAnalysis:

    def __init__(self, traN: Trace = None, traE: Trace = None, traZ: Trace = None, rotN: Trace = None,
                 rotE: Trace = None,
                 rotZ: Trace = None, method: str = 'pca'):
        self.traN, self.traE, self.traZ, self.rotN, self.rotE, self.rotZ = traN, traE, traZ, rotN, rotE, rotZ

        # Assert that input traces are ObsPy Trace objects
        assert isinstance(self.traN, Trace) and isinstance(self.traE, Trace) and isinstance(self.traZ, Trace) \
               and isinstance(self.rotN, Trace) and isinstance(self.rotE, Trace) and isinstance(self.rotZ, Trace), \
            "Input data must be of class obspy.core.Trace()!"

        # Assert that all traces have the same number of samples
        assert self.traN.stats.npts == self.traE.stats.npts and self.traN.stats.npts == self.traZ.stats.npts \
               and self.traN.stats.npts == self.rotN.stats.npts and self.traN.stats.npts == self.rotE.stats.npts \
               and self.traN.stats.npts == self.rotZ.stats.npts, \
            "All six traces must have the same number of samples!"


class TimeFrequencyAnalysis:

    def __init__(self):
        pass


class TimeDomainAnalysis:
    def __init__(self, traN: Trace = None, traE: Trace = None, traZ: Trace = None, rotN: Trace = None,
                 rotE: Trace = None, rotZ: Trace = None, window: dict = None, method: str = 'pca',
                 scaling_velocity: float = 1., free_surface: bool = True, verbose: bool = True):

        self.traN, self.traE, self.traZ, self.rotN, self.rotE, self.rotZ = traN, traE, traZ, rotN, rotE, rotZ

        # Assert that input traces are ObsPy Trace objects
        assert isinstance(self.traN, Trace) and isinstance(self.traE, Trace) and isinstance(self.traZ, Trace) \
               and isinstance(self.rotN, Trace) and isinstance(self.rotE, Trace) and isinstance(self.rotZ, Trace),\
            "Input data must be objects of class obspy.core.Trace()!"

        # Assert that all traces have the same number of samples
        assert self.traN.stats.npts == self.traE.stats.npts and self.traN.stats.npts == self.traZ.stats.npts \
               and self.traN.stats.npts == self.rotN.stats.npts and self.traN.stats.npts == self.rotE.stats.npts \
               and self.traN.stats.npts == self.rotZ.stats.npts, "All six traces must have the same number of samples!"

        if window is None:
            self.window = {'window_length_seconds': 1., 'overlap': 0.5}
        else:
            self.window = window
        self.method = method
        self.time = self.traN.times(type="utcdatetime")
        self.scaling_velocity = scaling_velocity
        self.free_surface = free_surface
        self.verbose = verbose
        self.delta = self.traN.stats.delta
        self.window_length_samples = 2 * int((self.window['window_length_seconds'] / self.delta) / 2)
        start, stop, incr = int(self.window_length_samples / 2), -1 - int(self.window_length_samples / 2), \
                            np.max([1, int((1 - self.window['overlap']) * self.window_length_samples)])
        self.time_polarization_analysis = self.time[start:stop:incr]

        # Compute the analytic signal
        u: np.ndarray = np.array([hilbert(self.traN).T, hilbert(self.traE).T, hilbert(self.traZ).T, \
                                           hilbert(self.rotN).T, hilbert(self.rotE).T, hilbert(self.rotZ).T]).T
        # Compute covariance matrices
        if self.verbose:
            print('Computing covariance matrices...\n')
        C: "np.ndarray[np.complex]" = np.einsum('...i,...j->...ij', np.conj(u), u).astype('complex')
        w = hanning(self.window_length_samples+2)[1:-1] # Hann window for covariance matrix averaging
        w /= np.sum(w)
        for j in range(C.shape[2]):
            for k in range(C.shape[1]):
                C[..., j, k] = \
                    convolve(C[..., j, k].real, w, mode='same') + convolve(C[..., j, k].imag, w, mode='same')*1j
        self.C = C[start:stop:incr, :, :]
        if self.verbose:
            print('Ready to perform polarization analysis!')

    def classify(self, svc: SupportVectorMachine = None):
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        df = pd.DataFrame(np.concatenate((eigenvectors[:, :, -1].real, eigenvectors[:, :, -1].imag), axis=1),
                          columns=['t1_real', 't2_real', 't3_real', 'r1_real', 'r2_real', 'r3_real',
                                    't1_imag', 't2_imag', 't3_imag', 'r1_imag', 'r2_imag', 'r3_imag'])

        df.loc[:] = eigenvectors[:, :, -1]

        test = 1



class EstimatorConfiguration:
    pass


class PolarizationModel:
    """
    Attributes:
        wave_type: WAVE TYPE
            'P' : P-wave
            'SV': SV-wave
            'SH': SH-wave
            'L' : Love-wave
            'R' : Rayleigh-wave
        vp: P-wave velocity (m/s) at the receiver location
        vs: S-wave velocity (m/s) at the receiver location
        vl: Love-wave velocity (m/s)
        vr: Rayleigh-wave velocity (m/s)
        theta: Inclination (degree), only for body waves
        xi: Ellipticity angle (rad) for Rayleigh waves
        v_scal: scaling velocity (m/s) to make translations dimensionless, Default: 1 (m/s)
        free_surface: True (default): the wave is recorded at the free surface, False: the wave is recorded inside the medium
        polarization: 6-C polarization vector (automatically computed from the other attributes)

    Methods:
        polarization_vector: Computes the polarization vector for the current instance of the class 'Wave'
        theta_rad: Outputs the inclination angle in rad
        phi_rad: Outputs the azimuth angle in rad
    """

    def __init__(self, wave_type=None, vp=None, vs=None, vl=None, vr=None, theta=None, phi=None, xi=None, v_scal=1,
                 free_surface=True, isnone=False):
        self.free_surface, self.wave_type = free_surface, wave_type
        self.vp, self.vs, self.vl, self.vr, self.theta, self.phi, self.xi = vp, vs, vl, vr, theta, phi, xi
        self.v_scal = v_scal
        self.isnone = isnone
        self.polarization = self.polarization_vector

    @property
    def polarization_vector(self):
        """
        Computes the six-component polarization vector for a specified wave-type
        at the free surface according to Equations 40 in Sollberger et al. (2018)

        wave: Instance of the class 'Wave'
        """
        if self.isnone:
            polarization = np.asmatrix(
                [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])
            return polarization

        if self.wave_type == 'P':
            if self.free_surface and 4 > self.vp / self.vs > 1.66:  # Exclude unphysical vp/vs ratios
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)

                v = (self.vp ** 2 - 2. * self.vs ** 2) / (2. * (self.vp ** 2 - self.vs ** 2))  # Poisson's ratio
                kappa = (2. * (1 - v) / (1 - 2 * v)) ** (1 / 2.)
                theta_s = np.arcsin((1 / kappa) * np.sin(theta_rad))  # angle of reflected S-wave

                # amplitude of reflected P-wave
                alpha_pp = (np.sin(2 * theta_rad) * np.sin(2 * theta_s) - kappa ** 2 * (np.cos(2 * theta_s)) ** 2) \
                           / (np.sin(2 * theta_rad) * np.sin(2 * theta_s) + kappa ** 2 * (np.cos(2 * theta_s)) ** 2)

                # amplitude of reflected S-wave
                alpha_ps = (2 * kappa * np.sin(2 * theta_rad) * np.cos(2 * theta_s)) \
                           / (np.sin(2 * theta_rad) * np.sin(2 * theta_s) + kappa ** 2 * (np.cos(2 * theta_s)) ** 2)

                v_x = -(np.sin(theta_rad) * np.cos(phi_rad)
                        + alpha_pp * np.sin(theta_rad) * np.cos(phi_rad)
                        + alpha_ps * np.cos(theta_s) * np.cos(phi_rad)) / self.v_scal
                v_y = -(np.sin(theta_rad) * np.sin(phi_rad)
                        + alpha_pp * np.sin(theta_rad) * np.sin(phi_rad)
                        + alpha_ps * np.cos(theta_s) * np.sin(phi_rad)) / self.v_scal
                v_z = -(np.cos(theta_rad)
                        - alpha_pp * np.cos(theta_rad)
                        + alpha_ps * np.sin(theta_s)) / self.v_scal
                w_x = (1 / 2.) * alpha_ps * np.sin(phi_rad) / self.vs
                w_y = -(1 / 2.) * alpha_ps * np.cos(phi_rad) / self.vs
                w_z = 0. * w_x

                polarization = np.asmatrix(
                    [np.complex(v_x), np.complex(v_y), np.complex(v_z), np.complex(w_x), np.complex(w_y),
                     np.complex(w_z)])
            elif not self.free_surface and self.vp / self.vs > 1.66:
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)
                v_x = - (1. / self.v_scal) * np.sin(theta_rad) * np.cos(phi_rad)
                v_y = - (1. / self.v_scal) * np.sin(theta_rad) * np.sin(phi_rad)
                v_z = - (1. / self.v_scal) * np.cos(theta_rad)
                w_x = 0.
                w_y = 0.
                w_z = 0.
                polarization = np.asmatrix(
                    [np.complex(v_x), np.complex(v_y), np.complex(v_z), np.complex(w_x), np.complex(w_y),
                     np.complex(w_z)])
            else:
                polarization = np.asmatrix(
                    [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])

        elif self.wave_type == 'SV':
            if self.free_surface and 4 > self.vp / self.vs > 1.66:
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)
                v = (self.vp ** 2 - 2. * self.vs ** 2) / (2. * (self.vp ** 2 - self.vs ** 2))  # poisson's ratio
                kappa = (2 * (1 - v) / (1 - 2 * v)) ** (1 / 2.)
                theta_crit = np.arcsin(1 / kappa)

                # Check whether the current incidence angle is at or above the critical angle
                if theta_rad == theta_crit:
                    theta_p = np.pi / 2.
                    alpha_sp = (4. * (kappa ** 2 - 1)) / (kappa * (2 - kappa ** 2))
                    alpha_ss = -1
                elif theta_crit < theta_rad < 0.9 * np.pi / 4:
                    # Incidence angles above the critical angle will yield a complex polarization
                    theta_p = np.arcsin(complex(np.sin(theta_rad) * self.vp / self.vs, 0))
                    alpha_ss = (4 * (np.sin(theta_rad) ** 2 - kappa ** (-2)) * np.sin(2 * theta_rad) ** 2 *
                                np.sin(theta_rad) ** 2
                                - np.cos(theta_rad) ** 4 + 4 * 1j * (np.sin(theta_rad) ** 2 - kappa ** -2) ** (
                                        1 / 2.) *
                                np.sin(2 * theta_rad) * np.sin(theta_rad) * (np.cos(2 * theta_rad)) ** 2) \
                               / (np.cos(2 * theta_rad) ** 4 + 4 * (np.sin(theta_rad) ** 2 - kappa ** -2) *
                                  np.sin(2 * theta_rad) ** 2 * np.sin(theta_rad) ** 2)
                    alpha_sp = (2 * kappa ** -1 * np.sin(2 * theta_rad) * np.cos(2 * theta_rad)
                                * (np.cos(2 * theta_rad) ** 2 - 2
                                   * 1j * (np.sin(theta_rad) ** 2 - kappa ** (-2)) ** (1 / 2.)
                                   * np.sin(2 * theta_rad) * np.sin(theta_rad))) / \
                               (np.cos(2 * theta_rad) ** 4 + 4 * (np.sin(theta_rad) ** 2 - kappa ** -2)
                                * np.sin(2 * theta_rad) ** 2 * np.sin(theta_rad) ** 2)

                elif theta_rad < theta_crit:

                    theta_p = np.arcsin(np.sin(theta_rad) * self.vp / self.vs)

                    alpha_ss = (np.sin(2 * theta_rad) * np.sin(2 * theta_p) - kappa ** 2 * (
                        np.cos(2 * theta_p)) ** 2) \
                               / (np.sin(2 * theta_rad) * np.sin(2 * theta_p) + kappa ** 2 * (
                        np.cos(2 * theta_rad)) ** 2)
                    alpha_sp = -(kappa * np.sin(4 * theta_rad)) \
                               / (np.sin(2 * theta_rad) * np.sin(2 * theta_p)
                                  + kappa ** 2 * (np.cos(2 * theta_rad)) ** 2)
                else:
                    theta_p = float("nan")
                    alpha_ss = float("nan")
                    alpha_sp = float("nan")

                v_x = (np.cos(theta_rad) * np.cos(phi_rad)
                       - alpha_ss * np.cos(theta_rad) * np.cos(phi_rad)
                       - alpha_sp * np.sin(theta_p) * np.cos(phi_rad)) / self.v_scal
                v_y = (np.cos(theta_rad) * np.sin(phi_rad)
                       - alpha_ss * np.cos(theta_rad) * np.sin(phi_rad)
                       - alpha_sp * np.sin(theta_p) * np.sin(phi_rad)) / self.v_scal
                v_z = -(np.sin(theta_rad)
                        + alpha_ss * np.sin(theta_rad)
                        - alpha_sp * np.cos(theta_p)) / self.v_scal

                w_x = (1 / 2.) * (1 + alpha_ss) * np.sin(phi_rad) / self.vs
                w_y = -(1 / 2.) * (1 + alpha_ss) * np.cos(phi_rad) / self.vs
                w_z = 0. * w_x
                polarization = np.asmatrix(
                    [np.complex(v_x), np.complex(v_y), np.complex(v_z), np.complex(w_x), np.complex(w_y),
                     np.complex(w_z)])
                if theta_rad > 0.9 * np.pi / 4:
                    polarization = np.asmatrix(
                        [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])
            elif not self.free_surface and 4 > self.vp / self.vs > 1.66:
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)
                v_x = (1. / self.v_scal) * np.cos(theta_rad) * np.cos(phi_rad)
                v_y = (1. / self.v_scal) * np.cos(theta_rad) * np.sin(phi_rad)
                v_z = -(1. / self.v_scal) * np.sin(theta_rad)
                w_x = (2 * self.vs) ** -1 * np.sin(phi_rad)
                w_y = - (2 * self.vs) ** -1 * np.cos(phi_rad)
                w_z = 0.
                polarization = np.asmatrix(
                    [np.complex(v_x), np.complex(v_y), np.complex(v_z), np.complex(w_x[0]), np.complex(w_y),
                     np.complex(w_z)])

            else:
                polarization = np.asmatrix(
                    [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])
        elif self.wave_type == 'SH':
            if self.free_surface:
                phi_rad = np.radians(self.phi)
                theta_rad = np.radians(self.theta)
                v_x = 2. / self.v_scal * np.sin(phi_rad)
                v_y = -2. / self.v_scal * np.cos(phi_rad)
                v_z = 0.
                w_x = 0.
                w_y = 0.
                w_z = 1. / self.vs * np.sin(theta_rad)
                polarization = np.asmatrix(
                    [np.complex(v_x), np.complex(v_y), np.complex(v_z), np.complex(w_x), np.complex(w_y),
                     np.complex(w_z)])
            else:
                phi_rad = np.radians(self.phi)
                theta_rad = np.radians(self.theta)
                v_x = (1. / self.v_scal) * np.sin(phi_rad)
                v_y = - (1. / self.v_scal) * np.cos(phi_rad)
                v_z = 0.
                w_x = - (2 * self.vs) ** -1 * np.cos(theta_rad) * np.cos(phi_rad)
                w_y = - (2 * self.vs) ** -1 * np.cos(theta_rad) * np.sin(phi_rad)
                w_z = - (2 * self.vs) ** -1 * np.sin(theta_rad)
                polarization = np.asmatrix(
                    [np.complex(v_x), np.complex(v_y), np.complex(v_z), np.complex(w_x[0]), np.complex(w_y),
                     np.complex(w_z)])

        elif self.wave_type == 'R':
            phi_rad = np.radians(self.phi)
            v_x = -1j * 1. / self.v_scal * np.sin(self.xi) * np.cos(phi_rad)
            v_y = -1j * 1. / self.v_scal * np.sin(self.xi) * np.sin(phi_rad)
            v_z = -1. / self.v_scal * np.cos(self.xi)

            w_x = 1. / self.vr * np.sin(phi_rad) * np.cos(self.xi)
            w_y = -1. / self.vr * np.cos(phi_rad) * np.cos(self.xi)
            w_z = 0.
            polarization = np.asmatrix(
                [np.complex(v_x), np.complex(v_y), np.complex(v_z), np.complex(w_x), np.complex(w_y),
                 np.complex(w_z)])
        elif self.wave_type == 'L':
            phi_rad = np.radians(self.phi)
            v_x = 1 / self.v_scal * np.sin(phi_rad)
            v_y = -1 / self.v_scal * np.cos(phi_rad)
            v_z = 0.

            w_x = 0.
            w_y = 0.
            w_z = 1. / (2 * self.vl)
            polarization = np.asmatrix(
                [np.complex(v_x), np.complex(v_y), np.complex(v_z), np.complex(w_x), np.complex(w_y),
                 np.complex(w_z)])

        else:
            sys.exit("Invalid wave type specified in 'Wave' object!")

        polarization = np.divide(polarization, np.linalg.norm(polarization))
        polarization = polarization.T
        return polarization
