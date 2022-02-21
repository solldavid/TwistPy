"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2022
:license:
    None
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

from builtins import *

import numpy as np
import pandas as pd
from obspy import Trace
from scipy.signal import hanning, convolve, hilbert

from twistpy.machinelearning import SupportVectorMachine


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
               and isinstance(self.rotN, Trace) and isinstance(self.rotE, Trace) and isinstance(self.rotZ, Trace), \
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
        u: np.ndarray = np.array([hilbert(self.traN).T, hilbert(self.traE).T, hilbert(self.traZ).T,
                                  hilbert(self.rotN).T, hilbert(self.rotE).T, hilbert(self.rotZ).T]).T
        # Compute covariance matrices
        if self.verbose:
            print('Computing covariance matrices...')
        C: np.ndarray = np.einsum('...i,...j->...ij', np.conj(u), u).astype('complex')
        w = hanning(self.window_length_samples + 2)[1:-1]  # Hann window for covariance matrix averaging
        w /= np.sum(w)
        for j in range(C.shape[2]):
            for k in range(C.shape[1]):
                C[..., j, k] = \
                    convolve(C[..., j, k].real, w, mode='same') + convolve(C[..., j, k].imag, w, mode='same') * 1j
        self.C = C[start:stop:incr, :, :]
        if self.verbose:
            print('Covariance matrices computed!')

    def classify(self, svm: SupportVectorMachine = None):
        if self.verbose:
            print('Performing eigendecomposition of covariance matrices...')
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        if self.verbose:
            print('Eigenvectors and eigenvalues have been computed!')
        df = pd.DataFrame(np.concatenate((eigenvectors[:, :, -1].real, eigenvectors[:, :, -1].imag), axis=1),
                          columns=['t1_real', 't2_real', 't3_real', 'r1_real', 'r2_real', 'r3_real',
                                   't1_imag', 't2_imag', 't3_imag', 'r1_imag', 'r2_imag', 'r3_imag'])

        model = svm.load_model()
        if self.verbose:
            print('Wave type classification in progress...')
        wave_types = model.predict(df)
        if self.verbose:
            print('Wave types have been classified!')
        test = 1


class EstimatorConfiguration:
    pass
