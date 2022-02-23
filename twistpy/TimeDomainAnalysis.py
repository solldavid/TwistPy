"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2022
:license:
    None
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import pickle
from builtins import *
from typing import List, Dict

import numpy as np
import pandas as pd
from obspy import Trace
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from twistpy.MachineLearning import SupportVectorMachine


class TimeDomainAnalysis:
    """Time domain polarization analysis.

    Single-station six degree-of-freedom polarization analysis in the time domain. Polarization analysis is performed
    in a sliding time window.

    .. note:: It is recommended to bandpass filter the data to a narrow frequency band before attempting a time-domain
       polarization analysis in order to avoid that dispersion effects  and wave type interference affect the
       polarization.

    Parameters
    ----------
    traN : :obj:`obspy.core.trace.Trace`
        North component of translation
    traE : :obj:`obspy.core.trace.Trace`
        East component of translation
    traZ : :obj:`obspy.core.trace.Trace`
        Vertical component of translation
    rotN : :obj:`obspy.core.trace.Trace`
        North component of rotation
    rotE : :obj:`obspy.core.trace.Trace`
        East component of rotation
    rotZ : :obj:`obspy.core.trace.Trace`
        Vertical component of rotation
    window : :obj:`dict`
        Window parameters defined as:

        |  window = {'window_length_seconds': :obj:`float`, 'overlap': :obj:`float`}

        |  Overlap should be on the interval 0 (no overlap between subsequent time windows) and 1
           (complete overlap, window is moved by 1 sample only).
    scaling_velocity : :obj:`float`, optional
        Scaling velocity (in m/s) to ensure numerical stability. The scaling velocity is
        applied to the translational data only (amplitudes are divided by scaling velocity) and ensures
        that both translation and rotation amplitudes are on the same order of magnitude and
        dimensionless. Ideally, v_scal is close to the S-Wave velocity at the receiver. After applying
        the scaling velocity, translation and rotation signals amplitudes should be similar. Defaults to 1.
    free_surface : :obj:`bool`, optional
        True (Default) or False. Specifies whether the recording station is located at the
        free surface and uses the corresponding polarization models.
    verbose : :obj:`bool`, optional
        True (Default) to run in verbose mode.
    """

    def __init__(self, traN: Trace = None, traE: Trace = None, traZ: Trace = None, rotN: Trace = None,
                 rotE: Trace = None, rotZ: Trace = None, window: dict = None,
                 scaling_velocity: float = 1., free_surface: bool = True, verbose: bool = True) -> None:

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
        self.time = self.traN.times(type="utcdatetime")
        self.scaling_velocity = scaling_velocity
        self.free_surface = free_surface
        self.verbose = verbose
        self.delta = self.traN.stats.delta
        self.window_length_samples = 2 * int((self.window['window_length_seconds'] / self.delta) / 2)
        self.classification: Dict[str, List[str]] = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []}
        start, stop, incr = [int(self.window_length_samples / 2),
                             -1 - int(self.window_length_samples / 2),
                             np.max([1, int((1 - self.window['overlap']) * self.window_length_samples)])]
        self.t_window = self.time[start:stop:incr]
        # Compute the analytic signal
        u: np.ndarray = np.array([hilbert(self.traN).T, hilbert(self.traE).T, hilbert(self.traZ).T,
                                  hilbert(self.rotN).T, hilbert(self.rotE).T, hilbert(self.rotZ).T]).T
        # Compute covariance matrices
        if self.verbose:
            print('Computing covariance matrices...')
        C: np.ndarray = np.einsum('...i,...j->...ij', np.conj(u), u).astype('complex')
        for j in range(C.shape[2]):
            for k in range(C.shape[1]):
                C[..., j, k] = \
                    uniform_filter1d(C[..., j, k], size=self.window_length_samples)
        self.C = C[start:stop:incr, :, :]
        if self.verbose:
            print('Covariance matrices computed!')

    def classify(self, svm: SupportVectorMachine, eigenvector_to_classify: int = 0) -> None:
        """Classify wave types using a support vector machine

        Parameters
        ----------
        svm : :obj:`twistpy.machinelearning.SupportVectorMachine`
            Trained support vector machine used for wave type classification
        eigenvector_to_classify : :obj:`int`, optional
            (Default = 0) Integer value identifying the eigenvector that will be classified, where the eigenvectors are
            sorted in descending order of their corresponding eigenvalue

            |  If 0: first or principal eigenvector, corresponding to the dominant signal in
               the time window (associated with the largest eigenvalue)

        """
        if self.classification[str(eigenvector_to_classify)]:
            print(f"Wave types are already classified for eigenvector with number '{eigenvector_to_classify}'! "
                  f"Classification will not be run a second time. To access the previous classification, "
                  f"check the attribute TimeDomainAnalysis.classification['{eigenvector_to_classify}']")
            return

        if self.verbose:
            print('Performing eigen-decomposition of covariance matrices...')
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)

        # The eigenvectors are initially arbitrarily oriented in the complex plane, here we ensure that
        # the real and imaginary parts are orthogonal. See Samson (1980): Some comments on the descriptions of the
        # polarization states of waves, Geophysical Journal of the Royal Astronomical Society, Eqs. (3)-(5)
        u1 = eigenvectors[:, :, -(eigenvector_to_classify + 1)]  # Select eigenvector for classification
        gamma = np.arctan2(2 * np.einsum('ij,ij->j', u1.real.T, u1.imag.T, optimize=True),
                           np.einsum('ij,ij->j', u1.real.T, u1.real.T, optimize=True) -
                           np.einsum('ij,ij->j', u1.imag.T, u1.imag.T, optimize=True))
        phi = - 0.5 * gamma
        eigenvectors = np.tile(np.exp(1j * phi), (6, 6, 1)).T * eigenvectors
        if self.verbose:
            print('Eigenvectors and eigenvalues have been computed!')
        df = pd.DataFrame(np.concatenate(
            (eigenvectors[:, :, -(eigenvector_to_classify + 1)].real,
             eigenvectors[:, :, -(eigenvector_to_classify + 1)].imag),
            axis=1),
            columns=['t1_real', 't2_real', 't3_real', 'r1_real', 'r2_real', 'r3_real',
                     't1_imag', 't2_imag', 't3_imag', 'r1_imag', 'r2_imag', 'r3_imag'])
        model = svm.load_model()
        if self.verbose:
            print('Wave type classification in progress...')
        wave_types = model.predict(df)
        self.classification[str(eigenvector_to_classify)] = wave_types
        if self.verbose:
            print('Wave types have been classified!')

    def save(self, name: str) -> None:
        """ Save the current TimeDomainAnalysis object to a file on the disk in the current working directory.

        name : :obj:`str`
            File name
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string!")

        fid = open(name, 'wb')
        pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL)
        fid.close()
