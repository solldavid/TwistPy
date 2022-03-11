from __future__ import (absolute_import, division, print_function, unicode_literals)

import pickle
from builtins import *
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Trace
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from twistpy.estimator import EstimatorConfiguration
from twistpy.machinelearning import SupportVectorMachine


class TimeDomainAnalysis:
    """Time domain six-component polarization analysis.

    Single-station six degree-of-freedom polarization analysis in the time domain. Polarization analysis is performed
    in a sliding time window.

    .. note:: It is recommended to bandpass filter the data to a narrow frequency band before attempting a time-domain
       polarization analysis in order to avoid that dispersion effects  and wave type interference affect the
       polarization.

    Parameters
    ----------
    traN : :obj:`~obspy.core.trace.Trace`
        North component of translation
    traE : :obj:`~obspy.core.trace.Trace`
        East component of translation
    traZ : :obj:`~obspy.core.trace.Trace`
        Vertical component of translation
    rotN : :obj:`~obspy.core.trace.Trace`
        North component of rotation
    rotE : :obj:`~obspy.core.trace.Trace`
        East component of rotation
    rotZ : :obj:`~obspy.core.trace.Trace`
        Vertical component of rotation
    window : :obj:`dict`
        Window parameters defined as:

        |  window = {'window_length_seconds': :obj:`float`, 'overlap': :obj:`float`}
        |  Overlap should be on the interval 0 (no overlap between subsequent time windows) and 1
           (complete overlap, window is moved by 1 sample only).
    scaling_velocity : :obj:`float`, default=1.
        Scaling velocity (in m/s) to ensure numerical stability. The scaling velocity is
        applied to the translational data only (amplitudes are divided by scaling velocity) and ensures
        that both translation and rotation amplitudes are on the same order of magnitude and
        dimensionless. Ideally, v_scal is close to the S-Wave velocity at the receiver. After applying
        the scaling velocity, translation and rotation signals amplitudes should be similar.
    free_surface : :obj:`bool`, default=True
        Specify whether the recording station is located at the
        free surface in order to use the corresponding polarization models.
    verbose : :obj:`bool`, default=True
        Run in verbose mode.
    timeaxis : :obj:'str', default='utc'
        Specify whether the time axis of plots is shown in UTC (timeaxis='utc') or in seconds relative to the first
        sample (timeaxis='rel').

    Attributes
    ----------
    classification : :obj:`dict`
        Dictionary containing the labels of classified wave types at each position of the sliding time window. The
        dictionary has up to six entries corresponding to classifications for each eigenvector.

        | classification = {'0': list_with_classifications_of_first_eigenvector, '1':
            list_with_classification_of_second_eigenvector, ... , '5': list_with_classification_of_last_eigenvector}
    t_windows : :obj:`list` of :obj:`~obspy.core.utcdatetime.UTCDateTime`
        Window positions of the sliding time window on the time axis (center point of the window)
    C : :obj:`~numpy.ndarray` of :obj:`~numpy.complex128`
        Complex covariance matrices at each window position
    time : :obj:`list` of :obj:`~obspy.core.utcdatetime.UTCDateTime`
        Time axis of the input traces
    delta : :obj:`float`
        Sampling interval of the input data in seconds
    window_length_samples : :obj:`int`
        Window length in samples

    """

    def __init__(self, traN: Trace, traE: Trace, traZ: Trace, rotN: Trace,
                 rotE: Trace, rotZ: Trace, window: dict,
                 scaling_velocity: float = 1., free_surface: bool = True, verbose: bool = True,
                 timeaxis: str = 'utc') -> None:

        self.dop = None
        self.phi_r = None
        self.c_r = None
        self.xi = None
        self.phi_l = None
        self.c_l = None
        self.P = {'P': None, 'SV': None, 'R': None, 'L': None, 'SH': None}
        self.traN, self.traE, self.traZ, self.rotN, self.rotE, self.rotZ = traN, traE, traZ, rotN, rotE, rotZ
        self.timeaxis = timeaxis

        # Assert that input traces are ObsPy Trace objects
        assert isinstance(self.traN, Trace) and isinstance(self.traE, Trace) and isinstance(self.traZ, Trace) \
               and isinstance(self.rotN, Trace) and isinstance(self.rotE, Trace) and isinstance(self.rotZ, Trace), \
            "Input data must be objects of class obspy.core.Trace()!"

        # Assert that all traces have the same number of samples
        assert self.traN.stats.npts == self.traE.stats.npts and self.traN.stats.npts == self.traZ.stats.npts \
               and self.traN.stats.npts == self.rotN.stats.npts and self.traN.stats.npts == self.rotE.stats.npts \
               and self.traN.stats.npts == self.rotZ.stats.npts, "All six traces must have the same number of samples!"

        self.window = window
        if self.timeaxis == 'utc':
            self.time = self.traN.times(type="matplotlib")
        else:
            self.time = self.traN.times()
        self.scaling_velocity = scaling_velocity
        self.free_surface = free_surface
        self.verbose = verbose
        self.delta = self.traN.stats.delta
        self.window_length_samples = 2 * int((self.window['window_length_seconds'] / self.delta) / 2)
        self.classification: Dict[str, List[str]] = {'0': None, '1': None, '2': None, '3': None, '4': None, '5': None}
        start, stop, incr = [int(self.window_length_samples / 2),
                             -1 - int(self.window_length_samples / 2),
                             np.max([1, int((1 - self.window['overlap']) * self.window_length_samples)])]
        self.t_windows = self.time[start:stop:incr]
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
        svm : :obj:`~twistpy.MachineLearning.SupportVectorMachine`
            Trained support vector machine used for wave type classification
        eigenvector_to_classify : :obj:`int`, optional, default=0
            Integer value identifying the eigenvector that will be classified. The eigenvectors are
            sorted in descending order of their corresponding eigenvalue

            |  If 0: first eigenvector, corresponding to the dominant signal in
               the time window (associated with the largest eigenvalue).

        """
        if self.classification[str(eigenvector_to_classify)] is not None:
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

    def polarization_analysis(self, estimator_configuration: EstimatorConfiguration = None, plot: bool = True):
        r"""Perform polarization analysis.

        Parameters
        ----------
        estimator_configuration
        """
        if estimator_configuration is None:
            raise ValueError("Please provide an EstimatorConfiguration for polarization analysis!")

        # Classify wave types if this has not been done already.
        if estimator_configuration.use_ml_classification and \
                self.classification[str(estimator_configuration.eigenvector)] is None:
            self.classify(estimator_configuration.svm, estimator_configuration.eigenvector)

        if self.verbose:
            print('Computing wave parameters...')
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)

        # The eigenvectors are initially arbitrarily oriented in the complex plane, here we ensure that
        # the real and imaginary parts are orthogonal. See Samson (1980): Some comments on the descriptions of the
        # polarization states of waves, Geophysical Journal of the Royal Astronomical Society, Eqs. (3)-(5)
        u1 = eigenvectors[:, :, -(estimator_configuration.eigenvector + 1)]  # Select eigenvector for classification
        gamma = np.arctan2(2 * np.einsum('ij,ij->j', u1.real.T, u1.imag.T, optimize=True),
                           np.einsum('ij,ij->j', u1.real.T, u1.real.T, optimize=True) -
                           np.einsum('ij,ij->j', u1.imag.T, u1.imag.T, optimize=True))
        phi = - 0.5 * gamma
        eigenvectors = np.tile(np.exp(1j * phi), (6, 6, 1)).T * eigenvectors
        # Compute degree of polarization after Samson (1980): Some comments on the descriptions of the
        # polarization states of waves, Geophysical Journal of the Royal Astronomical Society, Eq. (18)
        self.dop = ((eigenvalues[:, 0] - eigenvalues[:, 1]) ** 2
                    + (eigenvalues[:, 0] - eigenvalues[:, 2]) ** 2
                    + (eigenvalues[:, 0] - eigenvalues[:, 3]) ** 2
                    + (eigenvalues[:, 0] - eigenvalues[:, 4]) ** 2
                    + (eigenvalues[:, 0] - eigenvalues[:, 5]) ** 2
                    + (eigenvalues[:, 1] - eigenvalues[:, 2]) ** 2
                    + (eigenvalues[:, 1] - eigenvalues[:, 3]) ** 2
                    + (eigenvalues[:, 1] - eigenvalues[:, 4]) ** 2
                    + (eigenvalues[:, 1] - eigenvalues[:, 5]) ** 2
                    + (eigenvalues[:, 2] - eigenvalues[:, 3]) ** 2
                    + (eigenvalues[:, 2] - eigenvalues[:, 4]) ** 2
                    + (eigenvalues[:, 2] - eigenvalues[:, 5]) ** 2
                    + (eigenvalues[:, 3] - eigenvalues[:, 4]) ** 2
                    + (eigenvalues[:, 3] - eigenvalues[:, 5]) ** 2
                    + (eigenvalues[:, 4] - eigenvalues[:, 5]) ** 2) / (5 * np.sum(eigenvalues, axis=-1) ** 2)

        # Estimate wave parameters directly from the specified eigenvector if method=='ML'
        if estimator_configuration.method == 'ML':
            for wave_type in estimator_configuration.wave_types:
                indices = self.classification[str(estimator_configuration.eigenvector)] == wave_type
                eigenvector_wtype = eigenvectors[indices, :, -(estimator_configuration.eigenvector + 1)]
                if wave_type == 'L':
                    self.phi_l = np.empty_like(self.t_windows)
                    self.phi_l[:] = np.nan
                    self.c_l = np.empty_like(self.t_windows)
                    self.c_l[:] = np.nan
                    eigenvector_wtype[np.linalg.norm(np.abs(eigenvector_wtype[:, 0:2].real), axis=1) <
                                      np.linalg.norm(np.abs(eigenvector_wtype[:, 0:2].imag), axis=1)] = \
                        eigenvector_wtype[np.linalg.norm(np.abs(eigenvector_wtype[:, 0:2].real), axis=1) <
                                          np.linalg.norm(np.abs(eigenvector_wtype[:, 0:2].imag), axis=1)].conj() * 1j
                    phi_love = np.arctan2(np.real(eigenvector_wtype[:, 1]), np.real(eigenvector_wtype[:, 0]))
                    a_t = np.cos(phi_love) * eigenvector_wtype[:, 0] + np.sin(phi_love) * eigenvector_wtype[:, 1]
                    phi_love += np.pi / 2
                    phi_love[np.sign(eigenvector_wtype[:, 0].real) == np.sign(eigenvector_wtype[:, 1])] += np.pi
                    phi_love[phi_love > 2 * np.pi] -= 2 * np.pi
                    phi_love[phi_love < 0] += 2 * np.pi
                    # Love wave velocity: transverse acceleration divided by 2*vertical rotation
                    c_love = self.scaling_velocity * np.abs(a_t.real) / np.abs(eigenvector_wtype[:, 5].real) / 2
                    self.c_l[indices] = c_love
                    self.phi_l[indices] = np.degrees(phi_love)
                elif wave_type == 'R':
                    self.xi = np.empty_like(self.t_windows)
                    self.xi[:] = np.nan
                    self.c_r = np.empty_like(self.t_windows)
                    self.c_r[:] = np.nan
                    self.phi_r = np.empty_like(self.t_windows)
                    self.phi_r[:] = np.nan
                    eigenvector_wtype[np.linalg.norm(np.abs(eigenvector_wtype[:, 0:2].real), axis=1) >
                                      np.linalg.norm(np.abs(eigenvector_wtype[:, 0:2].imag), axis=1)] = \
                        eigenvector_wtype[np.linalg.norm(np.abs(eigenvector_wtype[:, 0:2].real), axis=1) >
                                          np.linalg.norm(np.abs(eigenvector_wtype[:, 0:2].imag), axis=1)].conj() * 1j
                    eigenvector_wtype[eigenvector_wtype[:, 2] < 0, :] *= -1  # Ensure that eigenvectors point into
                    # the same direction with translational z-component positive
                    phi_rayleigh = np.arctan2(np.imag(eigenvector_wtype[:, 1]), np.imag(eigenvector_wtype[:, 0]))
                    phi_rayleigh[phi_rayleigh < 0] += np.pi
                    # Compute radial translational component t_r and transverse rotational component r_t
                    r_t = -np.sin(phi_rayleigh) * eigenvector_wtype[:, 3].real + np.cos(phi_rayleigh) * \
                          eigenvector_wtype[:, 4].real
                    t_r = np.cos(phi_rayleigh) * eigenvector_wtype[:, 0].imag + np.sin(phi_rayleigh) * \
                          eigenvector_wtype[:, 1].imag
                    # Account for 180 degree ambiguity by evaluating signs
                    phi_rayleigh[np.sign(t_r) < np.sign(r_t)] += np.pi
                    phi_rayleigh[(np.sign(t_r) > 0) & (np.sign(r_t) > 0)] += np.pi

                    # Compute Rayleigh wave ellipticity angle
                    elli_rayleigh = -np.arctan(t_r / eigenvector_wtype[:, 2].real)
                    elli_rayleigh[np.sign(t_r) == np.sign(r_t)] *= -1

                    # Compute Rayleigh wave phase velocity
                    c_rayleigh = self.scaling_velocity * np.abs(eigenvector_wtype[:, 2].real) / np.abs(r_t)
                    self.xi[indices] = np.degrees(elli_rayleigh)
                    self.c_r[indices] = c_rayleigh
                    self.phi_r[indices] = np.degrees(phi_rayleigh)

        else:
            for wave_type in estimator_configuration.wave_types:
                if estimator_configuration.use_ml_classification:
                    indices = self.classification[str(estimator_configuration.eigenvector)] == wave_type
                else:
                    indices = np.ones_like(self.t_windows).astype('bool')
            steering_vectors = estimator_configuration.compute_steering_vectors(wave_type)

            if estimator_configuration.method == 'MUSIC':
                noise_space_vectors = \
                    eigenvectors[indices, :, :6 - estimator_configuration.music_signal_space_dimension]
                noise_space_vectors_H = np.transpose(noise_space_vectors.conj(), axes=(0, 2, 1))
                noise_space = np.einsum('...ij, ...jk->...ik', noise_space_vectors, noise_space_vectors_H,
                                        optimize=True)
                P = 1 / np.einsum('...sn,...nk,...sk->...s', steering_vectors.conj().T, noise_space, steering_vectors.T,
                                  optimize=True).real
            elif estimator_configuration.method == 'MVDR':
                P = 1 / np.einsum('...sn,...nk,...sk->...s', steering_vectors.conj().T,
                                  np.linalg.pinv(self.C[indices, :, :], rcond=1e-6, hermitian=True),
                                  steering_vectors.T, optimize=True).real
            elif estimator_configuration.method == 'BARTLETT':
                P = np.einsum('...sn,...nk,...sk->...s', steering_vectors.conj().T,
                              self.C[indices, :, :], steering_vectors.T, optimize=True).real
        if plot:
            self.plot_polarization_analysis(estimator_configuration=estimator_configuration)

    def plot_polarization_analysis(self, estimator_configuration: EstimatorConfiguration,
                                   dop_clip: np.float = 0):
        wave_types = estimator_configuration.wave_types
        method = estimator_configuration.method
        if isinstance(wave_types, str):
            wave_types = [wave_types]
        for wave_type in wave_types:
            if wave_type == 'L':
                assert self.c_l is not None, 'No polarization attributes for Love waves have been computed so far!'
                if method == 'ML':
                    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
                    self._plot_seismograms(ax[0])
                    phi_l = self.phi_l
                    phi_l[self.dop < dop_clip] = np.nan
                    c_l = self.c_l
                    c_l[self.dop < dop_clip] = np.nan
                    dop = self.dop
                    ax[1].plot(self.t_windows, phi_l, 'k.')
                    ax[1].set_ylabel('Degrees')
                    ax[1].set_title('Back-azimuth')
                    ax[2].plot(self.t_windows, c_l, 'k.')
                    ax[2].set_title('Phase velocity')
                    ax[2].set_ylabel('m/s')
                    ax[3].plot(self.t_windows, dop, 'k.')
                    ax[3].set_title('Degree of polarization')
                    ax[3].set_ylabel('DOP')
                    ax[3].set_xlabel('Time (UTC)')
                    fig.suptitle('Love wave analysis')
            elif wave_type == 'R':
                assert self.c_r is not None, 'No polarization attributes for Rayleigh waves have been computed so far!'
                if method == 'ML':
                    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
                    self._plot_seismograms(ax[0])
                    phi_r = self.phi_r
                    phi_r[self.dop < dop_clip] = np.nan
                    c_r = self.c_r
                    c_r[self.dop < dop_clip] = np.nan
                    xi = self.xi
                    xi[self.dop < dop_clip] = np.nan
                    dop = self.dop
                    ax[1].plot(self.t_windows, phi_r, 'k.')
                    ax[1].set_ylabel('Degrees')
                    ax[1].set_title('Back-azimuth')
                    ax[2].plot(self.t_windows, c_r, 'k.')
                    ax[2].set_title('Phase velocity')
                    ax[2].set_ylabel('m/s')
                    ax[3].plot(self.t_windows, xi, 'k.')
                    ax[3].set_ylabel('Degrees')
                    ax[3].set_title('Ellipticity angle')
                    ax[4].plot(self.t_windows, dop, 'k.')
                    ax[4].set_ylabel('DOP')
                    ax[4].set_title('Degree of polarization')
                    ax[4].set_xlabel('Time (UTC)')
                    fig.suptitle('Rayleigh wave analysis')
        plt.show()

    def _plot_seismograms(self, ax: plt.Axes):
        if self.timeaxis == 'utc':
            time = self.traN.times(type='matplotlib')
        else:
            time = self.traN.times()
        ax.plot(time, self.traN.data, 'k:', label='traN')
        ax.plot(time, self.traE.data, 'k--', label='traE')
        ax.plot(time, self.traZ.data, 'k', label='traZ')
        ax.plot(time, self.rotZ.data, 'r:', label='rotN')
        ax.plot(time, self.rotE.data, 'r--', label='rotE')
        ax.plot(time, self.rotZ.data, 'r', label='rotZ')
        if self.timeaxis == 'utc':
            ax.xaxis_date()
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def save(self, name: str) -> None:
        """ Save the current TimeDomainAnalysis object to a file on the disk in the current working directory.

         Include absolute path if you want to save in a different directory.

        name : :obj:`str`
            File name
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string!")

        fid = open(name, 'wb')
        pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL)
        fid.close()
