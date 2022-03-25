from __future__ import (absolute_import, division, print_function, unicode_literals)

import pickle
from builtins import *
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Trace, Stream
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from twistpy.polarization.estimator import EstimatorConfiguration
from twistpy.polarization.machinelearning import SupportVectorMachine


class TimeDomainAnalysis6C:
    """Time domain six-component polarization analysis.

    Single-station six degree-of-freedom polarization analysis in the time domain. Polarization analysis is performed
    in a sliding time window using the complex analytic signal [1].

    [1] Sollberger et al. (2018). *6-C polarization analysis using point measurements of translational and rotational
    ground-motion: theory and applications*, Geophysical Journal International, 213(1),
    https://doi.org/10.1093/gji/ggx542

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
    timeaxis : :obj:`str`, default='utc'
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
    dop : :obj:`numpy.ndarray` of :obj:`float`
        Degree of polarization estimated at each window position.

        .. hint::
                The definition of the degree of polarization follows the one from Samson & Olson (1980): *Some comments
                on the descriptions of the polarization states of waves*, Geophysical Journal of the Royal Astronomical
                Society, https://doi.org/10.1111/j.1365-246X.1980.tb04308.x. It is defined as:

                .. math::
                    P^2=\sum_{j,k=1}^{n}(\lambda_j-\lambda_k)^2/[2(n-1)(\sum_{j=1}^{n}(\lambda_j)^2)]

                where :math:`P^2` is the degree of polarization and :math:`\lambda_j` are the eigenvalues of the
                covariance matrix.
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

    def polarization_analysis(self, estimator_configuration: EstimatorConfiguration = None, plot: bool = False):
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
                    P = 1 / np.einsum('...sn,...nk,...sk->...s', steering_vectors.conj().T, noise_space,
                                      steering_vectors.T, optimize=True).real
                elif estimator_configuration.method == 'MVDR':
                    P = 1 / np.einsum('...sn,...nk,...sk->...s', steering_vectors.conj().T,
                                      np.linalg.pinv(self.C[indices, :, :], rcond=1e-6, hermitian=True),
                                      steering_vectors.T, optimize=True).real
                elif estimator_configuration.method == 'BARTLETT':
                    P = np.einsum('...sn,...nk,...sk->...s', steering_vectors.conj().T,
                                  self.C[indices, :, :], steering_vectors.T, optimize=True).real
        if plot:
            self.plot(estimator_configuration=estimator_configuration)

    def plot(self, estimator_configuration: EstimatorConfiguration,
             dop_clip: float = 0):
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
                    if self.timeaxis == 'utc':
                        ax[1].xaxis_date()
                        ax[2].xaxis_date()
                        ax[3].xaxis_date()
                        ax[3].set_xlabel('Time (UTC)')
                    else:
                        ax[3].set_xlabel('Time (s)')
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
                    if self.timeaxis == 'utc':
                        ax[1].xaxis_date()
                        ax[2].xaxis_date()
                        ax[3].xaxis_date()
                        ax[4].xaxis_date()
                        ax[4].set_xlabel('Time (UTC)')
                    else:
                        ax[4].set_xlabel('Time (s)')
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


class TimeDomainAnalysis3C:
    """Time domain three-component polarization analysis.

    Single-station three-component polarization analysis in the time domain. Polarization analysis is performed
    in a sliding time window using the complex analytic signal [1].

    [1] Vidale, J. E. (1986). *Complex polarization analysis of particle motion*, BSSA, 76(5),
    https://doi.org/10.1785/BSSA0760051393

    .. note:: It is recommended to bandpass filter the data to a narrow frequency band before attempting a time-domain
       polarization analysis in order to avoid that dispersion effects  and wave type interference affect the
       polarization.

    Parameters
    ----------
    N : :obj:`~obspy.core.trace.Trace`
     North component seismogram
    E : :obj:`~obspy.core.trace.Trace`
     East component seismogram
    Z : :obj:`~obspy.core.trace.Trace`
        Vertical component seismogram
    window : :obj:`dict`
        Window parameters defined as:

        |  window = {'window_length_seconds': :obj:`float`, 'overlap': :obj:`float`}
        |  Overlap should be on the interval 0 (no overlap between subsequent time windows) and 1
           (complete overlap, window is moved by 1 sample only).
    verbose : :obj:`bool`, default=True
        Run in verbose mode.
    timeaxis : :obj:`str`, default='utc'
        Specify whether the time axis of plots is shown in UTC (timeaxis='utc') or in seconds relative to the first
        sample (timeaxis='rel').

    Attributes
    ----------
    t_windows : :obj:`list` of :obj:`~obspy.core.utcdatetime.UTCDateTime`
        Window positions of the sliding time window on the time axis (center point of the window)
    C : :obj:`~numpy.ndarray` of :obj:`~numpy.complex128`
        Complex covariance matrices at each window position
    time : :obj:`list` of :obj:`~obspy.core.utcdatetime.UTCDateTime`
        Time axis of the input traces
    delta : :obj:`float`
        Sampling interval of the input data in seconds
    window_length_samples : :obj:`int`
        Window length in samples.
    dop : :obj:`numpy.ndarray` of :obj:`float`
        Degree of polarization estimated at each window position.

        .. hint::
                The definition of the degree of polarization follows the one from Samson & Olson (1980): *Some comments
                on the descriptions of the polarization states of waves*, Geophysical Journal of the Royal Astronomical
                Society, https://doi.org/10.1111/j.1365-246X.1980.tb04308.x. It is defined as:

                .. math::
                    P^2=\sum_{j,k=1}^{n}(\lambda_j-\lambda_k)^2/[2(n-1)(\sum_{j=1}^{n}(\lambda_j)^2)]

                where :math:`P^2` is the degree of polarization and :math:`\lambda_j` are the eigenvalues of the
                covariance matrix.
    elli : :obj:`numpy.ndarray` of :obj:`float`
        Ellipticity estimated at each window position
    inc1 : :obj:`numpy.ndarray` of :obj:`float`
        Inclination of the major semi-axis of the polarization ellipse estimated at each window position
    inc2 : :obj:`numpy.ndarray` of :obj:`float`
        Inclination of the minor semi-axis of the polarization ellipse estimated at each window position
    azi1 : :obj:`numpy.ndarray` of :obj:`float`
        Azimuth of the major semi-axis of the polarization ellipse estimated at each window position
    azi2 : :obj:`numpy.ndarray` of :obj:`float`
        Azimuth of the minor semi-axis of the polarization ellipse estimated at each window position
    """

    def __init__(self, N: Trace, E: Trace, Z: Trace, window: dict, verbose: bool = True,
                 timeaxis: str = 'utc') -> None:

        self.dop = None
        self.elli = None
        self.inc1 = None
        self.inc2 = None
        self.azi1 = None
        self.azi2 = None
        self.N, self.E, self.Z = N, E, Z
        self.timeaxis = timeaxis

        # Assert that input traces are ObsPy Trace objects
        assert isinstance(self.N, Trace) and isinstance(self.E, Trace) and isinstance(self.Z, Trace), \
            "Input data must be objects of class obspy.core.Trace()!"

        # Assert that all traces have the same number of samples
        assert self.N.stats.npts == self.E.stats.npts and self.N.stats.npts == self.Z.stats.npts, \
            "All three traces must have the same number of samples!"

        self.window = window
        if self.timeaxis == 'utc':
            self.time = self.N.times(type="matplotlib")
        else:
            self.time = self.N.times()
        self.verbose = verbose
        self.delta = self.N.stats.delta
        self.window_length_samples = 2 * int((self.window['window_length_seconds'] / self.delta) / 2)
        start, stop, incr = [int(self.window_length_samples / 2),
                             -1 - int(self.window_length_samples / 2),
                             np.max([1, int((1 - self.window['overlap']) * self.window_length_samples)])]
        self.t_windows = self.time[start:stop:incr]
        # Compute the analytic signal
        u: np.ndarray = np.array([hilbert(self.N).T, hilbert(self.E).T, hilbert(self.Z).T]).T
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
        self.polarization_analysis()

    def polarization_analysis(self):
        r"""Perform polarization analysis.
        """

        if self.verbose:
            print('Computing polarization attributes...')
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)

        # The eigenvectors are initially arbitrarily oriented in the complex plane, here we ensure that
        # the real and imaginary parts are orthogonal. See Samson (1980): Some comments on the descriptions of the
        # polarization states of waves, Geophysical Journal of the Royal Astronomical Society, Eqs. (3)-(5)
        u1 = eigenvectors[:, :, -1]  # Select principal eigenvector
        gamma = np.arctan2(2 * np.einsum('ij,ij->j', u1.real.T, u1.imag.T, optimize=True),
                           np.einsum('ij,ij->j', u1.real.T, u1.real.T, optimize=True) -
                           np.einsum('ij,ij->j', u1.imag.T, u1.imag.T, optimize=True))
        phi = - 0.5 * gamma
        eigenvectors = np.tile(np.exp(1j * phi), (3, 3, 1)).T * eigenvectors
        # Compute degree of polarization after Samson (1980): Some comments on the descriptions of the
        # polarization states of waves, Geophysical Journal of the Royal Astronomical Society, Eq. (18)
        self.dop = ((eigenvalues[:, 0] - eigenvalues[:, 1]) ** 2
                    + (eigenvalues[:, 0] - eigenvalues[:, 2]) ** 2
                    + (eigenvalues[:, 1] - eigenvalues[:, 2]) ** 2) \
                   / (2 * np.sum(eigenvalues, axis=-1) ** 2)
        self.azi1 = np.degrees(np.pi / 2 - np.arctan2(eigenvectors[:, 0, -1].real, eigenvectors[:, 1, -1].real))
        self.azi2 = np.degrees(np.pi / 2 - np.arctan2(eigenvectors[:, 0, -1].imag, eigenvectors[:, 1, -1].imag))
        self.azi1[self.azi1 < 0] += 180
        self.azi2[self.azi2 < 0] += 180
        self.azi1[self.azi1 > 180] -= 180
        self.azi2[self.azi2 > 180] -= 180
        self.inc1 = np.degrees(np.arctan2(np.linalg.norm(eigenvectors[:, 0:2, -1].real, axis=1),
                                          np.abs(eigenvectors[:, 2, -1].real)))
        self.inc2 = np.degrees(np.arctan2(np.linalg.norm(eigenvectors[:, 0:2, -1].imag, axis=1),
                                          np.abs(eigenvectors[:, 2, -1].imag)))
        self.elli = np.linalg.norm(eigenvectors[:, :, -1].imag, axis=1) / np.linalg.norm(eigenvectors[:, :, -1].real,
                                                                                         axis=1)
        if self.verbose:
            print('Polarization attributes have been computed!')

    def filter(self, plot_filtered_attributes: bool = False, **kwargs):
        """
        Parameters
        ----------
        plot_filtered_attributes : :obj:`bool`, default = False
            If set to True, a plot will be generated that shows the polarization attributes after filtering
        **kwargs : For example elli=[0, 0.3]
            Polarization attributes used for filtering. The filter parameters are specified as a list with two entries,
            specifying the range of the polarization attributes that are kept by the filter. In the above example the
            filter would only retain all signal with an ellipticity between 0 and 0.3 and suppress all signal
            with an ellipticity larger than 0.3. Multiple polarization attributes can be specified.

            .. hint:: Supported polariziation attributes are:

                dop (degree of polarization)

                elli (Ellipticity)

                inc1 (Incidence angle of major semi axis)

                inc2 (Incidence angle of minor semi axis)

                azi1 (Azimuth of the major semi axis)

                azi2 (Azimuth of the minor semi axis)

        Returns
        -------
        data_filtered : :obj:`~obspy.core.stream.Stream`
            Filtered data. The order of traces in the stream is N, E, Z.
        """

        if self.window['overlap'] != 1.:
            raise Exception("Polarization filtering is only supported if the polarization attributes have been computed"
                            "at each sample. To do so, recompute the polarization attributes by setting the overlap of"
                            "adjacent time windows to 1. (window['overlap']=1.)")

        params = {}
        for k in kwargs:
            if kwargs[k] is not None:
                params[k] = kwargs[k]
        if not params:
            raise Exception("No filter values provided!")

        if self.elli is None:
            raise Exception('No polarization attributes computed yet!')

        # Compute eigenvectors for projection
        start, stop = int(self.window_length_samples / 2), -1 - int(self.window_length_samples / 2)
        eigenvectors = np.zeros((self.N.stats.npts, 3, 3)).astype('complex')
        _, eigenvectors[start:stop, :, :] = np.linalg.eigh(self.C)

        # Initialize filter mask
        mask = np.ones((self.N.stats.npts,))
        # Populate filter mask (will be 1 where signal is kept and 0 everywhere else)
        for parameter in params:
            pol_attr = np.empty_like(mask)
            pol_attr[:] = np.nan
            pol_attr[start:stop] = np.real(getattr(self, parameter))
            if parameter == 'dop' or parameter == 'elli' and params[parameter][1] == 1.:
                alpha = pol_attr >= params[parameter][0]
            else:
                alpha = ((pol_attr >= params[parameter][0]) &
                         (pol_attr <= params[parameter][1])).astype('int')
            mask *= alpha

        mask = uniform_filter1d(mask, size=self.window_length_samples, axis=0)
        indx = mask > 0

        Z_sep = np.zeros_like(self.Z.data)
        N_sep = np.zeros_like(self.N.data)
        E_sep = np.zeros_like(self.E.data)

        N_hilbert, E_hilbert, Z_hilbert = hilbert(self.N.data), hilbert(self.E.data), hilbert(self.Z.data)
        data_hilbert = np.array([N_hilbert[indx].ravel(), E_hilbert[indx].ravel(), Z_hilbert[indx].ravel()]).T
        # Project data into coordinate frame spanned by eigenvectors
        data_proj = np.einsum('...i, ...ij -> ...j', data_hilbert, eigenvectors[indx, :, :], optimize=True)

        # Only keep data that is aligned with the principal eigenvector
        data_proj[:, 0:2] = 0

        # Back-projection into original coordinate frame
        data_filt = np.einsum('...i, ...ij -> ...j', data_proj, np.transpose(eigenvectors[indx.ravel(), :, :].conj(),
                                                                             axes=(0, 2, 1)), optimize=True)
        Z_sep[indx] = data_filt[:, 2].real
        N_sep[indx] = data_filt[:, 0].real
        E_sep[indx] = data_filt[:, 1].real

        Z_sep = mask * Z_sep
        N_sep = mask * N_sep
        E_sep = mask * E_sep
        #
        data_filtered = Stream(traces=[self.N.copy(), self.E.copy(), self.Z.copy()])
        data_filtered[0].data = N_sep
        data_filtered[1].data = E_sep
        data_filtered[2].data = Z_sep

        if plot_filtered_attributes:
            self.plot(show=False, alpha=mask[start:stop], seismograms=data_filtered)

        return data_filtered

    def plot(self, show: bool = True, alpha: np.ndarray = None, seismograms: Stream = None) -> None:
        """Plot polarization analysis.

        Parameters
        ----------
        show : :obj:`bool`, default=True
            Display the plot directly after plotting. If set to False, the plot will only show up once
            :func:`~matplotlib.pyplot.show` is called.
        alpha : :obj:`numpy.ndarray`, default = None
            A mask (values between zero and 1) of the same dimension as the polarization attributes, that will be
            plotted on the alpha channel.
        seismograms : :obj:`obspy.core.Stream`, default = None
            Manually provide seismograms to be plotted in the first panel. By default, the input data is plotted.
        """
        assert self.elli is not None, 'No polarization attributes for Love waves have been computed so far!'
        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
        if seismograms is None:
            self._plot_seismograms(ax[0])
        else:
            self._plot_seismograms(ax[0], seismograms=seismograms)

        filter_mask = np.ones_like(self.dop)
        if alpha is not None:
            filter_mask *= alpha
        ax[1].plot(self.t_windows, filter_mask * self.elli, 'k.')
        ax[1].set_ylabel('Ellipticity')
        ax[1].set_title('Ellipticity')
        ax[2].plot(self.t_windows, filter_mask * self.inc1, 'k.', label='Major')
        ax[2].plot(self.t_windows, filter_mask * self.inc2, 'r.', label='Minor')
        ax[2].set_title('Inclination of major and minor semi-axis')
        ax[2].legend()
        ax[2].set_ylabel('Degrees')
        ax[3].plot(self.t_windows, filter_mask * self.azi1, 'k.', label='Major')
        ax[3].plot(self.t_windows, filter_mask * self.azi2, 'r.', label='Minor')
        ax[3].legend()
        ax[3].set_ylabel('Degrees')
        ax[3].set_title('Azimuth of major and minor semi-axis')
        ax[4].plot(self.t_windows, filter_mask * self.dop, 'k.')
        ax[4].set_title('Degree of polarization')
        ax[4].set_ylabel('DOP')
        if self.timeaxis == 'utc':
            ax[1].xaxis_date()
            ax[2].xaxis_date()
            ax[3].xaxis_date()
            ax[4].xaxis_date()
            ax[4].set_xlabel('Time (UTC)')
        else:
            ax[4].set_xlabel('Time (s)')
        if show:
            plt.show()

    def _plot_seismograms(self, ax: plt.Axes, seismograms: Stream = None):
        if seismograms is None:
            if self.timeaxis == 'utc':
                time = self.N.times(type='matplotlib')
            else:
                time = self.N.times()
            ax.plot(time, self.N.data, 'k:', label='N')
            ax.plot(time, self.E.data, 'k--', label='E')
            ax.plot(time, self.Z.data, 'k', label='Z')
            if self.timeaxis == 'utc':
                ax.xaxis_date()
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            if self.timeaxis == 'utc':
                time = seismograms[0].times(type='matplotlib')
            else:
                time = seismograms[0].times()
            ax.plot(time, seismograms[0].data, 'k:', label='N')
            ax.plot(time, seismograms[1].data, 'k--', label='E')
            ax.plot(time, seismograms[2].data, 'k', label='Z')
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
