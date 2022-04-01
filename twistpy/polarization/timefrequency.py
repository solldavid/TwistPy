from __future__ import (absolute_import, division, print_function, unicode_literals)

import pickle
from builtins import *
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from obspy import Trace, Stream
from obspy.core.utcdatetime import UTCDateTime
from scipy.ndimage import uniform_filter1d

from twistpy.polarization.machinelearning import SupportVectorMachine
from twistpy.utils import stransform, istransform


class TimeFrequencyAnalysis6C:
    """Time-frequency domain six-component polarization analysis.

    Single-station six degree-of-freedom polarization analysis on time-frequency decomposed signals. The time-frequency
    representation is obtained via the S-transform (:func:`twistpy.utils.s_transform`) [1], [2].

    .. hint::
        [1] Sollberger et al. (2018). *6-C polarization analysis using point measurements of translational and
        rotational ground-motion: theory and applications*, Geophysical Journal International, 213(1),
        https://doi.org/10.1093/gji/ggx542

        [2] Sollberger et al. (2020). *Seismological processing of six degree-of-freedom ground-motion data*, Sensors,
        20(23), https://doi.org/10.3390/s20236904

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
        2-D time-frequency window within which the covariance matrices are averaged:

        |  window = {'number_of_periods': :obj:`float`, 'frequency_extent': :obj:`float`}
        |  The extent of the window in time is frequency-dependent and stretches over 'number_of_periods' periods, where
            the period is defined as 1/f. In frequency, the window stretches over the specified 'frequency_extent' in
            Hz.
    dsfact : :obj:`int`, default=1
        Down-sampling factor of the time-axis prior to the computation of polarization attributes.
    dsfacf : :obj:`int`, default=1
        Down-sampling factor of the frequency-axis prior to the computation of polarization attributes.

        .. warning::
                Downsampling of the frequency axis (dsfacf > 1) prevents the accurate computation of the inverse
                S-transform! Down-sampling should be avoided for filtering applications.

    frange : :obj:`tuple`, (f_min, f_max)
        Limit the analysis to the specified frequency band in Hz
    trange : :obj:`tuple`, (t_min, t_max)
        Limit the analysis to the specified time window. t_min and t_max should either be specified as a tuple of two
        :obj:`~obspy.core.utcdatetime.UTCDateTime` objects (if timeaxis='utc'), or as a tuple of :obj:`float` (if
        timeaxis='rel').
    k : :obj:`float`, default=1.
        k-value used in the S-transform, corresponding to a scaling factor that controls the number of oscillations in
        the Gaussian window. When k increases, the frequency resolution increases, with a corresponding loss of time
        resolution.

        .. seealso:: :func:`twistpy.utils.s_transform`

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
        Dictionary containing the labels of classified wave types at each position of the analysis window in time and
        frequency. The dictionary has six entries corresponding to classifications for each eigenvector.

        |  classification = {'0': list_with_classifications_of_first_eigenvector, '1':
            list_with_classification_of_second_eigenvector, ... , '5': list_with_classification_of_last_eigenvector}
    t_pol : :obj:`list` of :obj:`~obspy.core.utcdatetime.UTCDateTime`
        Time axis of the computed polarization attributes.
    f_pol : :obj:`~numpy.ndarray` of :obj:`float`
        Frequency axis of the computed polarization attributes.
    C : :obj:`~numpy.ndarray` of :obj:`~numpy.complex128`
        Complex covariance matrices at each window position
    time : :obj:`list` of :obj:`~obspy.core.utcdatetime.UTCDateTime`
        Time axis of the input traces
    delta : :obj:`float`
        Sampling interval of the input data in seconds

    """

    def __init__(self, traN: Trace, traE: Trace, traZ: Trace, rotN: Trace,
                 rotE: Trace, rotZ: Trace, window: dict, dsfacf: int = 1, dsfact: int = 1,
                 frange: Tuple[float, float] = None, trange: Tuple[UTCDateTime, UTCDateTime] = None, k: float = 1,
                 scaling_velocity: float = 1., free_surface: bool = True, verbose: bool = True,
                 timeaxis: 'str' = 'utc') -> None:

        self.traN, self.traE, self.traZ, self.rotN, self.rotE, self.rotZ = traN, traE, traZ, rotN, rotE, rotZ

        # Assert that input traces are ObsPy Trace objects
        assert isinstance(self.traN, Trace) and isinstance(self.traE, Trace) and isinstance(self.traZ, Trace) \
               and isinstance(self.rotN, Trace) and isinstance(self.rotE, Trace) and isinstance(self.rotZ,
                                                                                                Trace), "Input data " \
                                                                                                        "must be " \
                                                                                                        "objects of " \
                                                                                                        "class " \
                                                                                                        "obspy.core" \
                                                                                                        ".Trace()! "

        # Assert that all traces have the same number of samples
        assert self.traN.stats.npts == self.traE.stats.npts and self.traN.stats.npts == self.traZ.stats.npts \
               and self.traN.stats.npts == self.rotN.stats.npts and self.traN.stats.npts == self.rotE.stats.npts \
               and self.traN.stats.npts == self.rotZ.stats.npts, "All six traces must have the same number of samples!"

        self.window = window
        self.dsfact = dsfact
        self.dsfacf = dsfacf
        self.k = k
        self.timeaxis = timeaxis
        if self.timeaxis == 'utc':
            self.time = self.traN.times(type="matplotlib")
        else:
            self.time = self.traN.times()
        self.scaling_velocity = scaling_velocity
        self.free_surface = free_surface
        self.verbose = verbose
        self.delta = self.traN.stats.delta
        self.classification: Dict[str, List[str]] = {'0': None, '1': None, '2': None, '3': None, '4': None, '5': None}
        self.frange = frange
        self.trange = trange

        N = np.max(traN.data.shape)
        N_half = int(np.floor(N / 2))
        self.f_pol = (np.concatenate([np.arange(N_half + 1)]) / N)[::dsfacf] / self.delta

        # Compute extent of window in no. of samples in the time and frequency direction
        df = self.f_pol[1] - self.f_pol[0]
        periods = 1. / self.f_pol[1:]  # Exclude DC
        periods = np.append(float(self.time[-1] - self.time[0]), periods)
        window_t_samples = np.array(self.window["number_of_periods"] * periods / self.delta, dtype=int)
        window_t_samples[window_t_samples > len(self.time)] = len(self.time)
        window_t_samples[window_t_samples == 0] = 1
        window_f_samples = np.max([1, int(self.window["frequency_extent"] / df)])

        # Compute the S-transform of the input signal
        u: np.ndarray = np.moveaxis(np.array([stransform(traN.data, dsfacf=dsfacf, k=k)[0],
                                              stransform(traE.data, dsfacf=dsfacf, k=k)[0],
                                              stransform(traZ.data, dsfacf=dsfacf, k=k)[0],
                                              stransform(rotN.data, dsfacf=dsfacf, k=k)[0],
                                              stransform(rotE.data, dsfacf=dsfacf, k=k)[0],
                                              stransform(rotZ.data, dsfacf=dsfacf, k=k)[0]]), 0, 2)

        self.signal_amplitudes_st = np.sqrt(np.abs(u[:, :, 0]) ** 2 + np.abs(u[:, :, 1]) ** 2 + np.abs(u[:, :, 2]) ** 2
                                            + np.abs(u[:, :, 3]) ** 2 + np.abs(u[:, :, 4]) ** 2 + np.abs(
            u[:, :, 5]) ** 2)
        if self.trange is not None:
            indx_t = [(np.abs(self.time.astype('float') - float(self.trange[0]))).argmin(),
                      (np.abs(self.time.astype('float') - float(self.trange[1]))).argmin()]
        else:
            indx_t = [0, u.shape[1]]

        if self.frange is not None:
            indx_f = [(np.abs(self.f_pol - self.frange[0])).argmin(), (np.abs(self.f_pol - self.frange[1])).argmin()]
        else:
            indx_f = [0, u.shape[0]]

        self.t_pol = self.time[indx_t[0]:indx_t[1]]
        self.t_pol = self.t_pol[::dsfact]

        u = u[indx_f[0]:indx_f[1], indx_t[0]:indx_t[1], :]
        self.signal_amplitudes_st = self.signal_amplitudes_st[indx_f[0]:indx_f[1], indx_t[0]:indx_t[1]:dsfact]

        # Compute covariance matrices
        if self.verbose:
            print('Computing covariance matrices...')
        C: np.ndarray = np.einsum('...i,...j->...ij', np.conj(u), u).astype('complex')
        for j in range(C.shape[2]):
            for k in range(C.shape[3]):
                C[..., j, k] = uniform_filter1d(C[..., j, k], size=window_f_samples)
                for i in range(C.shape[0]):
                    C[i, :, j, k] = uniform_filter1d(C[i, :, j, k], size=window_t_samples[i])
        self.C = np.reshape(C[:, indx_t[0]:indx_t[1]:dsfact, :, :],
                            (len(self.t_pol) * len(self.f_pol), 6, 6))  # flatten the
        # time and frequency dimension
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
        self.classification[str(eigenvector_to_classify)] = np.reshape(wave_types, (len(self.f_pol), len(self.t_pol)))
        if self.verbose:
            print('Wave types have been classified!')

    def plot_classification(self, ax: Axes, classified_eigenvector: int = 0, clip: float = 0.05) -> None:
        """Plot wave type classification labels as a function of time and frequency.

        Parameters
        ----------
        ax : :obj:`~matplotlib.axes.Axes`
            Matplotlib Axes instance where the classification should be plotted
        classified_eigenvector : :obj:`int`, default = 0
            Specify for which eigenvector the classification is plotted
        clip : :obj:`float`, default=0.05 (between 0 and 1)
            Results are only plotted at time-frequency points where the signal amplitudes exceed a value of
            amplitudes * maximum amplitude in the signal (given by the l2-norm of all three components).
        """
        if self.classification[str(classified_eigenvector)] is not None:
            cmap = ListedColormap(['blue', 'red', 'green', 'yellow', 'white'])
            map = ScalarMappable(Normalize(vmin=0, vmax=4), cmap=cmap)
            d = {'P': 0, 'SV': 1, 'SH': 2, 'L': 2, 'R': 3, 'Noise': 4}
            classification = (self.classification[str(classified_eigenvector)]).flatten()
            # Convert classification label from string to numbers for plotting
            classification = np.array([d[wave_type] for wave_type in classification])
            classification = np.reshape(classification, self.classification[str(classified_eigenvector)].shape)
            classification[self.signal_amplitudes_st < clip * self.signal_amplitudes_st.max().max()] = 4
            ax.imshow(classification, origin='lower',
                      extent=[float(self.t_pol[0]), float(self.t_pol[-1]), self.f_pol[0], self.f_pol[-1]],
                      cmap=cmap, aspect='auto')
            cbar = plt.colorbar(map, ax=ax, extend='max')
            cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
            cbar.set_ticklabels(['P', 'SV', 'SH', 'R', 'Noise'])
            cbar.set_label(f"Wave type")
        else:
            raise Exception(f"No wave types have been classified for this eigenvector yet (eigenvector: "
                            f"{classified_eigenvector})!")

    def save(self, name: str) -> None:
        """ Save the current TimeFrequencyAnalaysis6C object to a file on the disk in the current working directory.

         Include absolute path if you want to save in a different directory.

        name : :obj:`str`
            File name
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string!")

        fid = open(name, 'wb')
        pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL)
        fid.close()


class TimeFrequencyAnalysis3C:
    """Time-frequency domain three-component polarization analysis.

    Single-station three-component polarization analysis on time-frequency decomposed signals. The time-frequency
    representation is obtained via the S-transform (:func:`twistpy.utils.s_transform`).

    Parameters
    ----------
    N : :obj:`~obspy.core.trace.Trace`
     North component seismogram
    E : :obj:`~obspy.core.trace.Trace`
     East component seismogram
    Z : :obj:`~obspy.core.trace.Trace`
        Vertical component seismogram
    window : :obj:`dict`
        2-D time-frequency window within which the covariance matrices are averaged:

        |  window = {'number_of_periods': :obj:`float`, 'frequency_extent': :obj:`float`}
        |  The extent of the window in time is frequency-dependent and stretches over 'number_of_periods' periods, where
            the period is defined as 1/f. In frequency, the window stretches over the specified 'frequency_extent' in
            Hz.
    dsfact : :obj:`int`, default=1
        Down-sampling factor of the time-axis prior to the computation of polarization attributes.
    dsfacf : :obj:`int`, default=1
        Down-sampling factor of the frequency-axis prior to the computation of polarization attributes.

        .. warning::
                Downsampling of the frequency axis (dsfacf > 1) prevents the accurate computation of the inverse
                S-transform! Down-sampling should be avoided for filtering applications.

    frange : :obj:`tuple`, (f_min, f_max)
        Limit the analysis to the specified frequency band in Hz
    trange : :obj:`tuple`, (t_min, t_max)
        Limit the analysis to the specified time window. t_min and t_max should either be specified as a tuple of two
        :obj:`~obspy.core.utcdatetime.UTCDateTime` objects (if timeaxis='utc'), or as a tuple of :obj:`float` (if
        timeaxis='rel').
    k : :obj:`float`, default=1.
        k-value used in the S-transform, corresponding to a scaling factor that controls the number of oscillations in
        the Gaussian window. When k increases, the frequency resolution increases, with a corresponding loss of time
        resolution.

        .. seealso:: :func:`twistpy.utils.s_transform`

    verbose : :obj:`bool`, default=True
        Run in verbose mode.
    timeaxis : :obj:'str', default='utc'
        Specify whether the time axis of plots is shown in UTC (timeaxis='utc') or in seconds relative to the first
        sample (timeaxis='rel').

    Attributes
    ----------
    t_pol : :obj:`list` of :obj:`~obspy.core.utcdatetime.UTCDateTime`
        Time axis of the computed polarization attributes.
    f_pol : :obj:`~numpy.ndarray` of :obj:`float`
        Frequency axis of the computed polarization attributes.
    C : :obj:`~numpy.ndarray` of :obj:`~numpy.complex128`
        Complex covariance matrices at each window position
    time : :obj:`list` of :obj:`~obspy.core.utcdatetime.UTCDateTime`
        Time axis of the input traces
    delta : :obj:`float`
        Sampling interval of the input data in seconds
    dop : 2D :obj:`numpy.ndarray` of :obj:`float`
        Degree of polarization estimated at each window position in time and frequency.

        .. hint::
                The definition of the degree of polarization follows the one from Samson & Olson (1980): *Some comments
                on the descriptions of the polarization states of waves*, Geophysical Journal of the Royal Astronomical
                Society, https://doi.org/10.1111/j.1365-246X.1980.tb04308.x. It is defined as:

                .. math::
                    P^2=\sum_{j,k=1}^{n}(\lambda_j-\lambda_k)^2/[2(n-1)(\sum_{j=1}^{n}(\lambda_j)^2)]

                where :math:`P^2` is the degree of polarization and :math:`\lambda_j` are the eigenvalues of the
                covariance matrix.
    elli : 2D :obj:`numpy.ndarray` of :obj:`float`
        Ellipticity estimated at each window position
    inc1 : 2D :obj:`numpy.ndarray` of :obj:`float`
        Inclination of the major semi-axis of the polarization ellipse estimated at each window position
    inc2 : 2D :obj:`numpy.ndarray` of :obj:`float`
        Inclination of the minor semi-axis of the polarization ellipse estimated at each window position
    azi1 : 2D :obj:`numpy.ndarray` of :obj:`float`
        Azimuth of the major semi-axis of the polarization ellipse estimated at each window position
    azi2 : 2D :obj:`numpy.ndarray` of :obj:`float`
        Azimuth of the minor semi-axis of the polarization ellipse estimated at each window position

    """

    def __init__(self, N: Trace, E: Trace, Z: Trace, window: dict, dsfacf: int = 1, dsfact: int = 1,
                 frange: Tuple[float, float] = None, trange: Tuple[UTCDateTime, UTCDateTime] = None, k: float = 1,
                 verbose: bool = True, timeaxis: 'str' = 'utc') -> None:

        self.dop = None
        self.elli = None
        self.inc1 = None
        self.inc2 = None
        self.azi1 = None
        self.azi2 = None
        self.N, self.E, self.Z = N, E, Z

        # Assert that input traces are ObsPy Trace objects
        assert isinstance(self.N, Trace) and isinstance(self.E, Trace) and isinstance(self.Z, Trace), "Input data " \
                                                                                                      "must be " \
                                                                                                      "objects of " \
                                                                                                      "class " \
                                                                                                      "obspy.core" \
                                                                                                      ".Trace()! "

        # Assert that all traces have the same number of samples
        assert self.N.stats.npts == self.E.stats.npts and self.N.stats.npts == self.Z.stats.npts, \
            "All six traces must have the same number of samples!"

        self.window = window
        self.dsfact = dsfact
        self.dsfacf = dsfacf
        self.k = k
        self.timeaxis = timeaxis
        if self.timeaxis == 'utc':
            self.time = self.N.times(type="matplotlib")
        else:
            self.time = self.N.times()
        self.verbose = verbose
        self.delta = self.N.stats.delta
        self.frange = frange
        self.trange = trange

        N_full = np.max(N.data.shape)
        N_half = int(np.floor(N_full / 2))
        self.f_pol = (np.concatenate([np.arange(N_half + 1)]) / N_full)[::dsfacf] / self.delta

        # Compute extent of window in no. of samples in the time and frequency direction
        df = self.f_pol[1] - self.f_pol[0]
        periods = 1. / self.f_pol[1:]  # Exclude DC
        periods = np.append(float(self.time[-1] - self.time[0]), periods)
        window_t_samples = np.array(self.window["number_of_periods"] * periods / self.delta, dtype=int)
        window_t_samples[window_t_samples == 0] = 1
        window_t_samples[window_t_samples > len(self.time)] = len(self.time)
        window_f_samples = np.max([1, int(self.window["frequency_extent"] / df)])

        self.window_t_samples = window_t_samples
        self.window_f_samples = window_f_samples

        # Compute the S-transform of the input signal
        u: np.ndarray = np.moveaxis(np.array([stransform(N.data, dsfacf=dsfacf, k=k)[0],
                                              stransform(E.data, dsfacf=dsfacf, k=k)[0],
                                              stransform(Z.data, dsfacf=dsfacf, k=k)[0]]),
                                    0, 2)

        self.signal_amplitudes_st = np.sqrt(np.abs(u[:, :, 0]) ** 2 + np.abs(u[:, :, 1]) ** 2 + np.abs(u[:, :, 2]) ** 2
                                            )
        if self.trange is not None:
            indx_t = [(np.abs(self.time.astype('float') - float(self.trange[0]))).argmin(),
                      (np.abs(self.time.astype('float') - float(self.trange[1]))).argmin()]
        else:
            indx_t = [0, u.shape[1]]

        if self.frange is not None:
            indx_f = [(np.abs(self.f_pol - self.frange[0])).argmin(), (np.abs(self.f_pol - self.frange[1])).argmin()]
        else:
            indx_f = [0, u.shape[0]]

        self.t_pol = self.time[indx_t[0]:indx_t[1]]
        self.t_pol = self.t_pol[::dsfact]

        u = u[indx_f[0]:indx_f[1], indx_t[0]:indx_t[1], :]
        self.signal_amplitudes_st = self.signal_amplitudes_st[indx_f[0]:indx_f[1], indx_t[0]:indx_t[1]:dsfact]

        # Compute covariance matrices
        if self.verbose:
            print('Computing covariance matrices...')
        C: np.ndarray = np.einsum('...i,...j->...ij', np.conj(u), u).astype('complex')
        for j in range(C.shape[2]):
            for k in range(C.shape[3]):
                C[..., j, k] = uniform_filter1d(C[..., j, k], size=window_f_samples, axis=0)
                for i in range(C.shape[0]):
                    C[i, :, j, k] = uniform_filter1d(C[i, :, j, k], size=window_t_samples[i])
        self.C = np.reshape(C[:, indx_t[0]:indx_t[1]:dsfact, :, :],
                            (len(self.t_pol) * len(self.f_pol), 3, 3))  # flatten the
        # time and frequency dimension
        if self.verbose:
            print('Covariance matrices computed!')

    def polarization_analysis(self) -> None:
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
        self.dop = np.reshape(self.dop, (len(self.f_pol), len(self.t_pol)))
        self.azi1 = np.degrees(np.pi / 2 - np.arctan2(eigenvectors[:, 0, -1].real, eigenvectors[:, 1, -1].real))
        self.azi2 = np.degrees(np.pi / 2 - np.arctan2(eigenvectors[:, 0, -1].imag, eigenvectors[:, 1, -1].imag))
        self.azi1[self.azi1 < 0] += 180
        self.azi2[self.azi2 < 0] += 180
        self.azi1[self.azi1 > 180] -= 180
        self.azi2[self.azi2 > 180] -= 180
        self.azi1 = np.reshape(self.azi1, (len(self.f_pol), len(self.t_pol)))
        self.azi2 = np.reshape(self.azi2, (len(self.f_pol), len(self.t_pol)))
        self.inc1 = np.degrees(np.arctan2(np.linalg.norm(eigenvectors[:, 0:2, -1].real, axis=1),
                                          np.abs(eigenvectors[:, 2, -1].real)))
        self.inc2 = np.degrees(np.arctan2(np.linalg.norm(eigenvectors[:, 0:2, -1].imag, axis=1),
                                          np.abs(eigenvectors[:, 2, -1].imag)))
        self.inc1 = np.reshape(self.inc1, (len(self.f_pol), len(self.t_pol)))
        self.inc2 = np.reshape(self.inc2, (len(self.f_pol), len(self.t_pol)))
        self.elli = np.linalg.norm(eigenvectors[:, :, -1].imag, axis=1) / np.linalg.norm(
            eigenvectors[:, :, -1].real,
            axis=1)
        self.elli = np.reshape(self.elli, (len(self.f_pol), len(self.t_pol)))
        if self.verbose:
            print('Polarization attributes have been computed!')

    def filter(self, plot_filtered_attributes: bool = False, **kwargs) -> Stream:
        r"""Filter data based on polarization attributes.

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

        if self.dsfacf != 1 or self.dsfact != 1:
            raise Exception('Polarization filtering is only supported if the polarization attributes have been computed'
                            'at each time-frequency pixel. To do so, recompute the polarization attributes and set '
                            'dsfacf=1 and dsfact=1.')
        params = {}
        for k in kwargs:
            if kwargs[k] is not None:
                params[k] = kwargs[k]
        if not params:
            raise Exception("No values provided")

        if self.elli is None:
            raise Exception('No polarization attributes computed yet!')

        # Compute eigenvectors for projection
        _, eigenvectors = np.linalg.eigh(self.C)

        # Compute S-transform of each component
        Z_stran, f_stran = stransform(self.Z.data, k=self.k)
        N_stran, _ = stransform(self.N.data, k=self.k)
        E_stran, _ = stransform(self.E.data, k=self.k)

        # Initialize filter mask
        mask = np.ones((len(self.f_pol), len(self.t_pol)))

        # Populate filter mask (will be 1 where signal is kept and 0 everywhere else)
        for parameter in params:
            pol_attr = np.real(getattr(self, parameter))
            if parameter == 'dop' or parameter == 'elli' and params[parameter][1] == 1.:
                alpha = pol_attr >= params[parameter][0]
            else:
                alpha = ((pol_attr >= params[parameter][0]) &
                         (pol_attr <= params[parameter][1])).astype('int')
            mask *= alpha

        mask = uniform_filter1d(mask, size=self.window_f_samples, axis=0)
        for i in range(mask.shape[0]):
            mask[i, :] = uniform_filter1d(mask[i, :], size=self.window_t_samples[i])
        indx = mask > 0

        Z_sep = np.zeros_like(Z_stran).astype('complex')
        N_sep = np.zeros_like(N_stran).astype('complex')
        E_sep = np.zeros_like(E_stran).astype('complex')

        data_st = np.array([N_stran[indx].ravel(), E_stran[indx].ravel(), Z_stran[indx].ravel()]).T
        # Project data into coordinate frame spanned by eigenvectors
        data_proj = np.einsum('...i, ...ij -> ...j', data_st, eigenvectors[indx.ravel(), :, :], optimize=True)

        # Only keep data that is aligned with the principal eigenvector
        data_proj[:, 0:2] = 0

        # Back-projection into original coordinate frame
        data_filt = np.einsum('...i, ...ij -> ...j', data_proj, np.transpose(eigenvectors[indx.ravel(), :, :].conj(),
                                                                             axes=(0, 2, 1)), optimize=True)
        Z_sep[indx] = data_filt[:, 2]
        N_sep[indx] = data_filt[:, 0]
        E_sep[indx] = data_filt[:, 1]

        Z_sep = istransform(mask * Z_sep, f=f_stran, k=self.k)
        N_sep = istransform(mask * N_sep, f=f_stran, k=self.k)
        E_sep = istransform(mask * E_sep, f=f_stran, k=self.k)
        #
        data_filtered = Stream(traces=[self.N.copy(), self.E.copy(), self.Z.copy()])
        data_filtered[0].data = N_sep
        data_filtered[1].data = E_sep
        data_filtered[2].data = Z_sep

        if plot_filtered_attributes:
            self.plot(show=False, alpha=mask, seismograms=data_filtered)

        return data_filtered

    def plot(self, clip: float = 0.05, major_semi_axis: bool = True, show: bool = True,
             alpha: np.ndarray = None, seismograms: Stream = None) -> None:
        """Plot polarization analysis.

        Parameters
        ----------
        clip : :obj:`float`, default=0.05 (between 0 and 1)
            Results are only plotted at time-frequency points where the signal amplitudes exceed a value of
            amplitudes * maximum amplitude in the signal (given by the l2-norm of all three components).
        major_semi_axis : obj:`bool`, default=True
            If True, the inclination and azimuth of the major semi-axis is plotted. Otherwise (major_semi_axis=False),
            The inclination and azimuth are plotted for the minor semi-axis.
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
        ax[0].set_ylabel('Amplitude')

        alpha_channel = np.ones(self.dop.shape)
        if alpha is not None:
            alpha_channel *= alpha
        alpha_channel[self.signal_amplitudes_st < clip * self.signal_amplitudes_st.max().max()] = 0
        alpha_channel[alpha_channel > 1] = 1
        alpha_channel[alpha_channel < 0] = 0

        ax[1].imshow(self.elli, origin='lower', aspect='auto', alpha=alpha_channel,
                     extent=[self.t_pol[0], self.t_pol[-1], self.f_pol[0], self.f_pol[-1]], cmap='inferno',
                     vmin=0, vmax=1)
        map_elli = ScalarMappable(Normalize(vmin=0, vmax=1), cmap='inferno')
        cbar_elli = plt.colorbar(map_elli, ax=ax[1], extend='max')
        cbar_elli.set_label(f"Ellipticity")
        ax[1].set_title('Ellipticity')
        ax[1].set_ylabel('Frequency (Hz)')

        if major_semi_axis:
            ax[2].imshow(self.inc1, origin='lower', aspect='auto', alpha=alpha_channel,
                         extent=[self.t_pol[0], self.t_pol[-1], self.f_pol[0], self.f_pol[-1]], cmap='inferno',
                         vmin=0, vmax=90)
            map_inc1 = ScalarMappable(Normalize(vmin=0, vmax=90), cmap='inferno')
            cbar_inc1 = plt.colorbar(map_inc1, ax=ax[2], extend='max')
            cbar_inc1.set_label(f"Inclination (degrees)")
            ax[2].set_title('Inclination of major semi-axis')
        else:
            ax[2].imshow(self.inc2, origin='lower', aspect='auto', alpha=alpha_channel,
                         extent=[self.t_pol[0], self.t_pol[-1], self.f_pol[0], self.f_pol[-1]], cmap='inferno',
                         vmin=0, vmax=90)
            map_inc2 = ScalarMappable(Normalize(vmin=0, vmax=90), cmap='inferno')
            cbar_inc2 = plt.colorbar(map_inc2, ax=ax[2], extend='max')
            cbar_inc2.set_label(f"Inclination (degrees)")
            ax[2].set_title('Inclination of minor semi-axis')
        ax[2].set_ylabel('Frequency (Hz)')

        if major_semi_axis:
            ax[3].imshow(self.azi1, origin='lower', aspect='auto', alpha=alpha_channel,
                         extent=[self.t_pol[0], self.t_pol[-1], self.f_pol[0], self.f_pol[-1]], cmap='twilight',
                         vmin=0, vmax=180)
            map_azi1 = ScalarMappable(Normalize(vmin=0, vmax=180), cmap='twilight')
            cbar_azi1 = plt.colorbar(map_azi1, ax=ax[3], extend='max')
            cbar_azi1.set_label(f"Azimuth (degrees)")
            ax[3].set_title('Azimuth of major semi-axis')
        else:
            ax[3].imshow(self.azi2, origin='lower', aspect='auto', alpha=alpha_channel,
                         extent=[self.t_pol[0], self.t_pol[-1], self.f_pol[0], self.f_pol[-1]], cmap='twilight',
                         vmin=0, vmax=180)
            map_azi2 = ScalarMappable(Normalize(vmin=0, vmax=180), cmap='twilight')
            cbar_azi2 = plt.colorbar(map_azi2, ax=ax[3], extend='max')
            cbar_azi2.set_label(f"Azimuth (degrees)")
            ax[3].set_title('Azimuth of minor semi-axis')
        ax[3].set_ylabel('Frequency (Hz)')

        ax[4].imshow(self.dop, origin='lower', aspect='auto', alpha=alpha_channel,
                     extent=[self.t_pol[0], self.t_pol[-1], self.f_pol[0], self.f_pol[-1]], cmap='inferno',
                     vmin=0, vmax=1)
        map_dop = ScalarMappable(Normalize(vmin=0, vmax=1), cmap='inferno')
        cbar_dop = plt.colorbar(map_dop, ax=ax[4], extend='max')
        cbar_dop.set_label(f"DOP")
        ax[4].set_title('Degree of polarization')
        ax[4].set_ylabel('Frequency (Hz)')

        # Ensure that all three subplots have the same width
        plt.style.use("ggplot")
        pos = ax[4].get_position()
        pos0 = ax[0].get_position()
        ax[0].set_position([pos0.x0, pos0.y0, pos.width, pos.height])

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
        """ Save the current TimeFrequencyAnalaysis3C object to a file on the disk in the current working directory.

         Include absolute path if you want to save in a different directory.

        name : :obj:`str`
            File name
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string!")

        fid = open(name, 'wb')
        pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL)
        fid.close()
