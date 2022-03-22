from __future__ import (absolute_import, division, print_function, unicode_literals)

import pickle
from builtins import *
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import NonUniformImage
from matplotlib.patches import Rectangle
from obspy import Trace, Stream

from twistpy.polarization.estimator import EstimatorConfiguration
from twistpy.polarization.machinelearning import SupportVectorMachine
from twistpy.polarization.time import TimeDomainAnalysis6C


class DispersionAnalysis:
    r"""
    Single-station six-component surface wave dispersion estimation for Love and Rayleigh waves.

    Under the hood, the DispersionAnalysis class runs a :obj:`~twistpy.time.TimeDomainAnalysis6C`. The data is
    filtered to different frequency bands of interest, and the wave parameters are estimated at each frequency
    independently.

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
    fmin : :obj:`float`
        Minimum frequency to be considered in the analysis (in Hz)
    fmax : :obj:`float`
        Maximum frequency to be considered in the analysis (in Hz)
    octaves : :obj:`float`
        Width of the frequency band used for bandpass filtering in octaves
    window : :obj:`dict`
        Window parameters defined as:

        |  window = {'number_of_periods': :obj:`float`, 'overlap': :obj:`float`}
        |  Overlap should be on the interval 0 (no overlap between subsequent time windows) and 1
           (complete overlap, window is moved by 1 sample only). The window is frequency dependent and extends over
           number_of_periods times the dominant period at the current frequency band
    scaling_velocity : :obj:`float`, default=1.
        Scaling velocity (in m/s) to ensure numerical stability. The scaling velocity is
        applied to the translational data only (amplitudes are divided by scaling velocity) and ensures
        that both translation and rotation amplitudes are on the same order of magnitude and
        dimensionless. Ideally, v_scal is close to the S-Wave velocity at the receiver. After applying
        the scaling velocity, translation and rotation signals amplitudes should be similar.    svm
    verbose : :obj:`bool`, default=True
        Run in verbose mode.

    Attributes
    ----------
    parameters : :obj:`list` of :obj:`dict`
        List containing the estimated surface wave parameters at each frequency.
        The parameters for each frequency are saved in a dictionary with the following keys:

        |  'xi': Rayleigh wave ellipticity angle (degrees)

        |  'phi_r': Rayleigh wave back-azimuth (degrees)

        |  'c_r': Rayleigh wave phase-velocity (m/s)

        |  'phi_l': Love wave back azimuth (degrees)

        |  'c_l': Love wave phase velocity (m/s)

        |  'wave_types': Wave type classification labels

        |  'dop': Degree of polarization
    f : :obj:`numpy.ndarray` of :obj:`float`
        Frequency vector for parameters.
    """

    def __init__(self, traN: Trace = None, traE: Trace = None, traZ: Trace = None, rotN: Trace = None,
                 rotE: Trace = None, rotZ: Trace = None, fmin: float = 1.0, fmax: float = 20.0, octaves: float = 0.5,
                 window: dict = {'number_of_periods': 1, 'overlap': 0.}, scaling_velocity: float = 1.,
                 svm: SupportVectorMachine = None, verbose: bool = True):
        f_lower = []
        f_higher = []
        fcenter = fmax
        f_center_tot = []
        data = Stream()
        data += traN
        data += traE
        data += traZ
        data += rotN
        data += rotE
        data += rotZ

        # Compute width of bandpass filters (width in octaves)
        while fcenter > fmin:
            f_l = fcenter / 2 ** (octaves / 2)
            f_h = fcenter * (2 ** (octaves / 2))
            f_lower.append(f_l)
            f_higher.append(f_h)
            f_center_tot.append(fcenter)
            fcenter = fcenter / (2 ** octaves)

        self.traN, self.traE, self.traZ, self.rotN, self.rotE, self.rotZ = traN, traE, traZ, rotN, rotE, rotZ
        self.f = f_center_tot
        self.f_lower = f_lower
        self.f_higher = f_higher
        est = EstimatorConfiguration(wave_types=['L', 'R'], method='ML', scaling_velocity=scaling_velocity,
                                     use_ml_classification=True, svm=svm)
        self.parameters = []

        for i in range(len(f_lower)):
            if verbose:
                print(f"Estimating surface wave parameters at frequency f = {f_center_tot[i]: 0.2f} Hz. "
                      f"Frequency step {i + 1} out of {len(f_lower)}.")
            window_f = {'window_length_seconds': 1 / f_center_tot[i] * window['number_of_periods'],
                        'overlap': window['overlap']}
            data_f = data.copy()
            data_f.filter('lowpass', freq=f_higher[i], zerophase=True)
            data_f.filter('highpass', freq=f_lower[i], zerophase=True)
            analysis = TimeDomainAnalysis6C(traN=data_f[0], traE=data_f[1], traZ=data_f[2], rotN=data_f[3],
                                            rotE=data_f[4], rotZ=data_f[5], window=window_f,
                                            scaling_velocity=scaling_velocity, verbose=False)
            analysis.polarization_analysis(estimator_configuration=est)
            parameters_f = {'xi': analysis.xi, 'phi_r': analysis.phi_r, 'phi_l': analysis.phi_l, 'c_l': analysis.c_l,
                            'c_r': analysis.c_r, 'wave_types': analysis.classification['0'], 'dop': analysis.dop}
            self.parameters.append(parameters_f)

    def plot_dispersion_curves(self, nbins: int = 100, velocity_range: Tuple[float, float] = (0, 2500),
                               quantiles: Tuple[float, float] = (0.2, 0.8), dop_min: float = 0.3):

        counts_r = np.zeros((nbins, len(self.f)))
        counts_l = np.zeros((nbins, len(self.f)))
        counts_dop_r = np.zeros((nbins, len(self.f)))
        counts_dop_l = np.zeros((nbins, len(self.f)))
        counts_elli = np.zeros((nbins, len(self.f)))
        median_cr = np.zeros(len(self.f), )
        median_cl = np.zeros(len(self.f), )
        median_elli = np.zeros(len(self.f), )
        cr_ndec = np.zeros(len(self.f), )
        cl_ndec = np.zeros(len(self.f), )

        for i, f in enumerate(self.f):
            dop_f = self.parameters[i]['dop']

            dop_r = dop_f
            dop_l = dop_f

            c_r_full = self.parameters[i]['c_r']
            dop_r = dop_r[dop_f > dop_min]
            c_r_full = c_r_full[dop_f > dop_min]
            dop_r = dop_r[~np.isnan(c_r_full)]
            c_r_full = c_r_full[~np.isnan(c_r_full)]
            dop_r = dop_r[c_r_full < np.quantile(c_r_full, quantiles[1])]
            c_r = c_r_full[c_r_full < np.quantile(c_r_full, quantiles[1])]
            dop_r = dop_r[c_r > np.quantile(c_r_full, quantiles[0])]
            c_r = c_r[c_r > np.quantile(c_r_full, quantiles[0])]
            dop_r = dop_r[c_r < velocity_range[1]]
            c_r = c_r[c_r < velocity_range[1]]
            dop_r = dop_r[c_r > velocity_range[0]]
            c_r = c_r[c_r > velocity_range[0]]
            cr_ndec[i] = len(c_r)
            counts, values_r = np.histogram(c_r, bins=nbins, range=[velocity_range[0], velocity_range[1]], density=True)
            counts_r[:, i] = counts.T
            median_cr[i] = np.median(c_r)

            counts, _ = np.histogram(dop_r, bins=nbins, range=[0, 1], density=True)
            counts_dop_r[:, i] = counts

            elli_full = self.parameters[i]['xi']
            elli_full = elli_full[dop_f > dop_min]
            elli_full = elli_full[~np.isnan(elli_full)]
            elli = elli_full[elli_full < np.quantile(elli_full, quantiles[1])]
            elli = elli[elli > np.quantile(elli_full, quantiles[0])]
            counts, values_elli = np.histogram(elli, bins=nbins, range=[-90, 90], density=True)
            counts_elli[:, i] = counts.T
            median_elli[i] = np.median(elli)

            c_l_full = self.parameters[i]['c_l']
            dop_l = dop_l[dop_f > dop_min]
            c_l_full = c_l_full[dop_f > dop_min]
            dop_l = dop_l[~np.isnan(c_l_full)]
            c_l_full = c_l_full[~np.isnan(c_l_full)]
            dop_l = dop_l[c_l_full < np.quantile(c_l_full, quantiles[1])]
            c_l = c_l_full[c_l_full < np.quantile(c_l_full, quantiles[1])]
            dop_l = dop_l[c_l > np.quantile(c_l_full, quantiles[0])]
            c_l = c_l[c_l > np.quantile(c_l_full, quantiles[0])]
            dop_l = dop_l[c_l < velocity_range[1]]
            c_l = c_l[c_l < velocity_range[1]]
            dop_l = dop_l[c_l > velocity_range[0]]
            c_l = c_l[c_l > velocity_range[0]]
            cl_ndec[i] = len(c_l)
            counts, values_l = np.histogram(c_l, bins=nbins, range=[velocity_range[0], velocity_range[1]], density=True)
            counts_l[:, i] = counts.T
            median_cl[i] = np.median(c_l)

            counts, _ = np.histogram(dop_l, bins=nbins, range=[0, 1], density=True)
            counts_dop_l[:, i] = counts

        fig1, ((ax2, ax1, ax5), (ax4, ax3, ax6), (ax7, ax8, ax9)) = \
            plt.subplots(3, 3, gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=(12, 10))

        im = NonUniformImage(ax1, interpolation='nearest', cmap='magma',
                             extent=[self.f[-1], self.f[0], velocity_range[0], velocity_range[1]])
        im.set_data(np.flip(self.f), np.linspace(velocity_range[0], velocity_range[1], nbins), np.flip(counts_r,
                                                                                                       axis=1))
        im.set_clim([0.005 * np.max(np.max(counts_r)), 0.8 * np.max(np.max(counts_r))])
        ax1.images.append(im)
        ax1.set_ylim([velocity_range[0], velocity_range[1]])
        ax1.set_ylabel(f"Phase velocity (ms)")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_title("Rayleigh")

        im = NonUniformImage(ax2, interpolation='nearest', cmap='magma',
                             extent=[self.f[-1], self.f[0], velocity_range[0], velocity_range[1]])
        im.set_data(np.flip(self.f), np.linspace(velocity_range[0], velocity_range[1], nbins),
                    np.flip(counts_l, axis=1))
        im.set_clim([0.005 * np.max(np.max(counts_l)), 0.6 * np.max(np.max(counts_l))])
        ax2.images.append(im)
        ax2.set_ylim([velocity_range[0], velocity_range[1]])
        ax2.set_ylabel(f"Phase velocity (ms)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_title("Love")

        for n in range(len(cl_ndec)):
            rect_l = Rectangle((self.f_lower[n], 0), self.f_higher[n] - self.f_lower[n], cl_ndec[n], edgecolor='k',
                               facecolor='none')
            rect_r = Rectangle((self.f_lower[n], 0), self.f_higher[n] - self.f_lower[n], cr_ndec[n], edgecolor='k',
                               facecolor='none')
            rect_re = Rectangle((self.f_lower[n], 0), self.f_higher[n] - self.f_lower[n], cr_ndec[n], edgecolor='k',
                                facecolor='none')

            ax4.add_patch(rect_l)
            ax3.add_patch(rect_r)
            ax6.add_patch(rect_re)

        im = NonUniformImage(ax5, interpolation='nearest', cmap='magma',
                             extent=[self.f[-1], self.f[0], 0, 1])
        im.set_data(np.flip(self.f), np.linspace(-90, 90, nbins), np.flip(counts_elli, axis=1))
        im.set_clim([0.005 * np.max(np.max(counts_elli)), 0.8 * np.max(np.max(counts_elli))])
        ax5.images.append(im)
        ax5.set_ylim([-90, 90])
        ax5.set_ylabel(f"Ellipticity")
        ax5.set_xlabel("Frequency (Hz)")
        ax5.set_title("Rayleigh")

        ax1.plot(np.flip(self.f), np.flip(median_cr), '-.r')
        ax5.plot(np.flip(self.f), np.flip(median_elli), '-.r')
        ax2.plot(np.flip(self.f), np.flip(median_cl), '-.r')
        ax4.set_xlim(ax2.get_xlim())
        ax3.set_xlim(ax1.get_xlim())
        ax6.set_xlim(ax1.get_xlim())
        ax4.set_ylim([0, 1.2 * np.max([cr_ndec.max(), cl_ndec.max()])])
        ax3.set_ylim([0, 1.2 * np.max([cr_ndec.max(), cl_ndec.max()])])
        ax6.set_ylim([0, 1.2 * np.max([cr_ndec.max(), cl_ndec.max()])])
        ax4.set_ylabel('No. of detections')

        im = NonUniformImage(ax7, interpolation='nearest', cmap='magma',
                             extent=[self.f[-1], self.f[0], 0, 1])
        im.set_data(np.flip(self.f), np.linspace(0, 1, nbins), np.flip(counts_dop_l, axis=1))
        im.set_clim([0.005 * np.max(np.max(counts_dop_l)), 0.8 * np.max(np.max(counts_dop_l))])
        ax7.images.append(im)
        ax7.set_ylim([0, 1])
        ax7.set_ylabel(f"Degree of polarization")
        ax7.set_xlabel("Frequency (Hz)")

        im = NonUniformImage(ax8, interpolation='nearest', cmap='magma',
                             extent=[self.f[-1], self.f[0], 0, 1])
        im.set_data(np.flip(self.f), np.linspace(0, 1, nbins), np.flip(counts_dop_r, axis=1))
        im.set_clim([0.005 * np.max(np.max(counts_dop_r)), 0.8 * np.max(np.max(counts_dop_r))])
        ax8.images.append(im)
        ax8.set_ylim([0, 1])
        ax8.set_xlabel("Frequency (Hz)")

        im = NonUniformImage(ax9, interpolation='nearest', cmap='magma',
                             extent=[self.f[-1], self.f[0], 0, 1])
        im.set_data(np.flip(self.f), np.linspace(0, 1, nbins), np.flip(counts_dop_r, axis=1))
        im.set_clim([0.005 * np.max(np.max(counts_dop_r)), 0.8 * np.max(np.max(counts_dop_r))])
        ax9.images.append(im)
        ax9.set_ylim([0, 1])
        ax7.set_xlim(ax2.get_xlim())
        ax8.set_xlim(ax1.get_xlim())
        ax9.set_xlim(ax1.get_xlim())
        ax9.set_xlabel("Frequency (Hz)")
        plt.show()

    def save(self, name: str) -> None:
        """ Save the current DispersionAnalysis object to a file on the disk in the current working directory.

         Include absolute path if you want to save in a different directory.

        name : :obj:`str`
            File name
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string!")

        fid = open(name, 'wb')
        pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL)
        fid.close()
