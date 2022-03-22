r"""
3-C Polarization analysis and filtering in the time domain
==========================================================
In this tutorial, you will learn how to train a machine learning model that enables the
efficient classification of wave types using six-component polarization analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
from obspy.core import Trace, Stream
from scipy.signal import hilbert, convolve

from twistpy.convenience import ricker
from twistpy.polarization import TimeDomainAnalysis3C, PolarizationModel

rng = np.random.default_rng(1)

# sphinx_gallery_thumbnail_number = -1

########################################################################################################################
# We start by generating a very simple synthetic data set for illustration purposes. The data will contain isolated wave
# arrivals for a P-wave, an SV-wave, an SH wave, a Love-wave, and a Rayleigh wave. To generate the data, we make use
# of the pure state polarization models that are implemented in the class 'twistpy.polarization.PolarizationModel'.
# Note that this example is purely meant to illustrate the usage of TwistPy's wave type classification tools, by using
# the same model to generate the synthetics as we use for training and classification, we are practically committing an
# inverse crime, so that the classification will always yield perfect results.

# Generate an empty time series for each wave
N = 1000  # Number of samples in the time series
signal1 = np.zeros((N, 6))  # Each signal has six components
signal2 = np.zeros((N, 6))
signal3 = np.zeros((N, 6))
signal4 = np.zeros((N, 6))
signal5 = np.zeros((N, 6))

dt = 1. / 1000.  # sampling interval
t = np.arange(0, signal1.shape[0]) * dt  # time axis
wavelet, t_wav, wcenter = ricker(t, 20.0)  # generate a Ricker wavelet with 30 Hz center frequency
wavelet = wavelet[wcenter - int(len(t) / 2): wcenter + int(len(t) / 2)]
wavelet_hilb = np.imag(hilbert(wavelet))  # Here we make use of the Hilbert transform to generate a Ricker wavelet
# with a 90 degree phase shift. This is to account for the fact that, for Rayleigh waves, the horizontal components are
# phase-shifted by 90 degrees with respect to the other components.


########################################################################################################################
# We now generate the relative amplitudes with which the waves are recorded on the six-component seismometer. All waves
# will arrive with a propagation azimuth of 30 degrees, the body waves will have an inclination angle of 20 degrees. The
# local P-wave and S-wave velocities at the recording station are assumed to be 1000 m/s and 400 m/s, respectively. Both
# the Love and Rayleigh wave velocities are assumed to be 300 m/s, and the Rayleigh wave ellipticity angle is set to be
# -45 degrees.

wave1 = PolarizationModel(wave_type='P', theta=20., phi=210., vr=300., vp=1000., vs=400.,
                          xi=45., free_surface=False)  # Generate a P-wave polarization model for
# a P-wave recorded at the free surface with an inclination of 20 degrees, an azimuth of 30 degrees. The local P- and
# S-wave velocities are 1000 m/s and 400 m/s
wave2 = PolarizationModel(wave_type='SV', theta=20., phi=30., vp=1000.,
                          vs=400., free_surface=False)  # Generate an SV-wave polarization model
wave3 = PolarizationModel(wave_type='SH', theta=20., phi=30., vs=400.,
                          vl=400., free_surface=False)  # Generate an SH-wave polarization model
wave4 = PolarizationModel(wave_type='L', phi=350., vl=300.)  # Generate a Love-wave polarization model
wave5 = PolarizationModel(wave_type='R', phi=30., vr=300.,
                          xi=-45.)  # Generate a Rayleigh-wave polarization model with a
# Rayleigh wave ellipticity angle of -45 degrees.


########################################################################################################################
# Now we populate our signal with the computed amplitudes by setting a spike with the respective amplitude onto the
# different locations of the time axis. Then we convolve the data with the Ricker wavelet to generate our synthetic
# test seismograms.

signal1[100, :] = wave1.polarization.real.T
signal2[300, :] = wave2.polarization.real.T
signal3[500, :] = wave3.polarization.real.T
signal4[700, :] = wave4.polarization.real.T
signal5[900, 2:] = np.real(wave5.polarization[2:].T)
signal5[900, 0:2] = np.imag(wave5.polarization[0:2].T)

for j in range(0, signal1.shape[1]):
    signal1[:, j] = convolve(signal1[:, j], wavelet, mode='same')
    signal2[:, j] = convolve(signal2[:, j], wavelet, mode='same')
    signal3[:, j] = convolve(signal3[:, j], wavelet, mode='same')
    signal4[:, j] = convolve(signal4[:, j], wavelet, mode='same')
    if j == 0 or j == 1:  # Special case for horizontal translational components of the Rayleigh wave
        signal5[:, j] = convolve(signal5[:, j], wavelet_hilb, mode='same')
    else:
        signal5[:, j] = convolve(signal5[:, j], wavelet, mode='same')

signal = signal1 + signal2 + signal3 + signal4 + signal5  # sum all signals together

signal = signal[:, 0:3]

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(t, signal[:, 0], 'k:', label='N')
plt.plot(t, signal[:, 1], 'k--', label='E')
plt.plot(t, signal[:, 2], 'k', label='Z')
plt.text((100 - 120) * dt, 0.7, 'P-wave')
plt.text((300 - 120) * dt, 0.7, 'SV-wave')
plt.text((500 - 120) * dt, 0.7, 'SH-wave')
plt.text((700 - 120) * dt, 0.7, 'Love-wave')
plt.text((900 - 120) * dt, 0.7, 'Rayleigh-wave')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Time (s)')

########################################################################################################################
# To make the synthetics accessible to TwistPy, we convert them to an Obspy Stream object.

data = Stream()
for n in range(signal.shape[1]):
    trace = Trace(data=signal[:, n], header={"delta": t[1] - t[0], "npts": int(signal.shape[0]), "starttime": 0.})
    data += trace

########################################################################################################################
# Now that we have trained the model, we can set up our analysis. We will perform 6C polarization analysis in the time
# domain and use a sliding time window that is 0.05 s long (50 samples) with an overlap between subsequent windows of
# 50%.

window = {'window_length_seconds': 20. * dt, 'overlap': 0.5}
analysis = TimeDomainAnalysis3C(N=data[0], E=data[1], Z=data[2], window=window, timeaxis='rel')

########################################################################################################################
# To classify the waves, we simply do (yielding a classification of the first eigenvector of the covariance matrix):

analysis.polarization_analysis()
t_windows = analysis.t_windows  # Positions of the sliding time windows where the classification was performed
analysis.plot_polarization_analysis()

from twistpy.polarization import TimeFrequencyAnalysis3C

data.resample(120)
window = {'number_of_periods': 0.01, 'frequency_extent': 2.}
analysis_tf = TimeFrequencyAnalysis3C(N=data[0], E=data[1], Z=data[2], window=window, timeaxis='rel')
analysis_tf.polarization_analysis()
analysis_tf.plot_polarization_analysis(major_semi_axis=True, clip=0.05)
