r"""
6-C Polarization Analysis: Wave parameter estimation
====================================================
In this tutorial, you will learn how to train a machine learning model that enables the
efficient classification of wave types using six-component polarization analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
from obspy.core import Trace, Stream
from scipy.signal import hilbert, convolve

from twistpy import TimeDomainAnalysis, PolarizationModel
from twistpy.convenience import ricker
from twistpy.machinelearning import SupportVectorMachine

rng = np.random.default_rng(1)

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

wave1 = PolarizationModel(wave_type='R', theta=20., phi=210., vr=300.,
                          xi=45.)  # Generate a P-wave polarization model for
# a P-wave recorded at the free surface with an inclination of 20 degrees, an azimuth of 30 degrees. The local P- and
# S-wave velocities are 1000 m/s and 400 m/s
wave2 = PolarizationModel(wave_type='SV', theta=20., phi=30., vp=1000.,
                          vs=400.)  # Generate an SV-wave polarization model
wave3 = PolarizationModel(wave_type='SH', theta=20., phi=30., vs=400.,
                          vl=400.)  # Generate an SH-wave polarization model
wave4 = PolarizationModel(wave_type='L', phi=350., vl=300.)  # Generate a Love-wave polarization model
wave5 = PolarizationModel(wave_type='R', phi=30., vr=300.,
                          xi=-45.)  # Generate a Rayleigh-wave polarization model with a
# Rayleigh wave ellipticity angle of -45 degrees.


########################################################################################################################
# Now we populate our signal with the computed amplitudes by setting a spike with the respective amplitude onto the
# different locations of the time axis. Then we convolve the data with the Ricker wavelet to generate our synthetic
# test seismograms.

signal1[100, 2:] = wave1.polarization[2:].real.T
signal1[100, 0:2] = wave1.polarization[0:2].imag.T
signal2[300, :] = wave2.polarization.real.T
signal3[500, :] = wave3.polarization.real.T
signal4[700, :] = wave4.polarization.real.T
signal5[900, 2:] = np.real(wave5.polarization[2:].T)
signal5[900, 0:2] = np.imag(wave5.polarization[0:2].T)

for j in range(0, signal1.shape[1]):
    signal2[:, j] = convolve(signal2[:, j], wavelet, mode='same')
    signal3[:, j] = convolve(signal3[:, j], wavelet, mode='same')
    signal4[:, j] = convolve(signal4[:, j], wavelet, mode='same')
    if j == 0 or j == 1:  # Special case for horizontal translational components of the Rayleigh wave
        signal5[:, j] = convolve(signal5[:, j], wavelet_hilb, mode='same')
        signal1[:, j] = convolve(signal1[:, j], wavelet_hilb, mode='same')
    else:
        signal5[:, j] = convolve(signal5[:, j], wavelet, mode='same')
        signal1[:, j] = convolve(signal1[:, j], wavelet, mode='same')

signal = signal1 + signal2 + signal3 + signal4 + signal5  # sum all signals together

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(t, signal[:, 0], 'k:', label='traN')
plt.plot(t, signal[:, 1], 'k--', label='traE')
plt.plot(t, signal[:, 2], 'k', label='traZ')
plt.plot(t, signal[:, 3], 'r:', label='rotN')
plt.plot(t, signal[:, 4], 'r--', label='rotE')
plt.plot(t, signal[:, 5], 'r', label='rotZ')
plt.text((100 - 120) * dt, 0.7, 'P-wave')
plt.text((300 - 120) * dt, 0.7, 'SV-wave')
plt.text((500 - 120) * dt, 0.7, 'SH-wave')
plt.text((700 - 120) * dt, 0.7, 'Love-wave')
plt.text((900 - 120) * dt, 0.7, 'Rayleigh-wave')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Time (s)')

########################################################################################################################
# Note that the translational components and the rotational components have different units. The PolarizationModel class
# yields the amplitudes in acceleration (m/s/s) for the translational components and in rotation rate (rad/s) for the
# rotational components. Since the rotational signal scales with the local wave slowness, it is barely visible in the
# plot above. For polarization analysis, we want to make sure that both the translational and the rotational components
# have the same units and that the amplitudes are comparable, we therefore divide the translational components by a
# scaling velocity and plot the data again. Here we choose a scaling velocity of 400 m/s. Applying a scaling velocity to
# the recorded data is a crucial step when processing real data. Choose a scaling velocity that ensures that the
# translational and rotational signals have comparable amplitudes.

scaling_velocity = 400.
signal[:, 0:3] /= scaling_velocity  # Apply scaling velocity to the translational data

signal += 0.1 * signal.max().max() * np.random.random((N, 6))

plt.figure(figsize=(10, 5))
plt.plot(t, signal[:, 0], 'k:', label='traN')
plt.plot(t, signal[:, 1], 'k--', label='traE')
plt.plot(t, signal[:, 2], 'k', label='traZ')
plt.plot(t, signal[:, 3], 'r:', label='rotN')
plt.plot(t, signal[:, 4], 'r--', label='rotE')
plt.plot(t, signal[:, 5], 'r', label='rotZ')
plt.text((100 - 120) * dt, 0.7 / scaling_velocity, 'P-wave')
plt.text((300 - 120) * dt, 0.7 / scaling_velocity, 'SV-wave')
plt.text((500 - 120) * dt, 0.7 / scaling_velocity, 'SH-wave')
plt.text((700 - 120) * dt, 0.7 / scaling_velocity, 'Love-wave')
plt.text((900 - 120) * dt, 0.7 / scaling_velocity, 'Rayleigh-wave')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Time (s)')

########################################################################################################################
# To make the synthetics accessible to TwistPy, we convert them to an Obspy Stream object.

data = Stream()
for n in range(signal.shape[1]):
    trace = Trace(data=signal[:, n], header={"delta": t[1] - t[0], "npts": int(signal.shape[0]), "starttime": 0.})
    data += trace

########################################################################################################################
# Now to the actual wave type classification. If we haven't done so already, we first need to train a machine learning
# model, that allows us to classify the waves. For this, we set up a support vector machine. In our example, we consider
# wave parameters that are typical for the near surface, so we give the support vector machine a fitting name

svm = SupportVectorMachine(name='love_wave')

########################################################################################################################
# Now we can train the model. For details, please check the example on how to train a
# 'twistpy.machinelearning.SupportVectorMachine' object. In short, we want to train the model for wave parameters
# that are typical for the near surface, and we want to be able to identify P, SV, SH and Rayleigh waves. This means
# that we do not make a distinction between Love and SH waves here, and Love waves will simply be contained in the SH
# wave class as the special case of horizontally propagating SH waves. Additionally, we make use of a Noise class, for
# analysis windows with a random polarization. We allow waves to arrive from all directions (azimuth range [0 360]
# degrees and inclination range [0 90] degrees).

svm.train(wave_types=['R', 'P', 'SV', 'L', 'Noise'],
          N=5000, scaling_velocity=scaling_velocity, vp=(400, 3000), vp_to_vs=(1.7, 2.4), vl=(100, 3000),
          vr=(100, 3000), phi=(0, 360), theta=(0, 90), xi=(-90, 90), free_surface=True, C=1, kernel='rbf')

########################################################################################################################
# Now that we have trained the model, we can set up our analysis. We will perform 6C polarization analysis in the time
# domain and use a sliding time window that is 0.05 s long (50 samples) with an overlap between subsequent windows of
# 50%.

window = {'window_length_seconds': 20. * dt, 'overlap': 0.5}
analysis = TimeDomainAnalysis(traN=data[0], traE=data[1], traZ=data[2], rotN=data[3], rotE=data[4], rotZ=data[5],
                              window=window, scaling_velocity=scaling_velocity, timeaxis='rel')

########################################################################################################################
# To classify the waves, we simply do (yielding a classification of the first eigenvector of the covariance matrix):

analysis.classify(svm=svm, eigenvector_to_classify=0)
classification = analysis.classification['0']
t_windows = analysis.t_windows  # Positions of the sliding time windows where the classification was performed

#  Wave parameter estimation
from twistpy import EstimatorConfiguration

est = EstimatorConfiguration(wave_types=['L', 'R'], method='ML', scaling_velocity=scaling_velocity,
                             use_ml_classification=True,
                             svm=svm)
analysis.polarization_analysis(estimator_configuration=est)
analysis.plot_polarization_analysis(wave_types=['L', 'R'], dop_clip=0.9)
