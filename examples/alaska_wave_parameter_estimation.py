r"""
6-C Wave parameter estimation: gulf of alaska earthquake
========================================================
This tutorial will teach you how to use TwistPy to process six component seismic data as recorded by collocated
translational and rotational seismometers. The data used in this tutorial was recorded at the large ring laser
observatory ROMY in Furstenfeldbruck, Germany after the January, 23rd 2018 M7.8 gulf of Alaska earthquake
(Occurence Time: 2018-01-23 09:31:40 UTC).
"""

from obspy import read

from twistpy.polarization import TimeFrequencyAnalysis6C, EstimatorConfiguration

########################################################################################################################
# We start by reading in the 6DOF data and apply some basic pre-processing to it using standard ObsPy functionality.
# To ensure that the amplitudes of all six time series are on the same order of magnitude and have the same unit,
# we apply a scaling velocity to the translational components:

data = read('../example_data/ROMY_gulf_of_alaska_teleseism.mseed')
scaling_velocity = 4500.

for n, trace in enumerate(data):
    trace.detrend('spline', order=5, dspline=100)
    trace.trim(starttime=trace.stats.starttime, endtime=trace.stats.endtime - 4500)
    trace.taper(0.2)
    if n < 3:
        trace.data /= scaling_velocity

########################################################################################################################
# Now we are ready to set up the 6DOF polarization analysis problem. The spectral matrices from which the polarization
# attributes are estimated are computed in a time-frequency window. We choose the window to be frequency-dependent and
# extend over 1 period (1/f) in the time direction and over 0.01 Hz in the frequency direction:

window = {'number_of_periods': 1, 'frequency_extent': 0.01}

########################################################################################################################
# Now we are ready to set up the 6DOF polarization analysis problem.  We want to perform the analysis in the
# time-frequency domain, so we set up a TimeFrequencyAnalysis6C object and feed it our data and the polarization
# analysis window. Additionally, we restrict the analysis to the frequency range between 0.01 and 0.15 Hz, where
# we expect surface waves to dominate. To reduce the computational effort, we only compute wave paramters at every 20th
# sample in time and in frequency:

analysis = TimeFrequencyAnalysis6C(traN=data[0], traE=data[1], traZ=data[2], rotN=data[3], rotE=data[4], rotZ=data[5],
                                   window=window, dsfacf=20, dsfact=20, frange=[0.01, 0.15])

########################################################################################################################
# Now we set up an EstimaatorConfiguration, specifying for which wave types we want to estimate wave parameters (in
# this example only Rayleigh waves), what kind of estimator we want to use, and the range of wave parameters that are
# tested.

est = EstimatorConfiguration(wave_types=['R'], method='DOT', scaling_velocity=scaling_velocity,
                             use_ml_classification=False, vr=[3000, 4000, 100], xi=[-90, 90, 2], phi=[150, 210, 1],
                             eigenvector=0)

########################################################################################################################
# Let's start the analysis!

analysis.polarization_analysis(estimator_configuration=est)

########################################################################################################################
# Once the wave parameters are computed, we can access them as a dictionary obtained from the attribute wave_parameters.

print(analysis.wave_parameters)

########################################################################################################################
# To directly extract the Rayleigh wave parameters, we can access them in the following way (here we extract the
# Rayleigh wave propagation azimuth). With the corresponding time and frequency vectors.

azi_rayleigh = analysis.wave_parameters['R']['phi']
f = analysis.f_pol
t = analysis.t_pol

########################################################################################################################
# You can either plot the estimated parameters yourself or make use of the implemented plotting routines, where lh_min
# and lh_max determine the estimator power range for which the parameters are plotted. For the 'DOT' method that we used
# here, this corresponds to a simple likelihood, meaning that we only want to plot the results at time-frequency pixels
# where the Rayleigh wave model fits the data with a likelihoood larger than 0.7.

analysis.plot_wave_parameters(estimator_configuration=est, lh_min=0.7, lh_max=1.0)
