r"""
6-C Polarization Analysis: Wave parameter estimation
====================================================
This tutorial will teach you how to use TwistPy to process six component seismic data as recorded by collocated
translational and rotational seismometers. The data used in this tutorial was recorded at the large ring laser
observatory ROMY in Furstenfeldbruck, Germany after the January, 23rd 2018 M7.8 gulf of Alaska earthquake
(Occurence Time: 2018-01-23 09:31:40 UTC).
"""

from obspy import read

from twistpy.polarization import TimeFrequencyAnalysis6C, EstimatorConfiguration

########################################################################################################################
# We start by reading in the 6DOF data and apply some basic pre-processing to it using standard ObsPy functionality
# To ensures that the amplitudes of all six time series are on the same order of magnitude and have the same unit,
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
# Now we are ready to set up the 6DOF polarization analysis problem.  The spectral matrices from which the polarization
# attributes are estimated are computed in a time-frequency window. We choose the window to be frequency-dependent and
# extend over 1 period (1/f) in the time direction and over 0.01 Hz in the frequency direction:

analysis = TimeFrequencyAnalysis6C(traN=data[0], traE=data[1], traZ=data[2], rotN=data[3], rotE=data[4], rotZ=data[5],
                                   window=window, dsfacf=20, dsfact=2, frange=[0.01, 0.15])

est = EstimatorConfiguration(wave_types=['R'], method='MVDR', scaling_velocity=scaling_velocity,
                             use_ml_classification=False, vr=[2000, 4000, 200], xi=[-90, 90, 2], phi=[-30, 10, 2])
analysis.polarization_analysis(estimator_configuration=est)
