r"""
Six-component dispersion analysis
=================================
In this tutorial, you will learn how to extract dispersion curves from single-station six-component recordings of
ambient noise.
"""

import numpy as np
from obspy.core import read

from twistpy.polarization import DispersionAnalysis
from twistpy.polarization.machinelearning import SupportVectorMachine

scaling_velocity = 800.

data = read('../example_data/6C_noise_data_long.mseed')

for i, trace in enumerate(data):
    if i < 3:
        trace.differentiate()
        trace.data /= scaling_velocity
    else:
        trace.data -= np.median(trace.data)
    trace.taper(0.05)

svm = SupportVectorMachine(name='dispersion_analysis')
svm.train(wave_types=['R', 'L', 'P', 'SV', 'Noise'], scaling_velocity=scaling_velocity, phi=(0, 360), vp=(100, 3000),
          vp_to_vs=(1.7, 2.4), vr=(50, 3000), vl=(50, 3000), xi=(-90, 90), theta=(0, 70))

window = {'number_of_periods': 2, 'overlap': 0.}
da = DispersionAnalysis(traN=data[1], traE=data[2], traZ=data[0], rotN=data[4], rotE=data[5], rotZ=data[3],
                        window=window, scaling_velocity=scaling_velocity, verbose=True, fmin=1., fmax=20., octaves=0.25,
                        svm=svm)
da.plot_dispersion_curves()
test = 1
