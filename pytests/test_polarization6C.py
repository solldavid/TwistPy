import numpy as np
from obspy.core import Trace

from twistpy.polarization.time import TimeDomainAnalysis6C

rng = np.random.default_rng(1)

data = rng.random((20, 6))

traN = Trace(data=data[:, 0], header={'starttime': 0, 'delta': 1})
traE = Trace(data=data[:, 1], header={'starttime': 0, 'delta': 1})
traZ = Trace(data=data[:, 2], header={'starttime': 0, 'delta': 1})
rotN = Trace(data=data[:, 3], header={'starttime': 0, 'delta': 1})
rotE = Trace(data=data[:, 4], header={'starttime': 0, 'delta': 1})
rotZ = Trace(data=data[:, 5], header={'starttime': 0, 'delta': 1})


def test_tda():
    window = {'window_length_seconds': 5., 'overlap': 1.}
    tda = TimeDomainAnalysis6C(traN=traN, traE=traE, traZ=traZ, rotN=rotN, rotE=rotE, rotZ=rotZ, scaling_velocity=1.,
                               free_surface=True, window=window, verbose=False)
