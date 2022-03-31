import numpy as np
from obspy.core import Trace, Stream

from twistpy.polarization.time import TimeDomainAnalysis3C
from twistpy.polarization.timefrequency import TimeFrequencyAnalysis3C

rng = np.random.default_rng(1)

data = rng.random((20, 3))

N = Trace(data=data[:, 0], header={'starttime': 0, 'delta': 1})
E = Trace(data=data[:, 1], header={'starttime': 0, 'delta': 1})
Z = Trace(data=data[:, 2], header={'starttime': 0, 'delta': 1})


def test_td3cpa():
    window = {'window_length_seconds': 5., 'overlap': 1.}
    tda = TimeDomainAnalysis3C(N=N, E=E, Z=Z, window=window, verbose=False)
    tda.polarization_analysis()
    data_filtered = tda.filter(plot_filtered_attributes=True, dop=[0, 1], elli=[0.1, 0.5], inc1=[0, 85],
                               inc2=[0, 85], azi1=[0, 170], azi2=[0, 170])
    assert isinstance(data_filtered, Stream)
    assert tda.plot(show=False) is None
    assert tda.dop.shape == (15,)


def test_tfd3cpa():
    window = {'number_of_periods': 1, 'frequency_extent': 0.01}

    tfa = TimeFrequencyAnalysis3C(N=N, E=E, Z=Z, window=window, verbose=False)
    tfa.polarization_analysis()
    data_filtered = tfa.filter(plot_filtered_attributes=True, dop=[0, 1], elli=[0.1, 0.5], inc1=[0, 85],
                               inc2=[0, 85], azi1=[0, 170], azi2=[0, 170])
    assert isinstance(data_filtered, Stream)
    assert tfa.plot(show=False) is None
    assert tfa.dop.shape == (11, 20)
