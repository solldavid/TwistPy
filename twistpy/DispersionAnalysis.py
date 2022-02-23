"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2022
:license:
    None
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

from builtins import *

from obspy import Trace


class DispersionAnalysis:

    def __init__(self, traN: Trace = None, traE: Trace = None, traZ: Trace = None, rotN: Trace = None,
                 rotE: Trace = None,
                 rotZ: Trace = None, method: str = 'pca'):
        self.traN, self.traE, self.traZ, self.rotN, self.rotE, self.rotZ = traN, traE, traZ, rotN, rotE, rotZ

        # Assert that input traces are ObsPy Trace objects
        assert isinstance(self.traN, Trace) and isinstance(self.traE, Trace) and isinstance(self.traZ, Trace) \
               and isinstance(self.rotN, Trace) and isinstance(self.rotE, Trace) and isinstance(self.rotZ, Trace), \
            "Input data must be of class obspy.core.Trace()!"

        # Assert that all traces have the same number of samples
        assert self.traN.stats.npts == self.traE.stats.npts and self.traN.stats.npts == self.traZ.stats.npts \
               and self.traN.stats.npts == self.rotN.stats.npts and self.traN.stats.npts == self.rotE.stats.npts \
               and self.traN.stats.npts == self.rotZ.stats.npts, \
            "All six traces must have the same number of samples!"
