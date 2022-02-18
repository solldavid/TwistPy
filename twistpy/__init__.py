"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2022
:license:
    None
"""
import inspect
import os
import sys

from twistpy.core import TimeDomainAnalysis, TimeFrequencyAnalysis, DispersionAnalysis
from twistpy.polarizationmodel import PolarizationModel

LOCAL_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, LOCAL_PATH)
