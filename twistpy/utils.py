"""
TwistPy utility functions
"""

import pickle
from typing import Union

from twistpy.DispersionAnalysis import DispersionAnalysis
from twistpy.TimeDomainAnalysis import TimeDomainAnalysis
from twistpy.TimeFrequencyAnalysis import TimeFrequencyAnalysis


def load_analysis(file: str = None) -> Union[TimeDomainAnalysis, TimeFrequencyAnalysis, DispersionAnalysis]:
    """Read a TwistPy analysis object from the disk.

    Parameters
    ----------
    file : :obj:`str`
        File name (include absolute path if the file is not in the current working directory)

    Returns
    -------
    obj : :obj:`~twistpy.TimeDomainAnalysis.TimeDomainAnalysis` or :obj:`~twistpy.TimeFrequencyAnalysis.TimeFrequencyAnalysis` or :obj:`~twistpy.DispersionAnalysis.DispersionAnalysis`
    """

    if file is None:
        raise Exception('Please specify the name of the file that you want to load!')
    fid = open(file, 'rb')
    obj = pickle.load(fid)
    fid.close()
    return obj
