from __future__ import (absolute_import, division, print_function, unicode_literals)

from builtins import *
from typing import List


class EstimatorConfiguration:
    """
    :param method: Specifies the way that the polarization fit is determined
        'MUSIC': MUSIC algorithm
        'ML': Maximum likelihood method
        'DOT': Minimize great-circle distance / dot  product (corresponding to the angle between the mode
        polarization vector and the polarization direction in the data) on the 5-sphere between the
        measured and tested polarization vector (DEFAULT)

    :param search: Specifies the optimization procedure
        'grid': Simple grid search across the parameter space (DEFAULT)
        'global': Global optimization for speed-up using a differential evolution algorithm

    Range of parameters used to compute steering vectors. Tuple of the form range=(range_min, range_max, range_increment)
    If 'search' == 'global', the range_increment argument is ignored

    :param vl_range: Range of Love wave velocities (in m/s) to be tested.
    :param vr_range: Range of Rayleigh wave velocities (in m/s) to be tested.
    :param vp_range: Range of P-wave velocities (in m/s) to be tested.
    :param vs_range: Range of S-wave velocities (in m/s) to be tested.
    :param theta_range: Range of incidence angles (in degrees) to be tested.
    :param phi_range: Range of azimuth angles (in degrees) to be tested.
    :param xi_range: Range of Rayleigh wave ellipticity angles (in rad) to be tested.
    """

    def __init__(self, wave_types: List[str] = None, method: str = 'MUSIC', search: str = 'grid'):
        if wave_types is None:
            wave_types = ['P', 'SV', 'SH', 'L', 'R']
