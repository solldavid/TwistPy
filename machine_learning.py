"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2022
:license:
    None
"""
import sys
from os.path import exists
import numpy as np
from core import PolarizationModel
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
sys.path.insert(0, "./SVC_models")


class SupportVectorMachine:
    def __init__(self, name: str = None, scaling_velocity: float=1.):
        if name is None:
            raise Exception('Please specify a name for this SupportVectorMachine!')
        self.name = name

    def train(self, N: int = 5000, vp: tuple = (50., 2000.), vp_to_vs: tuple = (1.7, 2.4), vl: tuple = (50, 2000),
              vr: tuple = (50, 2000), phi: tuple = (0, 360), theta: tuple = (0, 90), xi: tuple = (-90, 90)):
        df = pd.DataFrame(index=np.arange(0, 5 * N),
                          columns=['t1_real', 't2_real', 't3_real', 'r1_real', 'r2_real', 'r3_real',
                                   't1_imag', 't2_imag', 't3_imag', 'r1_imag', 'r2_imag', 'r3_imag', 'wave_type'])
        phi = np.random.uniform(phi[0], phi[1], (N, 1))
        theta = np.random.uniform(theta[0], theta[1], (N, 1))
        xi = np.random.uniform(xi[0], xi[1], (N, 1))
        vr = np.random.uniform(vr[0], vr[1], (N, 1))
        vl = np.random.uniform(vl[0], vl[1], (N, 1))
        vp_to_vs = np.random.uniform(vp_to_vs[0], vp_to_vs[1], (N, 1))
        vp = np.random.uniform(vp[0], vp[1], (N, 1))
        pol = PolarizationModel(wave_type='P', vp=vp, vs=vp / vp_to_vs, phi=phi, theta=theta)


    def load_model(self):
        file_name = self.name + 'pkl'
        if not exists(file_name):
            pass