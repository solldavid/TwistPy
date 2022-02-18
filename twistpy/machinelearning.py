"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2022
:license:
    None
"""
import pickle
import sys
from os.path import exists

import numpy as np
import pandas as pd
from numpy.core.umath_tests import inner1d
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from twistpy.polarizationmodel import PolarizationModel


class SupportVectorMachine:
    def __init__(self, name: str = None):
        if name is None:
            raise Exception('Please specify a name for this SupportVectorMachine!')
        self.name = name

    def train(self, N: int = 5000, scaling_velocity: float = 1., vp: tuple = (50., 2000.), vp_to_vs: tuple = (1.7, 2.4),
              vl: tuple = (50, 2000), vr: tuple = (50, 2000), phi: tuple = (0, 360), theta: tuple = (0, 90),
              xi: tuple = (-90, 90), free_surface: bool = True, C: float = 1, kernel: str = 'rbf'):
        pkl_filename = sys.path[0] + "/SVC_models/" + self.name + ".pkl"
        if exists(pkl_filename):
            print(f"A trained model already exists with this name and is saved at '{pkl_filename}'\n")
            print('Nothing will be done! Please delete the file above if you want to retrain this model.')
            return
        df = pd.DataFrame(index=np.arange(0, 6 * N),
                          columns=['t1_real', 't2_real', 't3_real', 'r1_real', 'r2_real', 'r3_real',
                                   't1_imag', 't2_imag', 't3_imag', 'r1_imag', 'r2_imag', 'r3_imag', 'wave_type'])
        phi = np.random.uniform(phi[0], phi[1], (N, 1))
        theta = np.random.uniform(theta[0], theta[1], (N, 1))
        xi = np.random.uniform(xi[0], xi[1], (N, 1))
        vr = np.random.uniform(vr[0], vr[1], (N, 1))
        vl = np.random.uniform(vl[0], vl[1], (N, 1))
        vp_to_vs = np.random.uniform(vp_to_vs[0], vp_to_vs[1], (N, 1))
        vp = np.random.uniform(vp[0], vp[1], (N, 1))

        print('Generating random polarization models for training! \n')
        # Generate random P-wave polarization models drawn from a uniform distribution
        pm_p = PolarizationModel(wave_type='P', vp=vp, vs=vp / vp_to_vs, phi=phi, theta=theta,
                                 scaling_velocity=scaling_velocity, free_surface=free_surface)
        pol_p = np.random.choice([-1, 1], (1, N)) * pm_p.polarization_vector

        # Generate random SV-wave polarization models drawn from a uniform distribution
        pm_sv = PolarizationModel(wave_type='SV', vp=vp, vs=vp / vp_to_vs, phi=phi, theta=theta,
                                  scaling_velocity=scaling_velocity, free_surface=free_surface)
        pol_sv = np.random.choice([-1, 1], (1, N)) * pm_sv.polarization_vector  # direction can be positive or negative

        # Generate random SH-wave polarization models drawn from a uniform distribution
        pm_sh = PolarizationModel(wave_type='SH', vs=vp / vp_to_vs, phi=phi, theta=theta,
                                  scaling_velocity=scaling_velocity, free_surface=free_surface)
        pol_sh = np.random.choice([-1, 1], (1, N)) * pm_sh.polarization_vector

        # Generate random Rayleigh-wave polarization models drawn from a uniform distribution
        pm_r = PolarizationModel(wave_type='R', vr=vr, phi=phi, xi=xi,
                                 scaling_velocity=scaling_velocity, free_surface=free_surface)
        pol_r = np.random.choice([-1, 1], (1, N)) * pm_r.polarization_vector

        # Generate random Love-wave polarization models drawn from a uniform distribution
        pm_l = PolarizationModel(wave_type='L', vl=vl, phi=phi,
                                 scaling_velocity=scaling_velocity, free_surface=free_surface)
        pol_l = np.random.choice([-1, 1], (1, N)) * pm_l.polarization_vector

        # Generate random polarization vectors for Noise class
        u1_real: np.ndarray = np.random.normal(0.0, 1, size=(6, N))
        u1_imag: np.ndarray = np.random.normal(0.0, 1, size=(6, N))
        u1: np.ndarray = u1_real + 1j * u1_imag
        u1: np.ndarray = (u1 / np.linalg.norm(u1, axis=0))
        u1: np.ndarray = np.random.choice([-1, 1]) * u1
        gamma = np.arctan2(2 * inner1d(u1.real.T, u1.imag.T),
                           inner1d(u1.real.T, u1.real.T) -
                           inner1d(u1.imag.T, u1.imag.T))
        phi1 = -0.5 * gamma
        pol_noise = np.exp(1j * phi1) * u1

        for n in range(N):
            df.loc[n + 0 * N] = pol_r[:, n].real.tolist() + pol_r[:, n].imag.tolist() + ['R']
            df.loc[n + 1 * N] = pol_p[:, n].real.tolist() + pol_p[:, n].imag.tolist() + ['P']
            df.loc[n + 2 * N] = pol_sv[:, n].real.tolist() + pol_sv[:, n].imag.tolist() + ['SV']
            df.loc[n + 3 * N] = pol_l[:, n].real.tolist() + pol_l[:, n].imag.tolist() + ['L']
            df.loc[n + 4 * N] = pol_sh[:, n].real.tolist() + pol_sh[:, n].imag.tolist() + ['SH']
            df.loc[n + 5 * N] = pol_noise[:, n].real.tolist() + pol_noise[:, n].imag.tolist() + ['noise']
        print('Training Support Vector Machine!')
        df = df.dropna()
        X = df.drop(['wave_type'], axis='columns')
        y = df.wave_type
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
        model = SVC(kernel=kernel, C=C)
        model.fit(Xtrain, ytrain)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)

        print(
            f"Training successfully completed. Model score on independent test data is '{model.score(Xtest, ytest)}! \n'")
        print(f"Model has been saved as '{pkl_filename}'! \n'")

    def load_model(self):
        file_name = sys.path[0] + "/SVC_models/" + self.name + '.pkl'
        if not exists(file_name):
            raise Exception(f"The model '{self.name}' has not been trained yet")
        else:
            with open(file_name, 'rb') as file:
                model = pickle.load(file)
        return model
