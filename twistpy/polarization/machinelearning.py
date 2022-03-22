import pickle
import sys
from os.path import exists, join
from typing import Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from twistpy.polarization.model import PolarizationModel


class SupportVectorMachine:
    """Support vector machine for wave type classification based on six component polarization analysis.

    Used to train and classify wave types via 6-C polarization analysis. This class merely exists for convenience.
    The core functionality of this class inherits from :obj:`sklearn.svm.SVC`.

    Parameters
    ----------
    name : :obj:`str`
        Name of the support vector machine
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.C: float = None
        self.kernel: str = None
        self.free_surface: bool = None
        self.N: int = None
        self.scaling_veloctiy: float = None
        self.vp: Tuple[float, float] = None
        self.vp_to_vs: Tuple[float, float] = None
        self.vl: Tuple[float, float] = None
        self.vr: Tuple[float, float] = None
        self.phi: Tuple[float, float] = None
        self.theta: Tuple[float, float] = None
        self.xi: Tuple[float, float] = None
        self.plot_confusion_matrix: bool = None
        self.scaling_velocity: float = None
        if not self.name:
            raise Exception('Please specify a name for this SupportVectorMachine!')

    def train(self, wave_types: List[str] = ['R', 'L', 'P', 'SV', 'SH', 'Noise'],
              N: int = 5000, scaling_velocity: float = 1., vp: Tuple[float, float] = (50., 2000.),
              vp_to_vs: Tuple[float, float] = (1.7, 2.4), vl: Tuple[float, float] = (50, 2000),
              vr: Tuple[float, float] = (50, 2000), phi: Tuple[float, float] = (0, 360),
              theta: Tuple[float, float] = (0, 90), xi: Tuple[float, float] = (-90, 90),
              free_surface: bool = True, C: float = 1, kernel: str = 'rbf', gamma: Union[str, float] = 'scale',
              plot_confusion_matrix: bool = True) -> None:
        """Train support vector machine with random polarization models from the specified parameter range.

        Parameters
        ----------
        wave_types : :obj:`list` of :obj:`str`, default=['R', 'L', 'P', 'SV', 'SH', 'Noise']
            List of wave-types that are used for training.

            |  'P':     P-wave
            |  'SV':    SV-wave
            |  'SH':    SH-wave
            |  'L':     Love-wave
            |  'R':     Rayleigh-wave
        N : :obj:`int`, default=5000
            Number of randomly generated polarization models for each wave type that are used for training.
        scaling_velocity : :obj:`float`, default=1.
            Scaling velocity (in m/s) that was applied to the translational components of the real data.
        vp : :obj:`tuple`
            P-wave velocity range (in m/s) from which the random parametrization of the training set of
            polarization vectors is drawn as (vp_min, vp_max).
        vp_to_vs : :obj:`tuple`
            Range of P-to-S wave velocity ratios from which the random parametrization of the training set of
            polarization vectors is drawn (vp_to_vs_min, vp_to_vs_max).
        vl : :obj:`tuple`
            Range of Love wave velocities in m/s.
        vr : :obj:`tuple`
            Range of Rayleigh wave velocities in m/s.
        phi : :obj:`tuple`
            Azimuth angle range in degrees.
        theta : :obj:`tuple`
            Inclination angle range in degrees.
        xi : :obj:`tuple`
            Rayleigh wave ellipticity angle range in degrees.
        free_surface : :obj:`bool`, default=True
            Specifies whether free-surface polarization models apply
        C : :obj:`float`, default=1.0
            Regularization parameter for the support vector machine. See :obj:`sklearn.svm.SVC`.
        kernel : :obj:`str` or *callable*, default='rbf'
            Kernel type used for the support vector machine. Defaults to a radial basis function kernel.
            See :obj:`sklearn.svm.SVC`.
        gamma : :obj:`str` or float, default='scale'
            Kernel coefficient. See :obj:`sklearn.svm.SVC`.
        plot_confusion_matrix : :obj:`bool`, default=True
            Specify whether a confusion matrix will be plotted after training
        """
        # Initial sanity checks
        wtypes_implemented = ['R', 'L', 'P', 'SV', 'SH', 'Noise']
        for w_type in wave_types:
            assert w_type in wtypes_implemented, f"Invalid wave type specified: {w_type}! Must be in " \
                                                 f"[{wtypes_implemented}]"
        # Set class attributes
        self.C, self.kernel, self.free_surface = C, kernel, free_surface
        self.N, self.scaling_velocity = N, scaling_velocity
        self.vp, self.vp_to_vs, self.vl, self.vr = vp, vp_to_vs, vl, vr
        self.phi, self.theta, self.xi = phi, theta, xi
        self.plot_confusion_matrix = plot_confusion_matrix

        pkl_filename = join(sys.path[0], "SVC_models", self.name + ".pkl")
        if exists(pkl_filename):
            print(f"A trained model already exists with this name and is saved at '{pkl_filename}'")
            print('Nothing will be done! Please delete the file above if you want to re-train this model.')
            return
        df = pd.DataFrame(index=np.arange(0, len(wave_types) * N),
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
        # Generate random Rayleigh-wave polarization models drawn from a uniform distribution
        columns = ['t1_real', 't2_real', 't3_real', 'r1_real', 'r2_real', 'r3_real',
                   't1_imag', 't2_imag', 't3_imag', 'r1_imag', 'r2_imag', 'r3_imag']
        for it, w_type in enumerate(wave_types):

            if w_type == 'Noise':
                u1_real: np.ndarray = np.random.normal(0.0, 1, size=(6, N))
                u1_imag: np.ndarray = np.random.normal(0.0, 1, size=(6, N))
                u1: np.ndarray = u1_real + 1j * u1_imag
                u1: np.ndarray = (u1 / np.linalg.norm(u1, axis=0))
                u1: np.ndarray = np.random.choice([-1, 1]) * u1
                gamma_samson = np.arctan2(2 * np.einsum('ij,ij->j', u1.real, u1.imag, optimize=True),
                                          np.einsum('ij,ij->j', u1.real, u1.real, optimize=True) -
                                          np.einsum('ij,ij->j', u1.imag, u1.imag, optimize=True))
                phi1 = -0.5 * gamma_samson
                pol_noise = np.exp(1j * phi1) * u1
                df.loc[it * N:(it + 1) * N - 1, 'wave_type'] = w_type
                for idx, column in enumerate(columns):
                    if idx < 6:
                        df.loc[it * N:(it + 1) * N - 1, column] = pol_noise[idx, :].real.tolist()
                    else:
                        df.loc[it * N:(it + 1) * N - 1, column] = pol_noise[idx - 6, :].imag.tolist()
            else:
                # Generate random 6C wave polarization models drawn from a uniform distribution
                pm = PolarizationModel(wave_type=w_type, vr=vr, vp=vp, vs=vp / vp_to_vs, phi=phi, xi=xi,
                                       theta=theta, vl=vl, scaling_velocity=scaling_velocity, free_surface=free_surface)
                # Direction of polarization can either be positive or negative
                pol = np.random.choice([-1, 1], (1, N)) * pm.polarization_vector
                df.loc[it * N:(it + 1) * N - 1, 'wave_type'] = w_type
                for idx, column in enumerate(columns):
                    if idx < 6:
                        df.loc[it * N:(it + 1) * N - 1, column] = pol[idx, :].real.tolist()
                    else:
                        df.loc[it * N:(it + 1) * N - 1, column] = pol[idx - 6, :].imag.tolist()

        print('Training Support Vector Machine!')
        df = df.dropna()
        X = df.drop(['wave_type'], axis='columns')
        y = df.wave_type
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
        model = SVC(kernel=kernel, C=C, gamma=gamma)
        model.fit(Xtrain, ytrain)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)

        print(
            f"Training successfully completed. Model score on independent test data is '{model.score(Xtest, ytest)}'!")
        print(f"Model has been saved as '{pkl_filename}'!'")

        if plot_confusion_matrix:
            from sklearn.metrics import ConfusionMatrixDisplay
            ConfusionMatrixDisplay.from_predictions(ytest, model.predict(Xtest), labels=model.classes_,
                                                    cmap='binary')
            plt.show()

    def load_model(self: str = None) -> SVC:
        """
        Loads a previously trained support vector machine from disk
        :return: sklearn.svm.SVC object
        """
        file_name = join(sys.path[0], "SVC_models", self.name + '.pkl')
        if not exists(file_name):
            raise Exception(f"The model '{self.name}' has not been trained yet")
        else:
            with open(file_name, 'rb') as file:
                model = pickle.load(file)
        return model
