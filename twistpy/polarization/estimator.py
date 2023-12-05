from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import ValueError
from typing import List, Tuple, Callable

import numpy as np
from numpy import ndarray

from twistpy.polarization.machinelearning import SupportVectorMachine
from twistpy.polarization.model import PolarizationModel6C


class EstimatorConfiguration:
    r"""Configure the type of estimator used for 6C polarization analysis.

    Parameters
    ----------
    wave_types : :obj:`list` of :obj:`str`, default = ['R', 'L', 'P', 'SV', 'SH']
        List of wave types for which wave parameters are estimated.
    method : :obj:`str`, default='ML'
        Method that is used for the estimation of wave parameters and polarization attributes.

        .. hint:: 'ML': Machine Learning: Initially classify the wave types using a machine learning model. Wave
            parameters are then directly estimated from the specified eigenvector. This is the most efficient way to
            estimate wave parameters with 6C polarization analysis

            'MUSIC': Wave parameters are estimated using multiple signal classification. This is a grid search approach
            and can be computationally expensive to compute. Enables the estimation of wave parameters for
            multiple overlapping signals.

            'MVDR': Wave parameters are estimated using the minimum variance distortionless response or Capon method
            (grid search approach).

            'BARTLETT': Wave parameters are estimated using the Bartlett method (grid search approach).

            'DOT': Minimize great-circle distance / dot  product (corresponding to the angle between the model
            polarization vector and the polarization direction in the data) on the 5-sphere between the
            measured and tested polarization vector
    free_surface : :obj:`bool`, default=True
        Specify whether free-surface polarization models should be used
    scaling_velocity : :obj:`float`, default = 1.
        Scaling velocity applied to the translational components in m/s
    use_ml_classification : :obj:`bool`, default = True
        For grid-search approaches: Specify whether an initial step of wave classification is performed using a
        machine learning model. Wave parameters are then only estimated in windows where the wave type of interest is
        detected. Results in a speed-up.
    svm : :obj:`~twistpy.machinelearning.SupportVectorMachine`, optional
        Pre-trained support vector machine for wave type classification. Needs to be provided if method='ML' or
        use_ml_classification = True.
    eigenvector : :obj:`int`, default=0
        Integer value identifying the eigenvector that will be used for wave parameter estimation. The eigenvectors are
        sorted in descending order of their corresponding eigenvalue

        |  If 0: first eigenvector, corresponding to the dominant signal in
                the time window (associated with the largest eigenvalue).
    music_signal_space_dimension : :obj:´int´, default=1
        Specify the number of overlapping waves for which wave parameters will be estimated using the MUSIC algorithm.
    vp : :obj:`tuple` (vp_min, vp_max, increment)
        Define the search space for the P-wave velocity in m/s for grid-search methods.
    vp_to_vs : :obj:`tuple` (vp_to_vs_min, vp_to_vs_max, increment)
        Define the search space for the P-wave to S-wave velocity ratio.
    vl : :obj:`tuple` (vl_min, vl_max, increment)
        Define the search space for the Love wave velocity in m/s.
    vr : :obj:`tuple` (vr_min, vr_max, increment)
        Define the search space for the Rayleigh wave velocity in m/s.
    vs : :obj:`tuple` (vr_min, vr_max, increment)
        Define the search space for the S wave velocity in m/s. Only relevant for SH waves for P-, and SV-waves,
        the S-wave velocity is computed over the vp-/vs-ratio
    phi : :obj:`tuple` (phi_min, phi_max, increment)
        Define the search space for the Azimuth in degrees.
    theta : :obj:`tuple` (theta_min, theta_max, increment)
        Define the search space for the inclination angle in degrees.
    xi : :obj:`tuple` (xi_min, xi_max, increment)
        Define the search space for the Rayleigh wave ellipticity angle in degrees.
    """

    def __init__(
        self,
        wave_types: List[str] = ["R", "L", "P", "SV", "SH"],
        method: str = "ML",
        free_surface: bool = True,
        scaling_velocity: float = 1.0,
        use_ml_classification: bool = True,
        svm: SupportVectorMachine = None,
        eigenvector: int = 0,
        music_signal_space_dimension: int = 1,
        vp: Tuple[float, float, float] = (100.0, 2000.0, 100.0),
        vp_to_vs: Tuple[float, float, float] = (1.7, 2.2, 0.1),
        vs: Tuple[float, float, float] = (50.0, 1000.0, 100.0),
        vl: Tuple[float, float, float] = (100, 2000, 100),
        vr: Tuple[float, float, float] = (100, 2000, 100),
        phi: Tuple[float, float, float] = (0, 360, 1),
        theta: Tuple[float, float, float] = (0, 90, 1),
        xi: Tuple[float, float, float] = (-90, 90, 1),
    ) -> None:
        # Initial sanity checks
        methods = ["ML", "MUSIC", "MVDR", "BARTLETT", "DOT"]
        wtypes_implemented = ["P", "SV", "SH", "L", "R"]
        assert (
            method in methods
        ), f"Invalid option '{method}' selected for method! Must be one of [{methods}]!"
        for w_type in wave_types:
            assert w_type in wtypes_implemented, (
                f"Invalid wave type specified: {w_type}! Must be one of "
                f"[{wtypes_implemented}]"
            )
        if method == "ML" or use_ml_classification:
            if svm is None:
                raise ValueError(
                    "A SupportVectorMachine object needs to be provided for machine-learning based wave "
                    "type classification!"
                )
            model = svm.load_model()
            for w_type in wave_types:
                assert w_type in model.classes_, (
                    f"The provided SupportVectorMachine model was not trained for "
                    f"wave_type '{w_type}'. Please use a different SupportVectorMachine"
                    f"that was trained for this particular wave type."
                )
        self.wave_types = wave_types
        self.method = method
        self.use_ml_classification = use_ml_classification
        self.vs = vs
        self.vp = vp
        self.vp_to_vs = vp_to_vs
        self.vl = vl
        self.vr = vr
        self.phi = phi
        self.theta = theta
        self.xi = xi
        self.free_surface = free_surface
        self.svm = svm
        self.eigenvector = eigenvector
        self.music_signal_space_dimension = music_signal_space_dimension
        self.scaling_velocity = scaling_velocity
        self.vp_n = int((vp[1] + vp[2] - vp[0]) / vp[2])
        self.vs_n = int((vs[1] + vs[2] - vs[0]) / vs[2])
        self.vp_to_vs_n = int((vp_to_vs[1] + vp_to_vs[2] - vp_to_vs[0]) / vp_to_vs[2])
        self.vl_n = int((vl[1] + vl[2] - vl[0]) / vl[2])
        self.vr_n = int((vr[1] + vr[2] - vr[0]) / vr[2])
        self.phi_n = int((phi[1] + phi[2] - phi[0]) / phi[2])
        self.theta_n = int((theta[1] + theta[2] - theta[0]) / theta[2])
        self.xi_n = int((xi[1] + xi[2] - xi[0]) / xi[2])

    def compute_steering_vectors(self, wave_type: str) -> Callable[[], ndarray]:
        """Compute steering vectors for polarization analysis with grid search methods for the specified wave type.

        Parameters
        ----------
        wave_type : :obj:`str`
            Wave type for which steering vectors should be computed.

        Returns
        -------
        steering_vectors : :obj:`ndarray` (6, N)
            6-C steering vectors. N is the search-space dimension.
        """
        if wave_type == "P":
            vp = np.arange(self.vp[0], self.vp[1] + self.vp[2], self.vp[2])
            vp_to_vs = np.arange(
                self.vp_to_vs[0], self.vp_to_vs[1] + self.vp_to_vs[2], self.vp_to_vs[2]
            )
            if vp_to_vs[-1] > self.vp_to_vs[1]:
                vp_to_vs = vp_to_vs[:-1]
            theta = np.arange(
                self.theta[0], self.theta[1] + self.theta[2], self.theta[2]
            )
            phi = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])

            vp, vp_to_vs, theta, phi = np.meshgrid(
                vp, vp_to_vs, theta, phi, indexing="ij"
            )

            pm = PolarizationModel6C(
                wave_type=wave_type,
                vp=vp.ravel(),
                vs=vp.ravel() / vp_to_vs.ravel(),
                theta=theta.ravel(),
                phi=phi.ravel(),
                scaling_velocity=self.scaling_velocity,
                free_surface=self.free_surface,
            )
            return pm.polarization

        elif wave_type == "SV":
            vp = np.arange(self.vp[0], self.vp[1] + self.vp[2], self.vp[2])
            vp_to_vs = np.arange(
                self.vp_to_vs[0], self.vp_to_vs[1] + self.vp_to_vs[2], self.vp_to_vs[2]
            )
            if vp_to_vs[-1] > self.vp_to_vs[1]:
                vp_to_vs = vp_to_vs[:-1]
            theta = np.arange(
                self.theta[0], self.theta[1] + self.theta[2], self.theta[2]
            )
            phi = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])

            vp, vp_to_vs, theta, phi = np.meshgrid(
                vp, vp_to_vs, theta, phi, indexing="ij"
            )

            pm = PolarizationModel6C(
                wave_type=wave_type,
                vp=vp.ravel(),
                vs=vp.ravel() / vp_to_vs.ravel(),
                theta=theta.ravel(),
                phi=phi.ravel(),
                scaling_velocity=self.scaling_velocity,
                free_surface=self.free_surface,
            )
            return pm.polarization

        elif wave_type == "SH":
            vs = np.arange(self.vs[0], self.vs[1] + self.vs[2], self.vs[2])
            theta = np.arange(
                self.theta[0], self.theta[1] + self.theta[2], self.theta[2]
            )
            phi = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])

            vs, theta, phi = np.meshgrid(vs, theta, phi, indexing="ij")

            pm = PolarizationModel6C(
                wave_type=wave_type,
                vs=vs.ravel(),
                theta=theta.ravel(),
                phi=phi.ravel(),
                scaling_velocity=self.scaling_velocity,
                free_surface=self.free_surface,
            )
            return pm.polarization

        elif wave_type == "R":
            vr = np.arange(self.vr[0], self.vr[1] + self.vr[2], self.vr[2])
            phi = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])
            xi = np.arange(self.xi[0], self.xi[1] + self.xi[2], self.xi[2])

            vr, phi, xi = np.meshgrid(vr, phi, xi, indexing="ij")

            pm = PolarizationModel6C(
                wave_type=wave_type,
                vr=vr.ravel(),
                phi=phi.ravel(),
                xi=xi.ravel(),
                scaling_velocity=self.scaling_velocity,
                free_surface=self.free_surface,
            )
            return pm.polarization

        elif wave_type == "L":
            vl = np.arange(self.vl[0], self.vl[1] + self.vl[2], self.vl[2])
            phi = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])

            vl, phi = np.meshgrid(vl, phi, indexing="ij")

            pm = PolarizationModel6C(
                wave_type=wave_type,
                vl=vl.ravel(),
                phi=phi.ravel(),
                scaling_velocity=self.scaling_velocity,
                free_surface=self.free_surface,
            )
            return pm.polarization
