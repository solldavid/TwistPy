from typing import Union

import numpy as np
from matplotlib import axes
from obspy.core import Stream, UTCDateTime
from scipy.fft import rfft
from scipy.linalg import pinvh
from spectrum import dpss


class BeamformingArray:
    r"""Beamforming using the multi-taper technique.

    Specify a BeamformingArray object that can be used for beamforming using the multi-taper technique [1],[2].

    [1] Meng et al. (2011). A window into the complexity of the dynamic rupture of the 2011 Mw 9 Tohoku-Oki earthquake.
    GRL:  https://doi.org/10.1029/2011GL048118
    [2] Meng et al. (2012). High-resoultion backprojection at regional distance. JGR:
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JB008702

    Parameters
    ----------
    name : :obj:`str`
        Array name
    coordinates : :obj:`~numpy.ndarray`
        Coordinates of each receiver. Array of dimensions (Nx3), with N being the number of receivers
    reference_receiver : :obj:`int`
        Index of receiver to use as a reference station (time delays are computed relative to
        this receiver). Defaults to 0 (the first receiver that is passed)

    Attributes
    ----------
    N : :obj:`int`
        Number of stations in the array
    data : :obj:`~obspy.core.stream.Stream`
        Seismic data in ObsPy format. len(data) = N
    steering_vectors : :obj:`~numpy.ndarray`
        Steering vectors used for beam forming

    """

    def __init__(self,
                 name: str = '',
                 coordinates: np.ndarray = None, reference_receiver: int = 0):
        """
        Instantiates a BeamformingArray() object

        """
        self.name = name
        if coordinates is None:
            raise Exception('Array coordinates need to be specified when instantiating an object of this class!')
        self.coordinates = coordinates
        self.reference_receiver = reference_receiver
        self.N: int = coordinates.shape[0]
        self.data: Stream = Stream()
        self.has_data: bool = False
        self.steering_vectors: np.ndarray = None
        self.has_steering_vectors: bool = False
        self.n_inc: int = None
        self.n_azi: int = None
        self.n_vel: int = None

    def beamforming(self, method: str = 'MUSIC', event_time: UTCDateTime = None, frequency_band: tuple = (90., 110.),
                    window: int = 5, number_of_sources: int = 1) -> np.ndarray:
        r"""Compute beam power at specified time window location for this instance of class BeamformingArray.

        Parameters
        ----------
        method : :obj:`str`, default='MUSIC'
             Beamforming method to use.

             .. hint:: 'BARTLETT': Conventional beamforming.

                'MVDR': Minimum Variance Distortionless Response beamformer or Capon beamformer. In seismology,
                often referred to as FK method.

                'MUSIC': Multiple Signal Classification.
        event_time : :obj:`~obspy.core.datetime.UTCDateTime`
            Start time of the analysis window
        frequency_band : :obj:`tuple`
            Frequency band over which beamforming is performed, specified as (minimum_frequency, maximum_frequency).
            (Covariance matrices are averaged within this frequency band)
        window : :obj:`float`
            Length of the time window defined as the number of dominant periods included in the window,
            where the dominant period is defined as 1/mean(minimum_frequency, maximum_frequency))
        number_of_sources : :obj:`int`
             Number of sources that are estimated  (only relevant for MUSIC, defaults to 1)

        Returns
        -------
        P : :obj:`~numpy.ndarray`
            Beampower as an array of shape (N_incilination_grid, N_azimuth_grid, N_velocity_grid)
        """
        if not self.has_data:
            raise Exception('There is no data attached to this Array yet', self.name)
        if not self.has_steering_vectors:
            raise Exception('Steering vectors need to be precomputed for this Array!', self.name)
        window_length_samples = int(window * self.data[0].stats.sampling_rate / np.mean(frequency_band))
        data_windowed = self.data.slice(starttime=event_time,
                                        endtime=event_time + (window_length_samples - 1) * self.data[0].stats.delta)
        data = np.asarray([tr.data for tr in data_windowed])

        # Rule of thumb is used here to compute the number of taper windows
        Nw = max(1, int(2 * (window_length_samples / data_windowed[0].stats.sampling_rate) * (0.2 * frequency_band[1])))
        C = cmmt(data, Nw, freq_band=frequency_band, fsamp=self.data[0].stats.sampling_rate)
        P = self._beamforming(C, method=method, number_of_sources=number_of_sources)
        return np.real(P)

    def add_data(self, data: Stream) -> None:
        """Add data in ObsPy stream format to this instance of class BeamformingArray.

        Parameters
        ----------
        data : :obj:`~obspy.core.stream.Stream`
            ObsPy stream of len(N) containing the seismic data for each receiver in the array
        """
        assert isinstance(data, Stream), 'Data must be in Stream format!'
        assert len(
            data) == self.N, f'Number of traces in data ({len(Stream)}) does not agree with number of statioon in ' \
                             f'array ({self.N}) '

        self.data = data
        self.has_data = True
        print('Data successfully added to the BeamformingArray object!')

    def compute_steering_vectors(self, frequency: float, intra_array_velocity: Union[float, tuple] = 6000,
                                 inclination: tuple = (-90, 90, 1), azimuth: tuple = (0, 360, 1)) -> None:
        r"""Precompute the steering vectors

        Compute the steering vectors for the specified parameter range. For parameters that are specified as a tuple,
        the grid search is performed over the range: (min_value, max_value, increment)

        Parameters
        ----------
            frequency : :obj:`float`
                Discrete frequency at which beamforming is performed
            intra_array_velocity : :obj:`float` or :obj:`tuple`
                Specifies the velocity as a float (if known) or grid over which search is performed
            inclination : :obj:`tuple`
                Specifies inclination grid over which search is performed
            azimuth : :obj:`tuple`
                Specifies azimuth grid over which search is performed

        """
        if isinstance(intra_array_velocity, tuple):
            velocity_gridded = np.arange(intra_array_velocity[0], intra_array_velocity[1] + intra_array_velocity[2],
                                         intra_array_velocity[2])
            self.n_vel = len(velocity_gridded)
        else:
            velocity_gridded = intra_array_velocity
            self.n_vel = 1
        inclination_gridded = np.radians(np.arange(inclination[0], inclination[1] + inclination[2], inclination[2]))
        azimuth_gridded = np.radians(np.arange(azimuth[0], azimuth[1] + azimuth[2], azimuth[2]))
        self.n_inc, self.n_azi = len(inclination_gridded), len(azimuth_gridded)
        azimuth_gridded, inclination_gridded, velocity_gridded = np.meshgrid(azimuth_gridded, inclination_gridded,
                                                                             velocity_gridded)
        coordinates = self.coordinates - np.tile(self.coordinates[self.reference_receiver, :], (self.N, 1))
        wave_vector_x = (np.sin(inclination_gridded) * np.cos(azimuth_gridded)).ravel()
        wave_vector_y = (np.sin(inclination_gridded) * np.sin(azimuth_gridded)).ravel()
        wave_vector_z = (np.cos(inclination_gridded)).ravel()
        wave_vector_x, wave_vector_y, wave_vector_z = np.asmatrix(wave_vector_x).T, np.asmatrix(
            wave_vector_y).T, np.asmatrix(wave_vector_z).T
        wave_number = (-2 * np.pi * frequency / velocity_gridded).ravel()
        wave_number = np.asmatrix(wave_number).T
        coordinates = np.asmatrix(coordinates)
        steering_vectors: np.ndarray = np.exp(1j * np.multiply(np.tile(wave_number, (1, self.N)),
                                                               (wave_vector_x * coordinates[:, 0].T
                                                                + wave_vector_y * coordinates[:, 1].T
                                                                + wave_vector_z * coordinates[:, 2].T)))
        self.steering_vectors = steering_vectors / np.sqrt(self.N)  # Ensure that steering vectors are unit vectors
        self.has_steering_vectors = True
        print('Steering vectors computed!')

    def _beamforming(self, C: np.ndarray, method: str = 'MUSIC', number_of_sources: int = 1) -> np.ndarray:
        """Compute beam power.

        Parameters
        ----------
         C : :obj:`~numpy.ndarray` of :obj:`numpy.complex128`
             Covariance matrix
         method : :obj:`str`
             Beamforming method to use.

             ..hint:: 'BARTLETT': Conventional beamforming.

                'MVDR': Minimum Variance Distortionless Response beamformer or
                 Capon beamformer. In seismology, often referred to as FK method.

                'MUSIC': Multiple Signal Classification.
         number_of_sources : :obj:`int`
             Number of sources that are estimated  (only relevant for MUSIC, defaults to 1)

        Returns
        -------
        P : :obj:`~numpy.ndarray`
            Beampower as an array of shape (N_incilination_grid, N_azimuth_grid, N_velocity_grid)
        """
        if method == 'MUSIC':
            evalues, evectors = np.linalg.eigh(C)
            noise_space: np.ndarray = (evectors[:, :self.N - number_of_sources]).dot(
                np.matrix.getH(evectors[:, :self.N - number_of_sources]))
            P: np.ndarray = 1 / np.einsum("sn, nk, sk->s", self.steering_vectors.conj(), noise_space,
                                          self.steering_vectors, optimize=True)
        elif method == 'MVDR':
            P: np.ndarray = 1 / np.einsum("sn, nk, sk->s", self.steering_vectors.conj(), pinvh(C, rcond=1e-6),
                                          self.steering_vectors, optimize=True)
        elif method == 'BARTLETT':
            P: np.ndarray = np.einsum("sn, nk, sk->s", self.steering_vectors.conj(), C, self.steering_vectors,
                                      optimize=True)
        else:
            raise Exception(
                f"Unknown beam-forming method: '{method}'!. Available methods are: 'MUSIC', 'BARTLETT', 'MVDR'!")
        return np.reshape(P, (self.n_inc, self.n_azi, self.n_vel))


class GradiometryArray:
    """Compute rotation and strain from small-aperture array of three-component receivers.

    """
    pass


def cmmt(data: np.ndarray, Nw: int, freq_band: tuple, fsamp: float) -> np.ndarray:
    """Compute array data covariance matrix using the multitaper method

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        Data as an array of dimensions (NxNt) with N being the number of stations in the array and Nt the number of time
        samples.
    Nw : :obj:`int`
        Number of tapers to use for the multi-taper method
    freq_band : :obj:`tuple`
         Frequency band within which covariance matrices are averaged as (fmin, fmax)
    fsamp : :obj:`float`
        Sampling rate of the data in Hz

    Returns
    -------
    C : :obj:`~numpy.ndarray` of :obj:`~numpy.complex128`
        Covariance matrix averaged within specified frequency band.

    """
    # Number of stations (m), time sampling points (Nx)
    N, Nt = data.shape
    # Next power of 2 (for FFT)
    NFFT = 2 ** int(np.log2(Nt) + 1) + 1

    # Demean data
    data = (data.T - np.mean(data, axis=1)).T

    # Compute slepian sequences and eigenvalues
    tapers, eigenvalues = dpss(N=Nt, NW=Nw)
    tapers = np.tile(tapers.T, [N, 1, 1])
    tapers = np.swapaxes(tapers, 0, 1)

    # Compute weights from eigenvalues
    weights = eigenvalues / (np.arange(int(2 * Nw)) + 1).astype(float)

    # Mutitaper spectral estimation
    S = rfft(np.multiply(tapers, data), 2 * NFFT, axis=-1)

    # Inverse of weighted Power spectrum for scaling
    Sk_inv = 1 / np.sqrt(np.sum((np.abs(S) ** 2).T * weights, axis=-1).T)

    # Only compute covariance matrices within the specified frequency band
    df = fsamp / (2 * NFFT)
    f = np.arange(0, NFFT + 1) * df
    ind = (f >= freq_band[0]) & (f < freq_band[1])
    S = S[:, :, ind]
    Sk_inv = Sk_inv[:, ind]

    S = np.moveaxis(S, 1, 2)
    Sk_inv = np.moveaxis(Sk_inv, 0, 1)
    scales = np.einsum('...i,...j->...ij', Sk_inv, Sk_inv, optimize=True)

    # Compute covariance matrix
    C = scales * (np.einsum('...i,...j->...ij', S, S.conj(), optimize=True).astype('complex') *
                  np.tile(np.moveaxis(weights[np.newaxis, np.newaxis, np.newaxis], 3, 0), (1, S.shape[1], N, N)))
    # Sum over tapers
    C = np.sum(C, axis=0)

    # Average over frequency range
    C = np.nanmean(C, axis=0)

    return C


def plot_beam(grid: np.ndarray, beam_origin: np.ndarray, P: np.ndarray, inclination: tuple, azimuth: tuple,
              ax: axes, clip: float = 0.2) -> None:
    """Helper function to plot the beam power in 3D assuming a homogeneous velocity model.

    Args:
        grid : :obj:`~numpy.ndarray`
        x, y, and z coordinates where the beam should be plotted in an array of dimension Nx3 (N being the number of points)
        beam_origin: x, y, and z coordinate of the origin of the beam
        P : :obj:`numpy.ndarray` beam power as a function of inclination and azimuth (2D array or 3D array,
            depending on whether the velocity was included in the search)
        inclination : :obj:`tuple`
            Tuple with the inclination search parameters used to compute P
        azimuth : :obj:`tuple`
            Tuple with the azimuth search parameters used to compute P
        clip : :obj:`float`
            Percentage of beam power to be ignored for plotting, everything that has a power P < clip*P.max().max()
            will be ignored. (Defaults to 0.2).
        ax : :obj:`matplotlib.axes`
            Matplotlib axes object where the beam is plotted (defaults to the current axis)

    """
    dxdydz = beam_origin - grid
    azimuth_grid = np.degrees(np.arctan2(dxdydz[:, 1], dxdydz[:, 0]))
    azimuth_grid = np.around(azimuth_grid / azimuth[2], decimals=0) * azimuth[2]
    inclination_grid = np.degrees(np.arctan(np.linalg.norm(dxdydz[:, :-1], axis=1) / dxdydz[:, 2]))
    inclination_grid = np.around(inclination_grid / inclination[2], decimals=0) * inclination[2]
    inclination_grid_index = ((inclination_grid - inclination[0]) / inclination[2]).astype(int)
    azimuth_grid_index = ((azimuth_grid - azimuth[0]) / azimuth[2]).astype(int)
    inclination_grid_index_use = (inclination_grid_index < P.shape[0]) * (inclination_grid_index > 0)
    azimuth_grid_index_use = (azimuth_grid_index < P.shape[0]) * (azimuth_grid_index > 0)
    use = inclination_grid_index_use * azimuth_grid_index_use
    beam_power_grid = P[inclination_grid_index[use], azimuth_grid_index[use]].ravel()
    grid = grid[use]
    grid = grid[beam_power_grid > clip * P.max().max(), :]
    beam_power_grid = beam_power_grid[beam_power_grid > clip * P.max().max()]
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], marker='.', c=beam_power_grid, vmin=clip * P.max().max(),
               vmax=P.max().max(), cmap='inferno')
