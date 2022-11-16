"""
TwistPy utility functions.
"""

import pickle
from typing import Any, Tuple

import numpy as np
import scipy
from numpy.lib.stride_tricks import as_strided


def load_analysis(file: str) -> Any:
    """Read a TwistPy analysis object from the disk.

    Parameters
    ----------
    file : :obj:`str`
        File name (include absolute path if the file is not in the current working directory)

    Returns
    -------
    :obj:`~twistpy.polarization.time.TimeDomainAnalysis3C` or
    :obj:`~twistpy.polarization.time.TimeDomainAnalysis6C` or
    :obj:`~twistpy.polarization.timefrequency.TimeFrequencyAnalysis3C` or
    :obj:`~twistpy.polarization.timefrequency.TimeFrequencyAnalysis6C` or
    :obj:`~twistpy.polarization.dispersion.DispersionAnalysis`
    """

    if file is None:
        raise Exception("Please specify the name of the file that you want to load!")
    fid = open(file, "rb")
    obj = pickle.load(fid)
    fid.close()

    return obj


def stransform(signal, dsfacf: int = 1, k: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute the discrete S-transform of the input signal.

    The S-transform or Stockwell transform provides a time-frequency representation of a signal. It is an extension of
    the continuous wavelet transform (CWT), obtained using a scalable localizing Gaussian window [1]. It is defined as:

    .. math::

        S(\tau,\ f)\ =\ \frac{\left|f\right|}{k\sqrt{2\pi}}
            \int_{-\infty}^{\infty}{u(t)exp\left(\frac{{-f}^2{(\tau-t)}^2}{2k^2}\right)}e^{-i2\pi ft}dt,

    where :math:`u` is the signal as a continuous function of time :math:`t`, :math:`\tau` is the translation of the
    Gaussian window, :math:`f` is frequency, and :math:`k` is a user-set constant specifying a scaling factor
    controlling the number of oscillations in the window. When :math:`k` increases, the frequency resolution increases,
    with a corresponding loss of time resolution [2].

    .. hint::
            [1] Stockwell, R. G., Mansinha, L., and Lowe, R. P. (1996). Localization of the complex spectrum:
            the S transform, IEEE Transactions on Signal Processing, 44(4), https://ieeexplore.ieee.org/document/492555

            [2] Simon, C., Ventosa, S., Schimmel, M., Heldring, A., Dañobeitia, J. J., Gallart, J., & Mànuel, A. (2007).
            The S-transform and its inverses: Side effects of discretizing and filtering.IEEE Transactions on Signal
            Processing, 55(10), 4928–4937. https://doi.org/10.1109/TSP.2007.897893

    Parameters
    ----------
    signal : :obj:`numpy.ndarray` of :obj:`float`
        Real signal
    dsfacf : :obj:`int`, default=1
        Down-sampling factor in the frequency direction -> enables efficient computation for long signals.

        .. warning::
                Downsampling of the frequency axis (dsfacf > 1) prevents the accurate computation of the inverse
                transform!

    k : :obj:`float`, default=1.
        Scaling factor that controls the number of oscillations in the window. When k increases, the frequency
        resolution increases, with a corresponding loss of time resolution [2].

    Returns
    -------
    stran : :obj:`numpy.ndarray` of :obj:`numpy.complex128`
        S-transform of the signal.
    f : :obj:`numpy.ndarray` of :obj:`float`
        Normalized frequency vector (divide by sampling interval dt to get frequency in Hz).
    """
    U = scipy.fft.fft(signal)
    N = np.max(signal.shape)
    odd = N % 2
    N_half = int(np.floor(N / 2))
    q = np.concatenate([np.arange(N_half + 1), np.arange(-N_half + 1 - odd, 0)])
    m = np.concatenate([np.arange(N_half + 1)])
    m.shape = (m.shape[0], 1)
    m = m[::dsfacf]
    m = m[1:]
    strides = U.strides[0]
    U_toeplitz = as_strided(
        np.concatenate([U, U[:-1]]),
        shape=(m.shape[0] + 1, N),
        strides=(strides * dsfacf, strides),
    )
    gaussian = np.exp(-2 * (np.pi * q * k / m) ** 2)
    U_toeplitz = U_toeplitz[1:, :]
    stran = scipy.fft.ifft(U_toeplitz * gaussian, axis=-1)
    stran_dc = np.ones((1, N)) * np.mean(signal)
    stran = np.concatenate([stran_dc, stran])
    f = np.concatenate([np.arange(N_half + 1)]) / N

    return stran, f[::dsfacf]


def istransform(
    st: np.ndarray, f: np.ndarray, k: float = 1.0, use_filter: bool = False
) -> np.ndarray:
    r"""Compute the inverse S-transform.

    This function computes the approximate inverse S-transform after Schimmel et al. (2005) [1]. This inverse has some
    advantageous properties over the conventional inverse S-transform as it reduces filter artifacts if the
    time-frequency spectrum is modified before the back-transform. Note that this inverse is only an approximation of
    the true inverse transform (even though a very good one), as described in [2].
    The inverse transform that is implemented here is defined as:

    .. math::

        \ u(t)=\ k\sqrt{2\pi}\int_{-\infty}^{\infty}\frac{S(\tau,\ f)}{\left|f\right|}e^{+i2\pi ft}df,

    where :math:`S(\tau, \ f)` is the time-frequency spectrum obtained using the forward S-transform, :math:`\tau` is
    the translation of the Gaussian window, :math:`f` is frequency, and :math:`k` is the user-set constant specifying
    the scaling factor controlling the number of oscillations in the window specified in the forward transform (see
    :func:`twistpy.utils.stransform` and the example below).

    The conventional inverse S-transform can simply be obtained as:

     .. math::

        \ u(t)=\ \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}{S(\tau,\ f)}e^{+i2\pi ft}d\tau df

    Or in Python code::

        u = scipy.fft.irfft(numpy.sum(st, axis=-1))


    .. hint:: [1] Schimmel, M., & Gallart, J. (2005). The inverse S-transform in filters with time-frequency
        localization. IEEE Transactions on Signal Processing, 53(11), 4417–4422. https://doi.org/10.1109/TSP.2005.857065

        [2] Simon, C., Ventosa, S., Schimmel, M., Heldring, A., Dañobeitia, J. J., Gallart, J., & Mànuel, A. (2007).
        The S-transform and its inverses: Side effects of discretizing and filtering.IEEE Transactions on Signal
        Processing, 55(10), 4928–4937. https://doi.org/10.1109/TSP.2007.897893

    Parameters
    ----------
    st : :obj:`numpy.ndarray` of :obj:`numpy.complex128`
        S-transformed signal.
    f : :obj:`numpy.ndarray` of :obj:`float`
        Normalized frequency vector.
    k : :obj:`float`, default = 1.0
        Scaling factor that controls the number of oscillations in the window.
    use_filter : :obj:`bool`, default = False
        Deconvolve the filter that describes the approximation
    Returns
    -------
    signal : :obj:`numpy.ndarray` of :obj:`float`
        Real-valued signal.
    """
    weight = np.tile(np.reshape(np.abs(f), (len(f), 1)), (1, st.shape[1]))
    weight[0, :] = 1
    st, f = st.copy(), f.copy()
    st /= np.abs(weight)
    st *= k * np.sqrt(2 * np.pi)
    u = np.diag(
        np.fft.irfft(st, axis=0)
    )  # Approximate inverse after Schimmel, which is a filtered version of the
    # exact inverse
    if use_filter:
        M = len(u)
        n_half = int(np.floor(M / 2))
        odd = M % 2
        f = np.concatenate([np.arange(n_half + 1), np.arange(-n_half + 1 - odd, 0)]) / M
        f.shape = (M, 1)
        n = np.arange(M)
        term1 = np.exp(-(1 / 2) * (np.tile(f, (1, M)) / 1 * np.tile(n, (M, 1))) ** 2)
        term2 = np.exp(2 * 1j * np.pi * (np.tile(f, (1, M)) * np.tile(n, (M, 1))))
        # Filter after Simon et al. 2007: The S-Transform and its Inverses: Side Effects of Discretizing and Filtering
        I_t = np.real(np.sum(term1 * term2, axis=0)) / M
        I_fft = np.fft.fft(I_t)
        u_fft = np.fft.fft(u)
        ist = np.real(np.fft.ifft(u_fft / I_fft))
    else:
        ist = u
    return ist
