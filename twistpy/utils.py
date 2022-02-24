"""
TwistPy utility functions
"""

import pickle
from typing import Union

import numpy as np

from twistpy.dispersion import DispersionAnalysis
from twistpy.time import TimeDomainAnalysis
from twistpy.timefrequency import TimeFrequencyAnalysis


def load_analysis(file: str = None) -> Union[TimeDomainAnalysis, TimeFrequencyAnalysis, DispersionAnalysis]:
    """Read a TwistPy analysis object from the disk.

    Parameters
    ----------
    file : :obj:`str`
        File name (include absolute path if the file is not in the current working directory)

    Returns
    -------
    obj : :obj:`~twistpy.TimeDomainAnalysis` or :obj:`~twistpy.TimeFrequencyAnalysis` or :obj:`~twistpy.DispersionAnalysis`
    """

    if file is None:
        raise Exception('Please specify the name of the file that you want to load!')
    fid = open(file, 'rb')
    obj = pickle.load(fid)
    fid.close()
    return obj


def i_s_transform(st: np.ndarray, f: np.ndarray):
    """ Compute the inverse S-transform.

    """
    weight = np.tile(np.reshape(np.abs(f), (len(f), 1)), (1, st.shape[1]))
    weight[0, :] = 1
    st /= np.abs(weight)
    st *= np.sqrt(2 * np.pi)
    u = np.diag(np.fft.irfft(st, axis=0))  # Approximate inverse after Schimmel,
    # filtered version of the exact inverse
    M = len(u)
    n_half = int(np.floor(M / 2))
    odd = M % 2
    f = np.concatenate([np.arange(n_half + 1), np.arange(-n_half + 1 - odd, 0)]) / M
    f.shape = (M, 1)
    n = np.arange(M)
    term1 = np.exp(-(1 / 2) * (np.tile(f, (1, M)) / 1 * np.tile(n, (M, 1))) ** 2)
    term2 = np.exp(2 * 1j * np.pi * (np.tile(f, (1, M)) * np.tile(n, (M, 1))))
    # Filter after Simon et al. 2007: The S-Transform and its Inverses: Side Effects of Discretizing and Filtering
    I = np.real(np.sum(term1 * term2, axis=0)) / M
    I_fft = np.fft.fft(I)
    u_fft = np.fft.fft(u)

    return np.real(np.fft.ifft(u_fft / I_fft))


def s_transform(signal, dsfacf=1, k=1):
    """Compute the S-transform of the input signal.

    Returns:
        stran: S-transform of the signal with dimensions (next_pow_2(len(signal))/2+1, next_pow_2(len(signal)))
        f: Normalized frequency vector (divide by sampling interval dt to get frequency in Hz)
        dsfacf: Down-sampling factor in the frequency direction -> enables efficient computation for long signals
    """
    M = np.max(signal.shape)
    M_half = int(np.floor(M / 2))
    f = np.concatenate([np.arange(M_half + 1)]) / M
    stran = np.zeros((M_half + 1, M), dtype='complex')
    signal.shape = (1, M)
    m = np.arange(M_half + 1)
    m.shape = (M_half + 1, 1)
    stran = stran[::dsfacf, :]
    m = m[::dsfacf]
    m = np.tile(m, (1, M))
    for p in np.arange(M):
        stran[:, p] = _evaluate_stran(signal, p, m, k, M)
    stran[0, :] = np.mean(signal * np.ones((1, M)))
    return stran, f[::dsfacf]


def _evaluate_stran(signal, p, m, k, M):
    S = np.sum(signal * (np.abs(m) / (k * M * np.sqrt(2 * np.pi)))
               * np.exp(-0.5 * (m * (p - np.arange(M)) / (M * k)) ** 2)
               * np.exp(-2 * 1j * np.pi * np.arange(M) * m / M), axis=1)
    return S


def _s_transform_deprecated(signal, dsfacf=1):
    """
    Computes the S-transform of the input signal
    David Sollberger, 2020

    Returns:
        signal_strans: S-transform of the signal with dimensions (next_pow_2(len(signal))/2+1, next_pow_2(len(signal)))
        f: Normalized frequency vector (divide by sampling interval dt to get frequency in Hz)
        dsfacf: Down-sampling factor in the frequency direction -> enables efficient computation for long signals

    Code is adapted from a Matlab implementation by Vincent Perron, ETHZ, Switzerland,  which is based on the
    Matlab implementation by  Kalyan S. Dash, IIT Bhubaneswar, India

    """
    n = signal.shape[0]
    n_half = int(np.floor(n / 2))
    odd = n % 2

    f = np.concatenate([np.arange(n_half + 1), np.arange(-n_half + 1 - odd, 0)]) / n
    signal_fft = np.fft.fft(signal, n=n)
    signal_fft.shape = (1, n)
    periods = 1. / f[dsfacf:int(n_half) + 1:dsfacf]
    periods.shape = (periods.shape[0], 1)
    w = 2 * np.pi * np.tile(f, (periods.shape[0], 1)) * np.tile(periods, (1, n))
    # gaussian = (np.abs(2*np.pi*np.tile(1/periods, (1, w.shape[1])))/np.sqrt(2*np.pi))*np.exp((- w ** 2) / 2)
    gaussian = np.exp((- w ** 2) / 2)
    hw = _toeplitz_red(signal_fft[:, 1:n_half + 1].T, signal_fft.T, dsfacf)
    signal_strans = np.fft.ifft(hw * gaussian, n=n)
    signal_strans_zero = np.mean(signal) * np.ones(n)
    signal_strans_zero.shape = (1, signal_strans_zero.shape[0])
    signal_strans = np.concatenate([signal_strans_zero, signal_strans])

    return signal_strans.conj(), np.insert(f[dsfacf:n_half + 1:dsfacf], 0, 0)


def _toeplitz_red(c, r, dsfacf):
    """
    Constructs a non-symmetric Toeplitz matrix with c as its first column and r its first row
    :param c: first column
    :param r: first row
    :param dsfacf: down-sampling factor
    :return: T (Toeplitz matrix)
    David Sollberger, 2020 (david.sollberger@gmail.com)
    """

    v = np.concatenate((np.flip(r[1:]), c.conj()), axis=0)
    index = np.arange(r.shape[0] - 1, -1, -1) - 1
    index.shape = (index.shape[0], 1)
    ind = np.arange(dsfacf, c.shape[0] + 1, dsfacf)
    ind.shape = (1, ind.shape[0])

    index = np.add(index, ind).squeeze()
    T = v[index].squeeze().T
    return T
