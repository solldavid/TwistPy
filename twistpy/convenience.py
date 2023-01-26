"""TwistPy convenience functions

The functions provided herein are merely meant for illustration purposes and help to keep the Tutorials more readable
by limiting the ammount of code that needs to be included.
"""

import numpy as np
from obspy.core import UTCDateTime, Stream, Trace
from pathlib import Path
from scipy import fft

from twistpy.array_processing import BeamformingArray


def generate_synthetics(
    source_coordinates: np.ndarray,
    array: BeamformingArray,
    t: np.ndarray,
    velocity: float = 6000,
    center_frequency: float = 10,
):
    """Generates synthetics for a given BeamformingArray object assuming a homogeneous velocity model.

    Parameters
    ----------
    source_coordinates : :obj:`~numpy.ndarray`
        Source coordinates
    array : :obj:`~twistpy.array.BeamformingArray`
        Array for which to compute synthetics
    t : :obj:`numpy.ndarray`
        Time vector for the seismogram
    velocity : :obj:`float`
        Medium velocity in m/s
    center_frequency : :obj:`float`
        Center frequency of Ricker wavelet

    Returns
    -------
    data : :obj:`numpy.ndarray`
        Synthetic data as an array of shape (N, Nt), with N being the number of stations in the array and Nt the number
        of time samples
    """
    tt = (
        np.linalg.norm(
            array.coordinates - np.tile(source_coordinates, (array.N, 1)), axis=1
        )
    ) / velocity
    dt = t[1] - t[0]
    wavelet, t_wavelet, wcenter = ricker(t, f0=center_frequency)
    data = fft_roll(wavelet, tt, dt)
    data = data[:, wcenter:]
    if len(t) % 2 == 0:
        data_pad = np.zeros((array.N, len(t)))
        data_pad[:, 0 : data.shape[1]] = data
        data = data_pad
    return np.asarray(data, dtype="float")


def ricker(t, f0=10):
    r"""Ricker wavelet

    Create a Ricker wavelet given time axis ``t`` and central frequency ``f_0``

    _seealso:: This function is taken from Pylops: https://pylops.readthedocs.io/en/stable/

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    f0 : :obj:`float`, optional
        Central frequency

    Returns
    -------
    w : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    wcenter : :obj:`int`
        Index of center of wavelet

    """
    if len(t) % 2 == 0:
        t = t[:-1]
    w = (1 - 2 * (np.pi * f0 * t) ** 2) * np.exp(-((np.pi * f0 * t) ** 2))

    w = np.concatenate((np.flipud(w[1:]), w), axis=0)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    wcenter = np.argmax(np.abs(w))

    return w, t, wcenter


def fft_roll(signal: np.ndarray, dt: np.ndarray, samp: float) -> np.ndarray:
    """Helper routine to shift a signal according to moveouts `dt`.

    Parameters
    ----------
    signal : :obj:`numpy.ndarray`
        Signal to be shifted
    dt : :obj:`numpy.ndarray`
        Time shifts
    samp : :obj:`float`
        Sampling interval of input signal (in s)

    Returns
    -------
    signal_shifted : :obj:`numpy.ndarray`
        Shifted signals
    """

    Nrec = len(dt)
    Nt = len(signal)

    freqs = fft.rfftfreq(Nt, d=samp)
    F = fft.rfft(signal)

    R = np.zeros((Nrec, len(F)), dtype=np.complex128)

    for i in range(Nrec):
        R[i] = F * np.exp(-2j * np.pi * freqs * dt[i])

    signal_shifted = fft.irfft(R, axis=1)

    return signal_shifted


def to_obspy_stream(data: np.ndarray, starttime: UTCDateTime, dt: float) -> Stream:
    """Convert synthetic, dummy data from numpy array to Obspy Stream object.

    This function is only used for illustration purposes in the tutorials.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data as a numpy array
    starttime : :obj:`~obspy.core.datetime.UTCDateTime`
        Time of the first sample
    dt : :obj:`float`
        Sampling interval in seconds

    Returns
    -------
    st : :obj:`~obspy.core.stream.Stream`
        Data as an Obspy Stream object
    """
    nrec = data.shape[0]
    npts = data.shape[1]
    st = Stream()
    for n in range(nrec):
        header = {
            "delta": dt,
            "npts": int(npts),
            "sampling_rate": float(1 / dt),
            "starttime": starttime,
            "station": f"X{n:02d}",
            "network": "XX",
            "channel": "XXX",
        }  # Assign dummy Id
        tr = Trace(data=data[n, :], header=header)
        st += tr
    return st


def get_project_root() -> Path:
    return Path(__file__).parent.parent
