import math
from typing import Tuple

import numpy as np


def _next_pow2(n):
    return 2 ** int(math.ceil(math.log(n) / math.log(2.0)))


def _nearest_pow2(x: float) -> float:
    r"""Get nearest power of 2 for a given input x.

    Parameters
    ----------
    x : :obj:`float`
        Arbitrary number

    Returns
    -------
    y : :obj:`float`
        Nearest power of 2 from x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b


def transfer_function(
    response: np.ndarray, source: np.ndarray, dt: float, smooth: float
) -> tuple:
    r"""
    Calculate transfer function and complex coherence between two signals.
    The transfer function is calculated from smoothed cross- and non-smoothed
    autospectral densities of source and response signal. Smoothing is done by
    convolution with a Blackman window.
    The complex transfer function, the autospectral densities, and a
    corresponding frequency vector are returned.

    Parameters
    ----------
    response : :obj:`numpy.ndarray`
        Sample data of the response signal.
    source : :obj:`numpy.ndarray`
        Sample data of the source signal.
    dt : :obj:`float`
        Sampling interval [s].
    smooth : :obj:`float`
        Size of the Blackman window used for smoothing [Hz].

    Returns
    -------
    output : 5-:obj:`tuple` of :obj:`numpy.ndarray`
        (``freq``, ``XX``, ``YY``, ``Ars``, ``coh``)
        ``freq``: array of frequencies
        ``Grr``: autospectral density of response signal,
        ``Gss``: autospectral density of source signal,
        ``Ars``: source to response transfer function,
        ``coh``: smoothed complex coherence between source and response signal
    """

    assert response.size == source.size
    ndat = response.size

    nfft = int(_nearest_pow2(ndat))
    nfft *= 2
    gr = np.zeros(nfft)
    gs = np.zeros(nfft)
    gr[:ndat] = response
    gs[:ndat] = source

    # perform ffts
    Gr = np.fft.rfft(gr) * dt
    Gs = np.fft.rfft(gs) * dt
    freq = np.fft.rfftfreq(nfft, dt)

    # calculate autospectral and crossspectral densities
    Grs = Gr * Gs.conjugate()
    Grr = Gr * Gr.conjugate()
    Gss = Gs * Gs.conjugate()

    nsmooth = int(round(smooth / (freq[1] - freq[0])))
    if nsmooth != 0:
        w = np.blackman(nsmooth)
        Grs_smooth = np.convolve(Grs, w, mode="same")
        Grr_smooth = np.convolve(Grr, w, mode="same")
        Gss_smooth = np.convolve(Gss, w, mode="same")
    else:
        Grs_smooth = Grs
        Grr_smooth = Grr
        Gss_smooth = Gss

    Crs_smooth = Grs_smooth / np.sqrt(Grr_smooth * Gss_smooth)

    # calculate transfer function
    Ars = Crs_smooth * np.sqrt(Grr / Gss)

    return freq, Grr, Gss, Ars, Crs_smooth


def remove_tilt(
    response: np.ndarray,
    source: np.ndarray,
    dt: float,
    fmin: float = None,
    fmax: float = None,
    parallel: bool = True,
    threshold: float = 0.5,
    smooth: float = 1.0,
    g: float = 9.81,
    method: str = "coh",
    trans_coh: Tuple[np.ndarray, np.ndarray] = None,
) -> np.ndarray:

    r"""
    Remove tilt noise from translational accelerometer recordings.
    See the ``method`` argument for different correction options.
    The correction can optionally be applied only in a selected frequency band.
    The method is described in Bernauer et al. (2022) and is based on the work of
    Crawford and Webb (2000).

    Parameters
    ----------
    response : :obj:`numpy.ndarray`
        Data samples of the accelerometer signal [m/s**2].
    source : :obj:`numpy.ndarray`
        Data samples of the tilt signal [rad].
    dt : :obj:`float`
        Sampling interval [s].
    fmin : :obj:`float`, optional
        Minimum frequency for band-limited correction [Hz]. Only applicable in
        ``'coh'`` and ``'freq'`` methods.
    fmax : :obj:`float`, optional
        Maximum frequency for band-limited correction [Hz]. Only applicable in
        ``'coh'`` and ``'freq'`` methods.
    parallel : :obj:`bool`, optional, Default=True
        Flag to indicate if tilt and acceleration axes are parallel (``True``)
        or antiparallel (``False``).
    threshold : :obj:`float`, optional, Default=0.5
        Correction is applied only where ``abs(coherence) >= threshold``. Only
        applicable in ``'coh'`` method.
    smooth : :obj:`float`, optional, Defaault=1.0
        Size of the Blackman window [Hz] used for smoothing when calculating
        the coherence with :py:func:`tilt_utils.transfer_function`. Only
        applicable in ``'coh'`` and ``'freq'`` methods.
    g : :obj:`float`, optional, Default=9.81
        Gravitational acceleration [m/s**2].
    method : :obj:`str`, optional, Default='coh'
        Correction method to use.

        ..hint:: 'coh':  apply theoretical transfer
                 function where coherence is significant (via frequency domain),

            'freq': Use empirical transfer function estimate (via frequency
            domain)

            'direct': Apply theoretical transfer function directly (in time domain).

    trans_coh : (:obj:`numpy.ndarray`, :obj:`numpy.ndaarray`), optional
        If given, previously calculated transfer function and complex coherence
        between the tilt and accelerometer signals, used to decide where
        to apply the correction. The size of the given arrays must match the
        size of the spectra of ``source`` and ``response`` (the same
        zero-padding has to be applied). If set to ``None``, it is computed
        from ``response`` and ``source`` using
        :py:func:`twistpy.tilt.correction.transfer_function`.

    Returns
    -------
    signal_out : :obj:`numpy.ndarray`
        Data samples of corrected accelerometer signal [m/s**2].
    """

    assert response.size == source.size
    assert method in ("direct", "coh", "freq")

    sign = 1.0 if parallel else -1.0

    if method == "direct":
        return response - sign * g * np.sin(source)

    ndat = response.size

    nfft = int(_nearest_pow2(ndat))
    nfft *= 2

    if trans_coh is None:
        Ars, coh = transfer_function(response, source, dt, smooth)[-2:]
    else:
        Ars, coh = trans_coh

    Gr = np.fft.rfft(response, nfft)
    Gs = np.fft.rfft(source, nfft)
    freq = np.fft.rfftfreq(nfft, dt)

    assert Ars.shape == Gr.shape
    assert coh.shape == Gr.shape

    mask = np.where(np.abs(coh) >= threshold, 1.0, 0.0)
    if fmin is not None:
        mask[freq < fmin] = 0.0

    if fmax is not None:
        mask[freq > fmax] = 0.0

    if method == "coh":
        corr = sign * g * Gs * mask

    elif method == "freq":
        corr = sign * np.conjugate(Ars) * Gs

    else:
        raise ValueError("Invalid `method` argument: %s" % method)

    return np.fft.irfft(Gr - corr)[:ndat]
