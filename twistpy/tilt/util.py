#!/usr/bin/env python

import sys
from typing import Tuple

import numpy as np
from obspy import read_inventory
from obspy.core import UTCDateTime, read, Trace, Stream
from obspy.signal.trigger import classic_sta_lta, plot_trigger


def get_data(
    stream1: str,
    stream2: str,
    utctime: str,
    duration: float,
    seis_channel: str,
    rot_channel: str,
    inventory: str,
    ch_r: str,
    ch_s: str,
) -> Tuple[Stream, Stream]:
    r"""
    Read in data from two files and do basic pre-processing.
    1. sort the channels
    2. remove the response
    3. rotate to zne-system if required
    4. cut out required time span
    5. select source and reciever channles

    Parameters
    ----------
    stream1 : :obj:`str`
        Full path to data recorded on 'seis_channel'
    stream2 : :obj:`str`
        Full path to data recorded on 'rot_channel'
    utctime : :obj:`str`
        Start time of record to be analysed,
        format: YYYY-MM-DDThh:mm:ss
    duration : :obj:`float`
        Length of time span to by analysed in sec
    seis_channel : :obj:`str`
        Channel(s) containing seismometer recordings
    rot_channel : :obj:`str`
        Channel(s) containing rotation rate recordings
    inventory : :obj:`str`
        Path to *.xml file containing response information
    ch_r : :obj:`str`
        Receiver channel (data to be corrected)
    ch_s : :obj:`str`
        Source channel (data to correct for)

    Returns
    -------
    r : :obj:`~obspy.core.Stream`
        Receiver channel
    s : :obj:`~obspy.core.Stream`
        Source channel
    """
    # define some parameters
    # take some seconds before and after the series of steps
    p = 0.1
    dt = p * duration

    # read the inventory for meta data
    inv = read_inventory(inventory)

    # get the strat time right
    t = UTCDateTime(utctime)

    # define the seismometer and rotation sensor input channels
    chan1 = seis_channel  # TODO: not used and probably not needed?  # noqa
    chan2 = rot_channel  # noqa

    # -------------------------------------------------------------------------
    # process the classic seismometer records
    # 1. read in the records and sort the channels, detrend and taper
    sz1 = read(stream1, starttime=t - dt, endtime=t + duration + dt)
    sz1.sort()
    sz1.reverse()
    sz1.detrend("linear")
    sz1.detrend("demean")
    sz1.taper(0.1)

    # 2. remove response and out put velocity in m/s
    sz1.attach_response(inv)
    sz1.remove_response(water_level=60, output="VEL")

    # 3. rotate the components according to the orientation as documented in
    # the inventory
    sz1.rotate(method="->ZNE", inventory=inv, components=["ZNE"])

    # asign samplingrate and number of samples for seismometer channels
    df1 = sz1[0].stats.sampling_rate
    npts1 = sz1[0].stats.npts

    # -------------------------------------------------------------------------
    # process the rotation rate records
    # 1. read in the records and sort the channels, detrend and taper
    sz2 = read(stream2, starttime=t - dt, endtime=t + duration + dt)
    sz2.sort()
    sz2.detrend("demean")
    sz2.taper(0.1)

    # 2. remove response (scale by sensitivity) to out put rotation rate in
    # rad/s
    sz2.attach_response(inv)
    sz2.remove_sensitivity()
    # 3. rotate the components according to the orientation as documented in
    # the inventory
    sz2.rotate(method="->ZNE", inventory=inv, components=["321"])

    # asign samplingrate and number of samples for seismometer channels
    df2 = sz2[0].stats.sampling_rate
    npts2 = sz2[0].stats.npts

    # -------------------------------------------------------------------------
    # trim to the original time window and taper again
    sz1.trim(t, t + duration)
    sz2.trim(t, t + duration)
    sz1.taper(0.1)
    sz2.taper(0.1)

    # -------------------------------------------------------------------------
    # do sanity checks
    # 1. check for sampling rate
    # 2. check for number of samples
    if df1 != df2:
        print("Sampling rates not the same, exit!!")
        sys.exit(1)

    if npts1 != npts2:
        print("Number of data points not the same, exit!!")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # return the reciever and the source channel as defined in the arguments
    r = sz1.select(channel=ch_r)
    s = sz2.select(channel=ch_s)

    return r, s


def trigger(
    tr1: Trace,
    a: int,
    b: int,
    d0: float,
    d1: float,
    c_on: float,
    c_off: float,
    start: float,
    stop: float,
    plot: bool = False,
) -> Tuple[list, list]:
    r"""
    This function searches for time spans when the steps are performed.
    STA/LTA- trigger is used to calculate the characteristic function.
    A constant offset can be applied bacause the steps are uniform.

    Parameters
    ----------:
    tr1 : :obj:`~obspy.core.Trace`
        Rotation rate recording containing steps
    a : :obj:`int`
        Number of samples for short term average
    b : :obj:`int`
        Number of samples for long term average
    d0 : :obj:`float`
        Threshold for trigger-on
    d1 : :obj:`float`
        Threshold for trigger-off
    c_on : :obj:`float`
        Constant correction for trigger at the start of each step in sec
    c_off : :obj:`float`
        Constant correction for trigger at the end of each step in sec
    start : :obj:`float`
        Offset in sec to start searching for steps
    stop : :obj:`float`
        Offset in sec to stop searching for steps

    Returns
    -------
    on : :obj:`list`
        Start time of each step
    off : :obj:`list`
        End time of each step
    """
    # define some parameters
    data1 = tr1.data
    # df1 = tr1.stats.sampling_rate

    # get the characteristic function
    cft1 = classic_sta_lta(data1, int(a), int(b))

    # you can plot it if you want
    if plot:
        plot_trigger(tr1, cft1, d0, d1)

    # find the on/off time stamps of each step
    _on = np.where(cft1 < d0)[0]
    _off = np.where(cft1 > d1)[0]

    on = []
    on0 = 0
    for i in range(len(_on) - 1):
        if _on[i + 1] - _on[i] > 1:
            trigg = _on[i] * tr1.stats.delta
            if trigg >= start and trigg <= stop:
                if np.abs((trigg + c_on) - on0) > 1.0:
                    on.append(trigg + c_on)
                    on0 = trigg + c_on
    off = []
    off0 = 0
    for i in range(len(_off) - 1):
        if _off[i + 1] - _off[i] > 1:
            trigg = _off[i] * tr1.stats.delta
            if trigg >= start and trigg <= stop:
                if np.abs((trigg + c_off) - off0) > 1.0:
                    off.append(trigg + c_off)
                    off0 = trigg + c_off

    return on, off


def find_nearest(
    t: np.ndarray, data: np.ndarray, on: float, off: float
) -> Tuple[int, int, float, float]:
    """
    This function finds the nearest sample in 'data' to 'on' and 'off'

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Array containing timestamps of samples in data.
    data : :obj:`numpy.ndarray`
        Data array where the nearest samples should be found
    on : :obj:`float`
        Time stamp found with function 'trigger()'
    off : :obj:`float`
        Time stamp found with method 'trigger()'

    Returns
    -------
    idx_on : :obj:`int`
        Index of first sample in step.
    idx_off : :obj:`int`
        Index of last sample in step.
    data_on : :obj:`float`
        Corresponding data point.
    data_off : :obj:`float`
        Corresponding data point
    """
    idx_on = (np.abs(t - on)).argmin()
    idx_off = (np.abs(t - off)).argmin()
    return idx_on, idx_off, data[idx_on], data[idx_off]


def calc_residual_disp(
    tr1: Trace, on: list, off: list, r: np.ndarray, theo: bool = False
) -> Tuple[list, list, float, float]:
    """
    This function calculates the residual displacement (lateral displacement
    introduced by the tilt motion) which is left over after tilt correction.

    Parameters
    ----------
    tr1 : :obj:`~obspy.core.Trace`
        Trace containing tilt corrected velocity recording
    on : :obj:`list`
        List of time stamps found with method 'trigger()'
    off : :obj:`list`
        List of time stamps found with method 'trigger()'
    r : :obj:`numpy.ndarray`
        Array containing theoretical residual displacement. This is only used
        to shift the traces
    theo : :obj:`bool`
        Set True if theoretical displacement is calculated

    Returns
    -------
    time : :obj:`list`
        List containing time stamps of each step.
    disp : :obj:`list`
        List containing residual displacement for each step.
    mean : :obj:`float`
        Geometric mean value of 'disp'
    sigma : :obj:`float`
        Standard deviation of 'disp'
    """

    disp_tr = []
    disp = []
    time = []

    t = np.arange(len(tr1[0].data)) / (tr1[0].stats.sampling_rate)

    for i in range(len(on)):
        idx_on, idx_off, d_0, d_1 = find_nearest(t, tr1[0].data, on[i], off[i])

        data = tr1[0].data[idx_on:idx_off]
        stats = tr1[0].stats
        stats.starttime = tr1[0].stats.starttime + idx_on * tr1[0].stats.delta
        tr = Trace(data=data, header=stats)

        # suppose that velocity is zero at the beginning and at the end of a
        # step
        if not theo:
            tr.detrend("linear")
        y0 = tr.data[0]
        tr.data = tr.data - y0

        # integrate to displacement
        tr.integrate()

        # shift the whole trace to make it comparable to theoretical
        # displacement
        y0 = tr.data[0]
        diff = y0 - r[idx_on]
        tr.data = tr.data - diff

        disp.append(tr.data)
        time.append(t[idx_on:idx_off])

        disp_tr.append(np.abs(max(tr.data) - min(tr.data)))

    mean_tr = np.mean(disp_tr)
    sigma_tr = np.std(disp_tr)

    return time, disp, mean_tr, sigma_tr


def get_angle(st: Stream, on: list, off: list) -> np.ndarray:
    """
    This method calculates the absolute angle for each step.

    Parameters
    ----------
    st : :obj:`~obspy.core.Stream`
        Stream containing integrated rotation rate data (angle).
    on : :obj:`list`
        List of time stamps found with method 'trigger()'
    off : :obj:`list`
        List of time stamps found with method 'trigger()'

    Returns
    -------
    angle :  :obj:`numpy.ndarray`
        Array containing absolute angle for each step
    """
    t = np.arange(len(st[0].data)) / (st[0].stats.sampling_rate)
    alpha = []
    for i in range(len(on)):
        idx_on, idx_off, d_0, d_1 = find_nearest(t, st[0].data, on[i], off[i])
        alpha.append(np.abs(d_0 - d_1))
    return np.asarray(alpha)


def theo_resid_disp(
    alpha0: np.ndarray, dl: float, h: float, dh: float, rr: float
) -> Tuple[float, np.ndarray]:
    """
    This function calculates the theoretical residual displacement
    induced by a tilt movement of the angle alpha0

    Parameters
    ----------
    alpha0 : :obj:`numpy.ndarray`
        Integrated rotation rate recording (angle)
    dl : :obj:`float`
        Horizontal distance between axis of rotation and center of seismometer
        [m]
    h : :obj:`float`
        Vertical distance between bottom of seismometer and seismometer mass
        [m]
    dh : :obj:`float`
        Vertical distance between bottom of seismometer and axis of rotation
        [m]
    rr : :obj:`float`
        -

    Returns
    -------
    r : :obj:`float`

    c : :obj:`numpy.ndarray`
        Array containing theoretical residual displacement
    """
    x = dl * (1.0 - np.cos(alpha0))
    y = (dh + h) * np.cos((np.pi / 2) - alpha0)
    r = -1 * (x + y)
    c = np.sqrt(dl**2 + (dh + h) ** 2) * rr**2
    return r, c


def calc_height_of_mass(
    disp: list, dl: float, dh: float, alpha: np.ndarray
) -> Tuple[float, float]:
    """
    This method calculates the vertical distance between the bottom
    of the seismometer and the seismometer mass from the residual displacement.

    Parameters
    ----------
    disp : :obj:`list`
        List containing residual displacements for each step from
        'calc_residual_disp()'
    dl : :obj:`float`
        Horizontal distance between axis of rotation and center of seismometer
        [m]
    dh : :obj:`float`
        Vertical distance between bottom of seismometer and axis of rotation
        [m]
    alpha : :obj:`numpy.ndarray`
        Rotation angles for each step from 'get_angle()'

    Returns
    -------
    mean: :obj:`float`
        Mean of vertical distance between the bottom
        of the seismometer and the seismometer mass
    std : :obj:`float`
        Standard deviation of vertical distance between the bottom
        of the seismometer and the seismometer mass
    """
    alpha0 = alpha
    X = dl * (1.0 - np.cos(alpha0))
    A = disp - X
    B = A / np.cos(alpha0)
    h = np.tan((np.pi / 2.0) - alpha0) * B - dh

    return np.mean(h), np.std(h)


def p2r(radii: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    This function converts 'radii, angle' representation of a complex number to
    'r + ij' representation

    Parameters
    ----------
    radii : :obj:`np.ndarray`
        Radius of "radius, angle" representation
    angles : :obj:`np.ndarray`
        Angle of "radius, angle" representation

    Returns
    -------
    number : :obj:`numpy.ndarray`
        'r + ij' representation
    """
    return radii * np.exp(1j * angles)


def r2p(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function converts "r + ij" representation of a complex number to
    'radii, angle representation.
    x : :obj:`np.ndarray` of :obj:`np.complex`
        Complex "r + ij" representation

    Returns
    -------
    radii : :obj:`np.ndarray`
        Radius of "radius, angle" representation
    angle : :obj:`np.ndarray`
        Angle of "radius, angle" representation
    """
    return np.abs(x), np.angle(x)
