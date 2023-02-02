import matplotlib.pyplot as plt
import numpy as np
from obspy.core import UTCDateTime

from twistpy.array_processing import BeamformingArray, plot_beam
from twistpy.convenience import generate_synthetics, to_obspy_stream

rng = np.random.default_rng(1)
aperture = 5.0  # Array aperture in meters
center_point = np.asarray(
    [100, 100, 50]
)  # Coordinates of the center point of the array (meters)
coordinates = np.tile(center_point, (4, 1)) + aperture * np.array(
    [[-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, -1]]
)
source_coordinates = np.asarray([0, 0, 0])  # Source at the origin
velocity = 6000  # Set medium velocity to 6000 m/s
dt = 2.5e-4  # sampling rate of seismograms (s)
tmax = 0.1  # total simulation time (s)
wavelet_center_frequency = 100  # wavelet center frequency in Hz
t = np.arange(0, tmax, dt)
nt = int(len(t))  # Number of samples


def test_beamforming():
    array = BeamformingArray(
        name="Pytest array", coordinates=coordinates, reference_receiver=3
    )

    data = generate_synthetics(
        source_coordinates,
        array,
        t,
        velocity=velocity,
        center_frequency=wavelet_center_frequency,
    )  # Use helper function to generate synthetics
    data += rng.normal(scale=1e-2, size=data.shape)  # Add noise for numerical stability
    start_time = UTCDateTime(
        2022, 2, 1, 10, 00, 00, 0
    )  # Recording time of first sample
    data_st = to_obspy_stream(
        data, start_time, dt
    )  # Use helper function to convert data to ObsPy Stream()

    array.add_data(data_st)

    freq_band = (90.0, 110.0)
    inclination = (
        0,
        90,
        5,
    )  # Search space for the inclination in degrees (min_value, max_value, increment)
    azimuth = (
        0,
        360,
        5,
    )
    number_of_sources = 1

    array.compute_steering_vectors(
        frequency=np.mean(freq_band),
        inclination=inclination,
        azimuth=azimuth,
        intra_array_velocity=velocity,
    )

    event_time = start_time + 0.012
    P_MUSIC = array.beamforming(
        method="MUSIC",
        event_time=event_time,
        frequency_band=freq_band,
        window=5,
        number_of_sources=number_of_sources,
    )
    P_MVDR = array.beamforming(
        method="MVDR", event_time=event_time, frequency_band=freq_band, window=5
    )
    P_BARTLETT = array.beamforming(
        method="BARTLETT", event_time=event_time, frequency_band=freq_band, window=5
    )
    assert np.invert(np.isnan(P_MUSIC).any())
    assert np.invert(np.isnan(P_MVDR).any())
    assert np.invert(np.isnan(P_BARTLETT).any())
