r"""
Array Processing: Beamforming
=================================
In this tutorial, you will learn how to use TwistPy's Array processing tools for beamforming.
"""
import matplotlib.pyplot as plt
import numpy as np
from obspy.core import UTCDateTime

from twistpy.array_processing import BeamformingArray, plot_beam, beam
from twistpy.convenience import generate_synthetics, to_obspy_stream

rng = np.random.default_rng(1)

# sphinx_gallery_thumbnail_number = 4

########################################################################################################################
# Instantiate an object of class BeamformingArray(), the basic interface that we use to perform array processing.
# We pass a name to uniquely identify the array and the coordinates of the receivers. The coordinates should be defined
# in a left-handed coordinate system as North, East, Up. Additionally, we specify the index
# of a reference receiver, all phase delays inside the array are computed with respect to this receiver:
coordinates = np.array([[2250, 0, -7], [2250, 3.75877048314363, -4.36808057330267], [2252, 0, -3], [2248, 0, -3]])

coordinates2 = np.array([[2680, 0, -7], [2680, 3.75877048314363, -4.36808057330267], [2682, 0, -3], [2678, 0, -3]])


# Instantiate BeamformingArray object (the fourth receiver (with index 3) is specified to be the reference receiver)
array1 = BeamformingArray(name='Array 1', coordinates=coordinates)
array2 = BeamformingArray(name='Array 2', coordinates=coordinates2)

source_coordinates = source_coordinates = np.array([2358.25055303971, 138.661319613372, -17.3205080756888]) \
                                          + 0.5 * np.array([2347.00300032252 - 2358.25055303971, 187.379822852633
                                                            - 138.661319613372, 0]) \
                                          + 0.5 * np.array([2371.36225194215 - 2358.25055303971, 193.00359921123
                                                            - 138.661319613372, -60.6217782649107 + 17.3205080756888])

########################################################################################################################
# Plot the source-receiver configuration:

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(array1.coordinates[:, 1], array1.coordinates[:, 0], array1.coordinates[:, 2], marker='v', color='b',
           label='Array 1')
ax.scatter(array2.coordinates[:, 1], array2.coordinates[:, 0], array2.coordinates[:, 2], marker='v', color='r',
           label='Array 2')
ax.scatter(source_coordinates[1], source_coordinates[0], source_coordinates[2], marker='*', color='g', label='Source')
ax.legend()
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))

########################################################################################################################
# Specify medium and data parameters for the computation of synthetics:

velocity = 5500  # Set medium velocity to 5500 m/s
dt = 2.5e-5  # sampling rate of seismograms (s)
tmax = 0.1  # total simulation time (s)
wavelet_center_frequency = 1000  # wavelet center frequency in Hz
t = np.arange(0, tmax, dt)
nt = int(len(t))  # Number of samples

# Generate synthetic data using a Ricker wavelet
data1 = generate_synthetics(source_coordinates, array1, t, velocity=velocity, center_frequency=wavelet_center_frequency)
# Use helper function to generate synthetics
data2 = generate_synthetics(source_coordinates, array2, t, velocity=velocity, center_frequency=wavelet_center_frequency)
# Use helper function to generate synthetics
data1 += rng.normal(scale=1e-3, size=data1.shape)  # Add noise for numerical stability
data2 += rng.normal(scale=1e-3, size=data2.shape)  # Add noise for numerical stability

# Plot data
plt.figure()
plt.plot(t, data1.T)
plt.legend(['Receiver 0', 'Receiver 1', 'Receiver 2', 'Receiver 3'])

plt.figure()
plt.plot(t, data2.T)
plt.legend(['Receiver 0', 'Receiver 1', 'Receiver 2', 'Receiver 3'])
########################################################################################################################
# We now need to feed the data to our BeamformingArray object. BeamformingArray objects only accept data in ObsPy
# Stream() format.
# We therefore convert the synthetic data to an ObsPy Stream:

# Convert data to Obspy Stream format
start_time = UTCDateTime(2022, 2, 1, 10, 00, 00, 0)  # Recording time of first sample
data_st_1 = to_obspy_stream(data1, start_time, dt)  # Use helper function to convert data to ObsPy Stream()
data_st_2 = to_obspy_stream(data2, start_time, dt)  # Use helper function to convert data to ObsPy Stream()

print(data_st_1)

# Add data to array object
array1.add_data(data_st_1)# add_data() will automatically check that the data is in  Stream() format and that the
array2.add_data(data_st_2)
# number of traces in the data agrees with the number of receivers

########################################################################################################################
# Specify parameters for array processing:
# Frequency band over which to perform frequency-domain beamforming

freq_band = (950., 1050.)
inclination = (0, 90, 1)  # Search space for the inclination in degrees (min_value, max_value, increment)
azimuth = (0, 360, 1)  # Search space for the back-azimuth in degrees (min_value, max_value, increment)
velocity = 5500.  # Intra-array velocity in m/s, either a float or a tuple as for azimuth and inclination if velocity
# is unknown and part of the search
number_of_sources = 1  # Specify the number of interfering sources that will be estimated in the time window
# (only relevant for MUSIC)

########################################################################################################################
# Now we have everything we need to compute the steering vectors:

array1.compute_steering_vectors(frequency=np.mean(freq_band), inclination=inclination, azimuth=azimuth,
                               intra_array_velocity=velocity)
array2.compute_steering_vectors(frequency=np.mean(freq_band), inclination=inclination, azimuth=azimuth,
                               intra_array_velocity=velocity)

########################################################################################################################
# Let's now perform beamforming at a specific time. Currently, three different beamforming methods are implemented:
# 'MUSIC', 'MVDR' (minimum variance distortionless response or Capon Beamformer), and 'BARTLETT'
# (conventional beamforming). To compute the beam power at time=event_time, we do:

event_time = start_time + 0.0385  # Pick time, where the start of the analysis window is placed
event_time2 = start_time + 0.066  # Pick time, where the start of the analysis window is placed

# For a time dependent analysis, slide event_time down the trace
P_MUSIC_1 = array1.beamforming(method='MUSIC', event_time=event_time, frequency_band=freq_band, window=5,
                            number_of_sources=number_of_sources)
P_MVDR_1 = array1.beamforming(method='MVDR', event_time=event_time, frequency_band=freq_band, window=5)
P_BARTLETT_1 = array1.beamforming(method='BARTLETT', event_time=event_time, frequency_band=freq_band, window=5)

P_MUSIC_2 = array2.beamforming(method='MUSIC', event_time=event_time2, frequency_band=freq_band, window=5,
                            number_of_sources=number_of_sources)
P_MVDR_2 = array2.beamforming(method='MVDR', event_time=event_time2, frequency_band=freq_band, window=5)
P_BARTLETT_2 = array2.beamforming(method='BARTLETT', event_time=event_time2, frequency_band=freq_band, window=5)
# Window specifies the width of the time window, here corresponding to 5 times the dominant period in the specified
# frequency band
# If you want to perform a time-depenent analysis you would slide the window down the data by adjusting event_time

########################################################################################################################
# Plot the results. The azimuth is defined clock-wise from the North axis. The inclination is measured from the vertical
# axis downward. The extracted azimuth and inclination point into the direction of propagation of the wave (away from
# the source).

# Compute the true propagation direction at the center point of the array as a reference to evaluate beamforming
# performance

center_point = array1.coordinates[0, :]
center_point2 = array2.coordinates[0, :]
propagation_direction = array1.coordinates[0, :] - source_coordinates
print(np.linalg.norm(propagation_direction))
propagation_direction = propagation_direction / np.linalg.norm(propagation_direction)

# negative to go from counter-clockwise to clockwise definition of angles
azimuth_true = np.arctan2(propagation_direction[1], propagation_direction[0])
if azimuth_true < 0:
    azimuth_true += 2 * np.pi
inclination_true = np.arctan(np.linalg.norm(propagation_direction[:-1]) / propagation_direction[2])

# Plot beamforming results obtained with the 3 different methods
fig_bf, ax_bf = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 12))
ax_bf[0].imshow(P_MUSIC_1, extent=[azimuth[0], azimuth[1], inclination[0], inclination[1]], origin='lower')
ax_bf[0].plot(np.degrees(azimuth_true), np.degrees(inclination_true), 'r*')
ax_bf[0].set_xlabel('Azimuth (degrees)')
ax_bf[0].set_ylabel('Inclination (degrees)')
ax_bf[0].legend(['True'], loc='center left', bbox_to_anchor=(1, 0.5))
ax_bf[0].set_title('MUSIC')

ax_bf[1].imshow(P_MVDR_1, extent=[azimuth[0], azimuth[1], inclination[0], inclination[1]], origin='lower')
ax_bf[1].plot(np.degrees(azimuth_true), np.degrees(inclination_true), 'r*')
ax_bf[1].set_xlabel('Azimuth (degrees)')
ax_bf[1].set_ylabel('Inclination (degrees)')
ax_bf[1].legend(['True'], loc='center left', bbox_to_anchor=(1, 0.5))
ax_bf[1].set_title('MVDR (Capon)')

ax_bf[2].imshow(P_BARTLETT_1, extent=[azimuth[0], azimuth[1], inclination[0], inclination[1]], origin='lower')
ax_bf[2].plot(np.degrees(azimuth_true), np.degrees(inclination_true), 'r*')
ax_bf[2].set_xlabel('Azimuth (degrees)')
ax_bf[2].set_ylabel('Inclination (degrees)')
ax_bf[2].legend(['True'], loc='center left', bbox_to_anchor=(1, 0.5))
ax_bf[2].set_title('BARTLETT')

########################################################################################################################
# For a polar plot with the azimuth plotted as the polar angle and the inclination as the radius:

azi_plot = np.arange(azimuth[0], azimuth[1] + azimuth[2], azimuth[2])
inc_plot = np.arange(inclination[0], inclination[1] + inclination[2], inclination[2])

azi_plot, inc_plot = np.meshgrid(np.radians(azi_plot), inc_plot)

fig_bf_polar, ax_bf_polar = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 6), subplot_kw=dict(polar=True))
for ax_p in ax_bf_polar:
    ax_p.set_theta_direction(-1)
    ax_p.set_theta_offset(np.pi / 2.0)
ax_bf_polar[0].pcolormesh(azi_plot, inc_plot, P_MUSIC_1.squeeze())
ax_bf_polar[0].set_title('MUSIC')

ax_bf_polar[1].pcolormesh(azi_plot, inc_plot, P_MVDR_1.squeeze())
ax_bf_polar[1].set_title('MVDR (Capon)')

ax_bf_polar[2].pcolormesh(azi_plot, inc_plot, P_BARTLETT_1.squeeze())
ax_bf_polar[2].set_title('BARTLETT')

########################################################################################################################
# To visualize the beam power in 3D, we use a simple back-projection of the beam assuming straight rays:

# Plot the beam in 3D domain
nx, ny, nz = 121, 121, 61  # number of points in x-, y- and z- direction where beam is plotted
xmin, xmax = 2248, 2682  # min and max locations in x-direction where beam will be plotted
ymin, ymax = 0, 190
zmin, zmax = -60, 0
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
z = np.linspace(zmin, zmax, nx)

X, Y, Z, = np.meshgrid(x, y, z)
grid = np.asarray([X.flatten(), Y.flatten(), Z.flatten()]).T
beam_origin = center_point  # location where the origin of the beam is located

# Plot the beam intensity for the three different methods
fig_beam, ax_beam = plt.subplots(3, 1, figsize=(10, 15), subplot_kw={'projection': '3d'})

plot_beam(grid, beam_origin, P_MUSIC_1, inclination, azimuth, ax=ax_beam[0])  # Helper function to plot the beam power
plot_beam(grid, center_point2, P_MUSIC_2, inclination, azimuth, ax=ax_beam[0])  # Helper function to plot the beam power
ax_beam[0].scatter(array1.coordinates[:, 1], array1.coordinates[:, 0], array1.coordinates[:, 2], marker='v', color='b')
ax_beam[0].scatter(array2.coordinates[:, 1], array2.coordinates[:, 0], array2.coordinates[:, 2], marker='v', color='r')
ax_beam[0].scatter(source_coordinates[1], source_coordinates[0], source_coordinates[2], marker='*', color='g')
ax_beam[0].set_xlabel('Y')
ax_beam[0].set_ylabel('X')
ax_beam[0].set_zlabel('Z')
ax_beam[0].set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
ax_beam[0].set_title('MUSIC')

plot_beam(grid, beam_origin, P_MVDR_1, inclination, azimuth, ax=ax_beam[1], clip=0.05)
plot_beam(grid, center_point2, P_MVDR_2, inclination, azimuth, ax=ax_beam[1], clip=0.05)
ax_beam[1].scatter(array1.coordinates[:, 1], array1.coordinates[:, 0], array1.coordinates[:, 2], marker='v', color='b')
ax_beam[1].scatter(array2.coordinates[:, 1], array2.coordinates[:, 0], array2.coordinates[:, 2], marker='v', color='r')
ax_beam[1].scatter(source_coordinates[1], source_coordinates[0], source_coordinates[2], marker='*')
ax_beam[1].set_xlabel('Y')
ax_beam[1].set_ylabel('X')
ax_beam[1].set_zlabel('Z')
ax_beam[1].set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
ax_beam[1].set_title('MVDR (Capon)')

plot_beam(grid, beam_origin, P_BARTLETT_1, inclination, azimuth, ax=ax_beam[2], clip=0.8)
plot_beam(grid, center_point2, P_BARTLETT_2, inclination, azimuth, ax=ax_beam[2], clip=0.8)
ax_beam[2].scatter(array1.coordinates[:, 1], array1.coordinates[:, 0], array1.coordinates[:, 2], marker='v', color='b')
ax_beam[2].scatter(array2.coordinates[:, 1], array2.coordinates[:, 0], array2.coordinates[:, 2], marker='v', color='r')
ax_beam[2].scatter(source_coordinates[1], source_coordinates[0], source_coordinates[2], marker='*')
ax_beam[2].set_xlabel('E')
ax_beam[2].set_ylabel('N')
ax_beam[2].set_zlabel('Z')
ax_beam[2].set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
ax_beam[2].set_title('BARTLETT')

beam_power_1_MUSIC = beam(grid, beam_origin, P_MUSIC_1, inclination, azimuth)
beam_power_2_MUSIC = beam(grid, center_point2, P_MUSIC_2, inclination, azimuth)

# Imaging condition (find parts of the two beams that overlap)
clip = 0.2
bp_MUSIC = beam_power_1_MUSIC * beam_power_2_MUSIC
grid = grid[bp_MUSIC > clip * bp_MUSIC.max().max(), :]
bp_MUSIC = bp_MUSIC[bp_MUSIC > clip * bp_MUSIC.max().max()]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(grid[:, 1], grid[:, 0], grid[:, 2], marker='.', c=bp_MUSIC, cmap='inferno')
ax.scatter(array1.coordinates[:, 1], array1.coordinates[:, 0], array1.coordinates[:, 2], marker='v', color='b')
ax.scatter(array2.coordinates[:, 1], array2.coordinates[:, 0], array2.coordinates[:, 2], marker='v', color='r')
ax.scatter(source_coordinates[1], source_coordinates[0], source_coordinates[2], marker='*', color='g')
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)
ax.set_zlim(zmin, zmax)
ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
ax.view_init(elev=27., azim=-44)
ax.set_title('Combined beam intensity provides 3D location (MUSIC)')

########################################################################################################################
# Projection onto fault
dr = 2
tm = 2380
dip = 60
dipdir = 330
azi = dipdir+90-360
azi_bt = azi + 43
dipdir_bt = dipdir + 43

rs = np.arange(-200, 2700+dr, dr)
rd = np.arange(0, 120+dr, dr)
RS, RD = np.meshgrid(rs, rd)

x0 = tm
y0 = 0
z0 = 0
strike = azi_bt

dipdir = (strike-90)*np.pi/180
dip = dip*np.pi/180
strike = strike*np.pi/180

x = x0 + RS*np.cos(strike) + RD*np.cos(dip)*np.cos(dipdir)
y = y0 + RS*np.sin(strike) + RD*np.cos(dip)*np.sin(dipdir)
z = - (z0 + RD*np.sin(dip))

grid = np.asarray([x.ravel(), y.ravel(), z.ravel()]).T

beam_power_fault_1_MUSIC = beam(grid, beam_origin, P_MUSIC_1, inclination, azimuth)
beam_power_fault_2_MUSIC = beam(grid, center_point2, P_MUSIC_2, inclination, azimuth)


# Imaging condition (find parts of the two beams that overlap)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(grid[:, 1], grid[:, 0], grid[:, 2], marker='.', c=beam_power_fault_1_MUSIC, cmap='inferno')
ax.scatter(array1.coordinates[:, 1], array1.coordinates[:, 0], array1.coordinates[:, 2], marker='v', color='b')
ax.scatter(array2.coordinates[:, 1], array2.coordinates[:, 0], array2.coordinates[:, 2], marker='v', color='r')
ax.scatter(source_coordinates[1], source_coordinates[0], source_coordinates[2], marker='*', color='g')
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)
ax.set_zlim(zmin, zmax)
ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
ax.view_init(elev=32., azim=19)
ax.set_title('Projection onto fault (array 1)')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(grid[:, 1], grid[:, 0], grid[:, 2], marker='.', c=beam_power_fault_2_MUSIC, cmap='inferno')
ax.scatter(array1.coordinates[:, 1], array1.coordinates[:, 0], array1.coordinates[:, 2], marker='v', color='b')
ax.scatter(array2.coordinates[:, 1], array2.coordinates[:, 0], array2.coordinates[:, 2], marker='v', color='r')
ax.scatter(source_coordinates[1], source_coordinates[0], source_coordinates[2], marker='*', color='g')
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)
ax.set_zlim(zmin, zmax)
ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
ax.view_init(elev=32., azim=19)
ax.set_title('Projection onto fault (array 2)')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(grid[:, 1], grid[:, 0], grid[:, 2], marker='.', c=beam_power_fault_1_MUSIC*beam_power_fault_2_MUSIC, cmap='inferno')
ax.scatter(array1.coordinates[:, 1], array1.coordinates[:, 0], array1.coordinates[:, 2], marker='v', color='b')
ax.scatter(array2.coordinates[:, 1], array2.coordinates[:, 0], array2.coordinates[:, 2], marker='v', color='r')
ax.scatter(source_coordinates[1], source_coordinates[0], source_coordinates[2], marker='*', color='g')
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)
ax.set_zlim(zmin, zmax)
ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
ax.view_init(elev=32., azim=19)
ax.set_title('Projection onto fault (combined)')
plt.show()


