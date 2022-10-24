"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2020
:license:
    None
"""
"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2020
:license:
    None
"""
import numpy as np
import tables as tb
from obspy.core import Stream
from obspy.io.segy.core import _read_su
from tables import *

from twistpy.polarization import PolarizationModel6C
from twistpy.polarization import TimeFrequencyAnalysis6C, SupportVectorMachine

scal = 800
traN = _read_su('ARID/Vx_Source_3750_2500.su', byteorder='<')
traE = _read_su('ARID/Vy_Source_3750_2500.su', byteorder='<')
traZ = _read_su('ARID/Vz_Source_3750_2500.su', byteorder='<')
rotN = _read_su('ARID/Rotationx_Source_3750_2500.su', byteorder='<')
rotE = _read_su('ARID/Rotationy_Source_3750_2500.su', byteorder='<')
rotZ = _read_su('ARID/Rotationz_Source_3750_2500.su', byteorder='>')

svm = SupportVectorMachine(name='arid_4')
svm.train(wave_types=['R', 'L', 'P', 'SH', 'SV', 'Noise'],
          N=5000, scaling_velocity=scal, vp=(1050, 3000), vp_to_vs=(1.7, 2.4), vl=(400, 1000),
          vr=(400, 700), phi=(0, 360), theta=(0, 80), xi=(-90, 90), free_surface=True, C=1, kernel='rbf')

# Introduce artificial reflector
x = np.arange(0, 281) * 25
sx = x[140]
d = 2000  # Interface depth
v = 3

000  # Velocity
x_off = np.abs(x - sx)  # Absolute offset from source
s = 2 * np.sqrt((x_off / 2) ** 2 + d ** 2)
theta = np.arctan(x_off / 2 / d)
t = s / v
time = np.arange(0, 690) * 0.006
from twistpy.convenience import ricker, fft_roll

wavelet, t_wav, center = ricker(time, f0=30)
wavelet = wavelet[344:1034]
t_wav = t_wav[344:1034]
data_reflection = (fft_roll(wavelet, t - 2.02, 0.006)).T

# for trace in rotZ:
#    trace.data = 0*trace.data
st = Stream(traces=[traN[20], traE[20], traZ[20], rotN[20], rotE[20], rotZ[20]])
channel_names = ['EHN', 'EHE', 'EHZ', 'EJN', 'EJE', 'EJZ']
for n, trace in enumerate(st):
    trace = st[n]
    trace.stats.channel = channel_names[n]
    trace.stats.network = 'XX'
    trace.stats.station = 'XXXX'
st.write('Nearsurface_Synthetics.mseed')

traN = traN.differentiate()
traE = traE.differentiate()
traZ = traZ.differentiate()

# data = np.asarray([x.data for x in traZ]).T


for n in range(len(t)):
    model = PolarizationModel6C(wave_type='P', vp=1500., vs=1500 / 1.7, theta=np.degrees(theta[n]), phi=90.)
    pol = model.polarization
    traN[n].data -= 30e-7 * data_reflection[:, n] * pol[0].real
    traE[n].data -= 30e-7 * data_reflection[:, n] * pol[1].real
    traZ[n].data -= 30e-7 * data_reflection[:, n] * pol[2].real
    rotN[n].data -= 30e-7 * data_reflection[:, n] * pol[3].real
    rotE[n].data -= 30e-7 * data_reflection[:, n] * pol[4].real
    rotZ[n].data -= 30e-7 * data_reflection[:, n] * pol[5].real
np.random.seed(42)

for n, trace in enumerate(traN):
    trace.data /= scal
for n, trace in enumerate(traE):
    trace.data /= 1 * scal
for n, trace in enumerate(traZ):
    trace.data /= 1 * scal
for n, trace in enumerate(rotN):
    trace.data *= 1
for n, trace in enumerate(rotE):
    trace.data *= 1
for n, trace in enumerate(rotZ):
    trace.data *= 1

for stream in [traN, traE, traZ, rotN, rotE, rotZ]:
    stream.resample(130)
    stream.taper(0.2)

#

#     stream.trim(starttime=stream[0].stats.starttime, endtime=stream[0].stats.starttime+1-1/370)

window = {'number_of_periods': 5., 'frequency_extent': 5}
N = traN[0].stats.npts


class TraceSep(IsDescription):
    p = Float64Col(N)  # double (double-precision)
    s = Float64Col(N)
    r = Float64Col(N)
    l = Float64Col(N)


h5file = tb.open_file('rayleigh_love_p_WE_3750_2500.hdf', 'w')
group1 = h5file.create_group("/", 'surfacewaves', 'Separated  data')
group2 = h5file.create_group("/", 'bodywaves', 'Separated  data')

table_tn1 = h5file.create_table(group1, 'tn', TraceSep, "Separated translational North component")
table_te1 = h5file.create_table(group1, 'te', TraceSep, "Separated translational East component")
table_tz1 = h5file.create_table(group1, 'tz', TraceSep, "Separated translational Vertical component")
table_rn1 = h5file.create_table(group1, 'rn', TraceSep, "Separated rotational North component")
table_re1 = h5file.create_table(group1, 're', TraceSep, "Separated rotational East component")
table_rz1 = h5file.create_table(group1, 'rz', TraceSep, "Separated rotational Vertical component")
table_tn2 = h5file.create_table(group2, 'tn', TraceSep, "Separated translational North component")
table_te2 = h5file.create_table(group2, 'te', TraceSep, "Separated translational East component")
table_tz2 = h5file.create_table(group2, 'tz', TraceSep, "Separated translational Vertical component")
table_rn2 = h5file.create_table(group2, 'rn', TraceSep, "Separated rotational North component")
table_re2 = h5file.create_table(group2, 're', TraceSep, "Separated rotational East component")
table_rz2 = h5file.create_table(group2, 'rz', TraceSep, "Separated rotational Vertical component")
data_tn1 = table_tn1.row
data_te1 = table_te1.row
data_tz1 = table_tz1.row
data_rn1 = table_rn1.row
data_re1 = table_re1.row
data_rz1 = table_rz1.row
data_tn2 = table_tn2.row
data_te2 = table_te2.row
data_tz2 = table_tz2.row
data_rn2 = table_rn2.row
data_re2 = table_re2.row
data_rz2 = table_rz2.row

src_tot = len(traN)

for it in range(src_tot):
    # for it in [20]:
    print(f'Computing source number: {it}/{src_tot}:\n')
    pol = TimeFrequencyAnalysis6C(traN=traN[it], traE=traE[it], traZ=traZ[it],
                                  rotN=rotN[it], rotE=rotE[it], rotZ=rotZ[it],
                                  scaling_velocity=scal, dsfacf=1, dsfact=1, window=window, timeaxis='rel',
                                  verbose=False)
    data_sep_2 = pol.filter(svm=svm, wave_types=['R', 'L'], no_of_eigenvectors=2, suppress=True)
    data_sep_1 = pol.filter(svm=svm, wave_types=['R', 'L', 'P'], no_of_eigenvectors=2)

    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(3, 1, sharex='all')
    # from matplotlib import colors
    #
    # s_t_data = np.abs(stransform(traZ[it].data.T)[0])
    # alpha = np.zeros_like(s_t_data, dtype='float')
    # alpha[s_t_data >= 0.01 * s_t_data.max().max()] = 1
    # cmap = colors.ListedColormap(['blue', 'red', 'green', 'yellow', 'white'])
    # pol.plot_classification(ax=ax[0])
    # ax[0].set_title('Estimated wave types')
    # ax[2].plot(np.arange(traN[it].stats.npts) * traN[it].stats.delta, traN[it].data.T, 'k:', label='VN')
    # ax[2].plot(np.arange(traN[it].stats.npts) * traN[it].stats.delta, traE[it].data.T, 'k--', label='VE')
    # ax[2].plot(np.arange(traN[it].stats.npts) * traN[it].stats.delta, traZ[it].data.T, 'k', label='VZ')
    # ax[2].plot(np.arange(traN[it].stats.npts) * traN[it].stats.delta, rotN[it].data.T, 'r:', label='RN')
    # ax[2].plot(np.arange(traN[it].stats.npts) * traN[it].stats.delta, rotE[it].data.T, 'r--', label='RE')
    # ax[2].plot(np.arange(traN[it].stats.npts) * traN[it].stats.delta, rotZ[it].data.T, 'r', label='RZ')
    # ax[2].autoscale(enable=True, axis='both', tight=True)
    # handles, labels = ax[2].get_legend_handles_labels()
    # ax[2].legend(handles, labels, loc='upper left')
    # ax[1].imshow(s_t_data, origin='lower', aspect='auto',
    #              extent=[0, traN[it].stats.npts * traN[it].stats.delta, 0, 1 / (2 * traN[it].stats.delta)],
    #              interpolation=None)
    # ax[0].set_ylabel('Frequency (Hz)')
    # ax[1].set_ylabel('Frequency (Hz)')
    # ax[1].set_title('S-transform of VZ')
    # ax[2].set_xlabel('Time (s)')
    # ax[2].set_title('6C input data')
    #
    # plt.style.use("ggplot")
    # pos = ax[0].get_position()
    # pos0 = ax[1].get_position()
    # ax[1].set_position([pos0.x0, pos0.y0, pos.width, pos.height])
    # pos0 = ax[2].get_position()
    # ax[2].set_position([pos0.x0, pos0.y0, pos.width, pos.height])
    #
    # plt.show()

    data_tn1['r'] = data_sep_1['R'][:, 0]
    data_te1['r'] = data_sep_1['R'][:, 1]
    data_tz1['r'] = data_sep_1['R'][:, 2]
    data_rn1['r'] = data_sep_1['R'][:, 3]
    data_re1['r'] = data_sep_1['R'][:, 4]
    data_rz1['r'] = data_sep_1['R'][:, 5]

    data_tn1['p'] = data_sep_1['P'][:, 0]
    data_te1['p'] = data_sep_1['P'][:, 1]
    data_tz1['p'] = data_sep_1['P'][:, 2]
    data_rn1['p'] = data_sep_1['P'][:, 3]
    data_re1['p'] = data_sep_1['P'][:, 4]
    data_rz1['p'] = data_sep_1['P'][:, 5]

    data_tn1['l'] = data_sep_1['L'][:, 0]
    data_te1['l'] = data_sep_1['L'][:, 1]
    data_tz1['l'] = data_sep_1['L'][:, 2]
    data_rn1['l'] = data_sep_1['L'][:, 3]
    data_re1['l'] = data_sep_1['L'][:, 4]
    data_rz1['l'] = data_sep_1['L'][:, 5]

    data_tn2['r'] = data_sep_2['R'][:, 0]
    data_te2['r'] = data_sep_2['R'][:, 1]
    data_tz2['r'] = data_sep_2['R'][:, 2]
    data_rn2['r'] = data_sep_2['R'][:, 3]
    data_re2['r'] = data_sep_2['R'][:, 4]
    data_rz2['r'] = data_sep_2['R'][:, 5]

    data_tn1.append()
    data_te1.append()
    data_tz1.append()
    data_rn1.append()
    data_re1.append()
    data_rz1.append()

    data_tn2.append()
    data_te2.append()
    data_tz2.append()
    data_rn2.append()
    data_re2.append()
    data_rz2.append()

    table_tn1.flush()
    table_te1.flush()
    table_tz1.flush()
    table_rn1.flush()
    table_re1.flush()
    table_rz1.flush()

    table_tn2.flush()
    table_te2.flush()
    table_tz2.flush()
    table_rn2.flush()
    table_re2.flush()
    table_rz2.flush()

h5file.close()
# with open(PIK, "rb") as f:
#     print pickle.load(f)
