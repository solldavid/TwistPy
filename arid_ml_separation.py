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

from twistpy.polarization import TimeFrequencyAnalysis6C, SupportVectorMachine

scal = 800
traN = _read_su('ARID/Vx_Source_3750_3500.su')
traE = _read_su('ARID/Vy_Source_3750_3500.su')
traZ = _read_su('ARID/Vz_Source_3750_3500.su')
rotN = _read_su('ARID/Rotationx_Source_3750_3500.su')
rotE = _read_su('ARID/Rotationy_Source_3750_3500.su')
rotZ = _read_su('ARID/Rotationz_Source_3750_3500.su')

svm = SupportVectorMachine(name='arid')
svm.train(wave_types=['R', 'L', 'P', 'SH', 'SV', 'Noise'],
          N=5000, scaling_velocity=scal, vp=(1050, 2000), vp_to_vs=(1.7, 2.4), vl=(400, 1000),
          vr=(400, 800), phi=(0, 360), theta=(0, 80), xi=(-90, 90), free_surface=True, C=1, kernel='rbf')

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

window = {'number_of_periods': 1., 'frequency_extent': 10}
N = traN[0].stats.npts


class TraceSep(IsDescription):
    p = Float64Col(N)  # double (double-precision)
    s = Float64Col(N)
    r = Float64Col(N)
    l = Float64Col(N)


h5file = tb.open_file('separated_data_3750_2200.hdf', 'w')
group = h5file.create_group("/", 'data_sep', 'Separated  data')
table_tn = h5file.create_table(group, 'tn', TraceSep, "Separated translational North component")
table_te = h5file.create_table(group, 'te', TraceSep, "Separated translational East component")
table_tz = h5file.create_table(group, 'tz', TraceSep, "Separated translational Vertical component")
table_rn = h5file.create_table(group, 'rn', TraceSep, "Separated rotational North component")
table_re = h5file.create_table(group, 're', TraceSep, "Separated rotational East component")
table_rz = h5file.create_table(group, 'rz', TraceSep, "Separated rotational Vertical component")
data_tn = table_tn.row
data_te = table_te.row
data_tz = table_tz.row
data_rn = table_rn.row
data_re = table_re.row
data_rz = table_rz.row

src_tot = len(traN)

for it in range(src_tot):
    # for it in [20]:
    print(f'Computing source number: {it}/{src_tot}:\n')
    pol = TimeFrequencyAnalysis6C(traN=traN[it], traE=traE[it], traZ=traZ[it],
                                  rotN=rotN[it], rotE=rotE[it], rotZ=rotZ[it],
                                  scaling_velocity=scal, dsfacf=1, dsfact=1, window=window, timeaxis='rel')
    data_sep = pol.filter(svm=svm, wave_types=['P', 'SV', 'R', 'SH'], no_of_eigenvectors=1)

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

    data_tn['p'] = data_sep['P'][:, 0]
    data_te['p'] = data_sep['P'][:, 1]
    data_tz['p'] = data_sep['P'][:, 2]
    data_rn['p'] = data_sep['P'][:, 3]
    data_re['p'] = data_sep['P'][:, 4]
    data_rz['p'] = data_sep['P'][:, 5]

    data_tn['s'] = data_sep['SV'][:, 0]
    data_te['s'] = data_sep['SV'][:, 1]
    data_tz['s'] = data_sep['SV'][:, 2]
    data_rn['s'] = data_sep['SV'][:, 3]
    data_re['s'] = data_sep['SV'][:, 4]
    data_rz['s'] = data_sep['SV'][:, 5]

    data_tn['r'] = data_sep['R'][:, 0]
    data_te['r'] = data_sep['R'][:, 1]
    data_tz['r'] = data_sep['R'][:, 2]
    data_rn['r'] = data_sep['R'][:, 3]
    data_re['r'] = data_sep['R'][:, 4]
    data_rz['r'] = data_sep['R'][:, 5]

    data_tn['l'] = data_sep['SH'][:, 0]
    data_te['l'] = data_sep['SH'][:, 1]
    data_tz['l'] = data_sep['SH'][:, 2]
    data_rn['l'] = data_sep['SH'][:, 3]
    data_re['l'] = data_sep['SH'][:, 4]
    data_rz['l'] = data_sep['SH'][:, 5]

    data_tn.append()
    data_te.append()
    data_tz.append()
    data_rn.append()
    data_re.append()
    data_rz.append()

    table_tn.flush()
    table_te.flush()
    table_tz.flush()
    table_rn.flush()
    table_re.flush()
    table_rz.flush()

h5file.close()
# with open(PIK, "rb") as f:
#     print pickle.load(f)
