import numpy as np
from scipy.signal.windows import tukey

from twistpy.utils import s_transform, i_s_transform

dt = 0.01  # Sampling interval (s)
tmin = 0  # Start time of the signal (s)
tmax = 1  # End time of the signal (s)

t = np.arange(tmin, tmax, dt)  # Time axis

f1 = 10  # Frequency of first sinusoid

u = np.sin(2 * np.pi * f1 * t)

taper = tukey(len(t))  # Taper the signal to avoid edge artifacts
u *= taper
u_stran_1, f1 = s_transform(u, k=1)
u_stran_2, f2 = s_transform(u, k=3)
u_stran_3, _ = s_transform(u, dsfacf=2, k=1)
u_stran_4, _ = s_transform(u, dsfacf=3, k=1)
u_istran_1 = i_s_transform(u_stran_1, f1, k=1)
u_istran_2 = i_s_transform(u_stran_2, f1, k=3)


def test_stransform():
    assert f1[10] == f2[10]
    assert type(u_stran_1[0, 0]) == np.complex128
    assert u_stran_1.shape[0] == u_stran_1.shape[1] / 2 + 1
    assert len(f1) == u_stran_1.shape[0]
    assert len(t) == u_stran_1.shape[1]


def test_downsampling():
    assert u_stran_3.shape[0] == np.floor(u_stran_1.shape[0] / 2) + 1
    assert u_stran_3[0, 10] == u_stran_1[0, 10]
    assert u_stran_3[1, 10] == u_stran_1[2, 10]
    assert u_stran_4[1, 10] == u_stran_1[3, 10]


def test_i_stransform():
    assert np.allclose(u_istran_1, u_istran_2, atol=5e-2)
    assert np.allclose(u_istran_1, u, atol=5e-2)
    assert np.allclose(u_istran_2, u, atol=5e-2)
