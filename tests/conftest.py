import pytest
from scipy.signal import butter, filtfilt
import numpy as np
from harmoni.extratools import hilbert_


@pytest.fixture(scope="session")
def general_setting():
    sfreq = 512  # sampling frequency
    f0 = 10  # fundamental frequency
    t_len = 60 * 2  # total simulation time
    n_samp = sfreq * t_len  # number of time samples
    n = 2  # the ratio of the fast and slow oscillations, i.e. the harmonic of interest

    # the filter coefficients of the frequency band of the fundamental frequency
    b1, a1 = butter(N=2, Wn=np.array([f0 - 1, f0 + 1]) / sfreq * 2, btype='bandpass')

    # the filter coefficients of the frequency band of the n-th frequency
    b2, a2 = butter(N=2, Wn=np.array([n * f0 - 1, n * f0 + 1]) / sfreq * 2, btype='bandpass')
    return sfreq, b1, a1, b2, a2, n, n_samp


@pytest.fixture()
def generate_nonsinsig_fortest(general_setting):
    """
    help function to generate non-sinusoidal signal for test functions

    :param general_setting: the fixture to generate the general settings
    :return: a non-signusoidal signal with fundamental and 2nd harmonic components
    """
    sfreq, b1, a1, b2, a2, n, n_samp = general_setting
    x1 = filtfilt(b1, a1, np.random.randn(n_samp))
    x1_h = hilbert_(x1)
    y1 = np.real(np.abs(x1_h) * np.exp(1j * n * np.angle(x1_h)))
    return x1 + 0.25 * y1


@pytest.fixture()
def generate2_nonsinsig_fortest(general_setting):
    """
    help function to generate non-sinusoidal signal for test functions

    :param general_setting: the fixture to generate the general settings

    :return:
    z1: a non-signusoidal signal with fundamental and 2nd harmonic components
    z2: a mixed signal, whose alpha and beta components are not harmonics (they do not have phase synchronization)
    note: the beta part of z2 has 1:2 synchronization to the alpha part of z1

    """
    sfreq, b1, a1, b2, a2, n, n_samp = general_setting
    x1 = filtfilt(b1, a1, np.random.randn(n_samp))
    x1_h = hilbert_(x1)
    y1 = np.real(np.abs(x1_h) * np.exp(1j * n * np.angle(x1_h)))
    phi0 = np.pi / 2 * np.random.random(1) + np.pi / 4
    x2 = filtfilt(b1, a1, np.random.randn(n_samp))
    y2 = np.real(np.abs(x1_h) * np.exp(1j * n * np.angle(x1_h)) + phi0)
    z1 = x1 + 0.25 * y1
    z2 = x2 + 0.25 * y2
    return z1, z2
