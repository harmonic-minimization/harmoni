"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Juanli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
https://doi.org/10.1101/2021.10.06.463319
-----------------------------------------------------------------------

(c) Author: Mina Jamshidi (minajamshidi91@gmail.com)
https://github.com/minajamshidi
(c) please cite the above paper in case of adaptation, and/or usage of this code for your research

# License: BSD-3-Clause
-----------------------------------------------------------------------
"""

from scipy.signal import filtfilt
import numpy as np
import pytest
from harmoni.extratools import compute_plv
from harmoni import harmonic_removal, harmonic_removal_simple


# -----------------------------------------------------
# testing the main functions
# -----------------------------------------------------

def test_harmonic_removal_simple(generate_nonsinsig_fortest, general_setting):
    """
    to test the harmonic_removal_simple() function.

    test procedure:
    ------------------
    generate a non-sin signal and remove the harmonic part.

    """
    sig_nonsin = generate_nonsinsig_fortest
    sfreq, b1, a1, b2, a2, n, n_samp = general_setting
    sig1 = filtfilt(b1, a1, sig_nonsin)
    sig2 = filtfilt(b2, a2, sig_nonsin)

    sig2_res, c_opt, _ = harmonic_removal_simple(sig1, sig2, sfreq, n=n, return_all=True)
    coh1 = compute_plv(sig1, sig2, 1, n, plv_type='abs', coh=True)
    coh2 = compute_plv(sig1, sig2_res, 1, n, plv_type='abs', coh=True)
    assert (coh2 < coh1)
    assert(coh2 < 0.1)
    assert(c_opt > 0.9)


def test_harmonic_removal_mp(generate2_nonsinsig_fortest, general_setting):
    """
    - test the parallel option of the harmonic_removal function
    - test to make sure that the function performs good in the sense that the non-harmonic high-freq components
    are not removed (sig22 is synchronized to sig11, but it is non-harmonic, i.e. it is not synchronized to sig21)

    about the test procedure:
    ---------------------------
    z1 = sig11 + sig12
    z2 = sig21 + sig22

    sigi1 --> alpha band
    sigi2 --> beta band

    interactions:

    (1) sig11 <-- 1:2 --> sig12  (harmonic-driven)  this should be diminished after harmoni
    (2) sig11 <-- 1:2 --> sig22  (non-harmonic)     this should not be touched after harmoni
    (3) sig12 < -- x --> sig22                      this should remain small
    """

    sfreq, b1, a1, b2, a2, n, n_samp = general_setting
    z1, z2 = generate2_nonsinsig_fortest
    sig11 = filtfilt(b1, a1, z1)
    sig12 = filtfilt(b2, a2, z1)
    sig21 = filtfilt(b1, a1, z2)
    sig22 = filtfilt(b2, a2, z2)

    parcel_series_low = [sig11, sig21]
    parcel_series_high = [sig12, sig22]
    parcel_series_high_res = harmonic_removal(parcel_series_low, parcel_series_high, sfreq, n=n,
                                              coh=True, opt_strat='grid', mp=True, pool=None)
    sig12_res, sig22_res = parcel_series_high_res

    coh11_b = compute_plv(sig11, sig12, 1, n, plv_type='abs', coh=True)
    coh11_a = compute_plv(sig11, sig12_res, 1, n, plv_type='abs', coh=True)
    coh22_b = compute_plv(sig21, sig22, 1, n, plv_type='abs', coh=True)
    coh22_a = compute_plv(sig21, sig22_res, 1, n, plv_type='abs', coh=True)
    coh12_b = compute_plv(sig11, sig22, 1, n, plv_type='abs', coh=True)
    coh12_a = compute_plv(sig11, sig22_res, 1, n, plv_type='abs', coh=True)

    assert(coh11_a < coh11_b)  # criterion 1
    assert(coh11_a < coh22_b)  # criterion 1
    assert(coh22_a < coh22_b)  # criterion 3
    assert(np.abs(coh12_b - coh12_a) < 0.05)  # criterion 2


def test_harmonic_removal_seq(generate2_nonsinsig_fortest, general_setting):
    """
    the same as test_harmonic_removal_mp(), but for testing of non-parallel or sequetial option of harmonic_removal()
    """

    sfreq, b1, a1, b2, a2, n, n_samp = general_setting
    z1, z2 = generate2_nonsinsig_fortest
    sig11 = filtfilt(b1, a1, z1)
    sig12 = filtfilt(b2, a2, z1)
    sig21 = filtfilt(b1, a1, z2)
    sig22 = filtfilt(b2, a2, z2)

    parcel_series_low = [sig11, sig21]
    parcel_series_high = [sig12, sig22]
    parcel_series_high_res = harmonic_removal(parcel_series_low, parcel_series_high, sfreq, n=n,
                                              coh=True, opt_strat='grid', mp=False, pool=None)
    sig12_res, sig22_res = parcel_series_high_res

    coh11_b = compute_plv(sig11, sig12, 1, n, plv_type='abs', coh=True)
    coh11_a = compute_plv(sig11, sig12_res, 1, n, plv_type='abs', coh=True)
    coh22_b = compute_plv(sig21, sig22, 1, n, plv_type='abs', coh=True)
    coh22_a = compute_plv(sig21, sig22_res, 1, n, plv_type='abs', coh=True)
    coh12_b = compute_plv(sig11, sig22, 1, n, plv_type='abs', coh=True)
    coh12_a = compute_plv(sig11, sig22_res, 1, n, plv_type='abs', coh=True)

    assert(coh11_a < coh11_b)
    assert(coh11_a < coh22_b)
    assert(coh22_a < coh22_b)
    assert(np.abs(coh12_b - coh12_a) < 0.05)


# -----------------------------------------------------
# testing that the exceptions are handled
# -----------------------------------------------------
@pytest.mark.parametrize("ts1_2, ts2_2", [
    ([np.random.randn(10), np.random.randn(9)], [np.random.randn(10), np.random.randn(10)]),
    ([np.random.randn(9)], [np.random.randn(10)]),
    ([np.random.randn(5, 9)], [np.random.randn(5, 10)]),
    ([np.random.randn(5, 9)], [np.random.randn(4, 10)]),
    ([np.random.randn(10), np.random.randn(9)], [np.random.randn(10)]),
])
def test_harmonic_removal_size(general_setting, ts1_2, ts2_2):
    """
    tests that the number of time samples of the ROI signals are similar - for two ROIs
    """
    sfreq, _, _, _, _, n, _ = general_setting
    with pytest.raises(AssertionError):
        harmonic_removal(ts1_2, ts2_2, sfreq, n=n, coh=True, opt_strat='grid', mp=True, pool=None)


@pytest.mark.parametrize("ts1_1, ts2_1", [
    (np.random.randn(10, 5), np.random.randn(9)),
    (np.random.randn(9), np.random.randn(10, 5)),
    (np.random.randn(9), np.random.randn(10, 5, 3)),
    (np.random.randn(10, 5, 3), np.random.randn(9)),
    (np.random.randn(9), np.random.randn(5)),
])
def test_harmonic_removal_simple_size(general_setting, ts1_1, ts2_1):
    """
    tests that the AssertionError works if one of the inputs contains more than a signal
    """
    sfreq, _, _, _, _, n, _ = general_setting
    with pytest.raises(AssertionError):
        harmonic_removal_simple(ts1_1, ts2_1, sfreq, n=n, return_all=True)
