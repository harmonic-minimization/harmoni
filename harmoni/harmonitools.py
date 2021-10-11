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

import numpy as np
import multiprocessing
from functools import partial
from numpy import pi
from harmoni.extratools import hilbert_, compute_phaseconn_with_permtest


def optimize_1_gridsearch(sig_y_, sig_x2_, fs, coh, return_all=False):
    c_range = np.arange(-1, 1 + 0.01, 0.01)
    phi_range = np.arange(-pi/2, pi/2, pi / 10)
    plv_sigx_yres_c_phi = np.empty((c_range.shape[0], phi_range.shape[0]))
    for n_c, c in enumerate(c_range):
        for n_phi, phi in enumerate(phi_range):
            sig_res = sig_y_ - c * np.exp(1j * phi) * sig_x2_
            plv_sigx_yres_c_phi[n_c, n_phi] = compute_phaseconn_with_permtest(sig_x2_, sig_res, 1, 1, fs, plv_type='abs', coh=coh)
    ind_temp = np.unravel_index(np.argmin(plv_sigx_yres_c_phi), plv_sigx_yres_c_phi.shape)
    if return_all:
        return plv_sigx_yres_c_phi, c_range[ind_temp[0]], phi_range[ind_temp[1]]
    return c_range[ind_temp[0]], phi_range[ind_temp[1]]


def _regress_out_roi(parcel_series_low, parcel_series_high, fs, coh, n, opt_strat, n_lbl):
    this_lbl_ts_low = hilbert_(parcel_series_low[n_lbl], axis=1)
    this_lbl_ts_high = hilbert_(parcel_series_high[n_lbl], axis=1)

    this_lbl_high_std = np.std(np.real(this_lbl_ts_high), axis=1)
    this_lbl_ts_high /= this_lbl_high_std
    this_lbl_ts_low /= np.std(np.real(this_lbl_ts_low), axis=1)
    # this_lbl_ts_alpha2 = this_lbl_ts_low ** 2
    this_lbl_ts_low2 = np.abs(this_lbl_ts_low) * np.exp(1j * np.angle(this_lbl_ts_low) * n)
    this_lbl_ts_low2 /= np.std(np.real(this_lbl_ts_low2), axis=1)

    n_low = this_lbl_ts_low.shape[0]
    n_high = this_lbl_ts_high.shape[0]

    for n_b in range(n_high):
        for n_a in range(n_low):
            c_abs_opt_1, c_phi_opt_1 = optimize_1_gridsearch(this_lbl_ts_high[n_b, :],
                                                             this_lbl_ts_low2[n_a, :], fs, coh)
            this_lbl_ts_high[n_b, :] = \
                this_lbl_ts_high[n_b, :] - c_abs_opt_1 * np.exp(1j * c_phi_opt_1) * this_lbl_ts_low2[n_a, :]
    this_lbl_ts_high *= this_lbl_high_std
    return this_lbl_ts_high


def harmonic_removal(parcel_series_low, parcel_series_high, fs, n=2, coh=False, opt_strat='grid', mp=True, pool=None):
    """
    :param fs: [int], sampling frequency
    :param n: [int] the higher frequency is the n-th harmonic frequency of the lower frequency
    :param coh: [bool] if True then the absolute coherency is used as the synchronization measure.
                        if False, PLV is used
    :param opt_start [str] the optimization stategy
                            'grid' is the grid-search startegy
    :param mp [bool]. the multiprocessing is on if mp=True
    :param pool: the pool for mp

    :return: the list of the corrected higher frequency signal

    """
    n_parc = len(parcel_series_low)

    # exception handling: check the seize of the input signals ----------------
    assert len(parcel_series_high) == n_parc, "the two list of parcel time series should have the same length"

    for i_parc in range(n_parc):
        ts_low_i = parcel_series_low[i_parc]
        ts_high_i = parcel_series_high[i_parc]
        ts_low_i = ts_low_i[np.newaxis, :] if len(ts_low_i.shape) == 1 else ts_low_i
        ts_high_i = ts_high_i[np.newaxis, :] if len(ts_high_i.shape) == 1 else ts_high_i
        ts_low_i = ts_low_i.T if ts_low_i.shape[1] == 1 else ts_low_i
        ts_high_i = ts_high_i.T if ts_high_i.shape[1] == 1 else ts_high_i
        assert(ts_low_i.shape[-1] == ts_high_i.shape[-1]), \
            'the number of time samples of the signals of ROI {} should be the same'.format(i_parc)
        parcel_series_low[i_parc] = ts_low_i
        parcel_series_high[i_parc] = ts_high_i

    print('the harmonic correction has started, it may take a while ... ')
    if mp:
        pool = multiprocessing.Pool() if pool is None else pool
        func = partial(_regress_out_roi, parcel_series_low, parcel_series_high, fs, coh, n, opt_strat)
        parcel_series_high_corr = pool.map(func, range(n_parc))
        # pool.join()
        pool.close()

    else:
        parcel_series_high_corr = [None] * n_parc
        for i_lbl in range(n_parc):
            parcel_series_high_corr[i_lbl] = _regress_out_roi(parcel_series_low, parcel_series_high, fs,
                                                              coh, n, opt_strat, i_lbl)

    return parcel_series_high_corr


def harmonic_removal_simple(ts1, ts2, sfreq, n=2, return_all=False):
    """
    a function for running harmoni on two single time series

    :param sfreq: [int], sampling frequency
    :param n: [int] the higher frequency is the n-th harmonic frequency of the lower frequency
    :param return_all: [bool] if True all the optimizing arguments are also return

    :return: the list of the corrected higher frequency signal
    """
    # exception handling ----------------
    assert(len(ts1.shape) <= 2), \
        'The input signals cannot have a dimension with size more than 2: raise for the first signal'
    assert (len(ts2.shape) <= 2), \
        'The input signals cannot have a dimension with size more than 2: raise for the second signal'

    if len(ts1.shape) == 2:
        assert(1 in ts1.shape), \
            "the input signals should be a single time series. Fix the dimensions, raise for the first input signal"
    else:
        ts1 = ts1.ravel()

    if len(ts2.shape) == 2:
        assert(1 in ts1.shape), \
            "the input signals should be a single time series. Fix the dimensions, raise for the second input signal"
    else:
        ts2 = ts2.ravel()

    assert (ts1.shape[0] == ts2.shape[0]), \
        'the number of time samples of the two signals should be the same'

    print('the harmonic correction has started, it may take a while ... ')
    ts1_h = hilbert_(ts1)
    ts1_ = np.abs(ts1_h) * np.exp(1j * n * np.angle(ts1_h))
    ts1_ = ts1_ / np.std(np.real(ts1_))
    ts2_ = hilbert_(ts2) / np.std(np.real(ts2))

    plv_sigx_yres_c_phi_all, c_opt, phi_opt = optimize_1_gridsearch(ts2_, ts1_, sfreq, True, return_all=True)
    ts2_corr = ts2_ - c_opt * np.exp(1j * phi_opt) * ts1_
    if return_all:
        return ts2_corr, c_opt, phi_opt
    return ts2_corr
