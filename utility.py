"""ECLIPSR

This module contains utility functions for data processing, unit conversions
and some functions specific to TESS data.

Code written by: Luc IJspeert
"""

import os
import numpy as np
import numba as nb


@nb.njit(cache=True)
def fold_time_series(times, period, zero):
    """Fold the given time series over the orbital period to get a function of phase.
    Returns the phase array for all timestamps using the provided reference zero point.
    Returned phases are between -0.5 and 0.5
    """
    phases = ((times - zero) / period + 0.5) % 1 - 0.5
    return phases


@nb.njit(cache=True)
def runs_test(signal):
    """Bradley, (1968). Distribution-Free Statistical Tests, Chapter 12.
    To test a signal for its 'randomness'.
    Outcome of 0 means as many zero crossings as expected from a random signal.
    1 means there are more zero crossings than expected by 1 sigma (2 = 2 sigma etc.)
    -1 means there are less zero crossings than expected by 1 sigma
    So taking the absolute of the outcome gives the certainty level in sigma that the
        input signal is not random.
    [note: number of zero crossings = number of runs - 1
    See: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm
    """
    signal_above = (signal > 0).astype(np.int_)
    n_tot = len(signal)
    n_a = max(1, np.sum(signal_above))  # number above zero (make sure it doesn't become zero)
    n_b = max(1, n_tot - n_a)  # number below zero
    r = np.sum(np.abs(signal_above[1:] - signal_above[:-1])) + 1
    # expected number of or runs in a random time series
    r_exp = 2 * n_a * n_b / n_tot + 1
    # standard deviation of the number of runs
    s_r = 2 * n_a * n_b * (2 * n_a * n_b - n_tot) / (n_tot**2 * (n_tot - 1))
    z = (r - r_exp) / s_r
    return z


@nb.njit(cache=True)
def normalise_counts(flux_counts):
    """Median-normalises flux (counts or otherwise, should be positive) by
    dividing by the median (result varies around one and is still positive).
    """
    med = np.median(flux_counts)
    return flux_counts / med


@nb.njit(cache=True)
def mn_to_ppm(mn_flux):
    """Converts median normalised flux to parts per million.
    The result varies around zero, instead of one.
    """
    return (mn_flux - 1) * 1e6


@nb.njit(cache=True)
def ppm_to_mn(flux_ppm):
    """Converts from parts per million to median normalised flux.
    It is assumed that the ppm is around zero. The result varies around one.
    """
    return (flux_ppm / 1e6) + 1


# @nb.njit(cache=True)  (slowed down by jit)
def mn_to_mag(mn_flux):
    """Converts from parts per million to magnitude (varying around zero)."""
    return -2.5 * np.log10(mn_flux)


@nb.njit(cache=True)  # (not sped up significantly by jit)
def mag_to_mn(mag):
    """Converts from magnitude (varying around zero) to median normalised flux."""
    return ppm_to_mn(10**(-0.4 * mag))


@nb.njit(cache=True)  # (not sped up significantly by jit)
def norm_counts_tess(flux_counts, i_sectors):
    """Converts from flux (counts, should be positive) to parts per one,
    processing the light curve per sector.
    The result is positive and varies around one.
    """
    for s in i_sectors:
        flux_counts[s[0]:s[1]] = normalise_counts(flux_counts[s[0]:s[1]])
    return flux_counts


def get_tess_sectors(times, bjd_ref=2457000.0):
    """Load the times of the TESS sectors from a file and return a set of
    indices indicating the separate sectors in the time series.
    Make sure to use the appropriate BJD reference date for your data.
    """
    # the 0.5 offset comes from test results, and the fact that no exact JD were found (just calendar days)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # absolute dir the script is in
    jd_sectors = np.loadtxt(os.path.join(script_dir, 'tess_sectors.dat'), usecols=(2, 3)) - bjd_ref
    # use a quick searchsorted to get the positions of the sector transitions
    i_start = np.searchsorted(times, jd_sectors[:, 0])
    i_end = np.searchsorted(times, jd_sectors[:, 1])
    sectors_included = (i_start != i_end)  # this tells which sectors it received data for
    i_sectors = np.column_stack([i_start[sectors_included], i_end[sectors_included]])
    return i_sectors


@nb.njit(cache=True)
def rescale_tess(times, signal, i_sectors):
    """Scales different TESS sectors by a constant to make them match in amplitude.
    times are in TESS bjd by default, but a different bjd_ref can be given to use
    a different time reference point.
    This rescaling will make sure the rest of eclipse finding goes as intended.
    """
    signal_copy = np.copy(signal)
    # determine the range of the signal
    low = np.zeros(len(i_sectors))
    high = np.zeros(len(i_sectors))
    averages = np.zeros(len(i_sectors))
    threshold = np.zeros(len(i_sectors))
    for i, s in enumerate(i_sectors):
        masked_s = signal[s[0]:s[1]]
        averages[i] = np.mean(masked_s)
        low[i] = np.mean(masked_s[masked_s < averages[i]])
        high[i] = np.mean(masked_s[masked_s > averages[i]])
        threshold[i] = np.mean(masked_s[masked_s > high[i]])
        threshold[i] = np.mean(masked_s[masked_s > threshold[i]])
    
    difference = high - low
    if np.any(difference != 0):
        min_diff = np.min(difference[difference != 0])
    else:
        min_diff = 0
    threshold = 2 * threshold - averages  # to remove spikes (from e.g. momentum dumps)
    thr_mask = np.ones(len(times), dtype=np.bool_)
    # adjust the signal so that it has a more uniform range (and reject (mask) upward outliers)
    for i, s in enumerate(i_sectors):
        signal_copy[s[0]:s[1]] = (signal[s[0]:s[1]] - averages[i]) / difference[i] * min_diff + averages[i]
        thr_mask[s[0]:s[1]] = (signal[s[0]:s[1]] < threshold[i])
    return signal_copy, thr_mask


def check_constant(signal):
    """Does a simple check to see if the signal is worth while processing further.
    The signal must be median normalised.
    The 10th percentile of the signal centered around zero is compared to the
    10th percentile of the point-to-point differences.
    """
    low = 1 - np.percentile(signal, 10)
    low_diff = abs(np.percentile(np.diff(signal), 10))
    return (low < low_diff)


def ingest_signal(times, signal, tess_sectors=True):
    """Take a signal and process it for ingest into the algorithm.
    
    The signal (raw counts or ppm) will be median normalised
    after the removal of non-finite values.
    [Note] signal must not be mean subtracted (or otherwise negative)!
    If your signal is already 'clean' and normalised to vary around 1,
    skip this function.
    
    If tess_sectors is True, each sector is handled separately and
    the signal will be rescaled for more consistent eclipse depths across sectors.
    The separate sectors are also rescaled to better match in amplitude.
    
    Outputs the processed times and signal.
    """
    finites = np.isfinite(signal)
    times = times[finites].astype(np.float_)
    signal = signal[finites].astype(np.float_)
    if tess_sectors:
        i_sectors = get_tess_sectors(times)
        if (len(i_sectors) == 0):
            raise ValueError('given times do not fall into any TESS sectors. '
                             'Set tess_sectors=False or change the reference BJD.')
        signal = norm_counts_tess(signal, i_sectors)
        # rescale the different TESS sectors for more consistent amplitude and better operation
        signal, thr_mask = rescale_tess(times, signal, i_sectors)
        # remove upward outliers
        times = times[thr_mask]
        signal = signal[thr_mask]
    else:
        signal = normalise_counts(signal)
    return times, signal
