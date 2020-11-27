"""ECLIPSR

This module contains functions specific to TESS data

Code written by: Luc IJspeert
"""

import os
import numpy as np
import numba as nb


def get_tess_sectors(times, bjd_ref=2457000.0):
    """Load the times of the TESS sectors from a file and return a set of
    indices indicating the separate sectors in the time series.
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


# @nb.njit(cache=True)
def normalise_counts(flux_counts):
    """Median-normalises flux (counts or otherwise, should be positive) by
    dividing by the median (result varies around one and is still positive).
    """
    med = np.median(flux_counts)
    return flux_counts / med


# @nb.njit(cache=True)
def mn_to_ppm(mn_flux):
    """Converts median normalised flux to parts per million.
    The result varies around zero, instead of one.
    """
    return (mn_flux - 1) * 1e6


# @nb.njit(cache=True)
def ppm_to_mn(flux_ppm):
    """Converts from parts per million to median normalised flux.
    It is assumed that the ppm is around zero. The result varies around one.
    """
    return (flux_ppm / 1e6) + 1


# @nb.njit(cache=True)
def mn_to_mag(mn_flux):
    """Converts from parts per million to magnitude (varying around zero)."""
    return -2.5 * np.log10(mn_flux)


# @nb.njit(cache=True)
def mag_to_mn(mag):
    """Converts from magnitude (varying around zero) to median normalised flux."""
    return ppm_to_mn(10**(-0.4 * mag))


# @nb.njit(cache=True)
def norm_counts_tess(flux_counts, i_sectors):
    """Converts from flux (counts, should be positive) to parts per one,
    processing the light curve per sector.
    The result is positive and varies around one.
    """
    for s in i_sectors:
        flux_counts[s[0]:s[1]] = normalise_counts(flux_counts[s[0]:s[1]])
    return flux_counts


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