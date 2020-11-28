"""ECLIPSR

This module contains functions to analyse (large) sets of light curves
using parallelisation.

Code written by: Luc IJspeert
"""

import numpy as np
import astropy.io.fits as fits

from . import eclipse_finding as ecf


def get_fits_data(file_name, index=0):
    """Returns the data from a fits file.
    Optional arg: HDUlist index.
    """
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    
    with fits.open(file_name) as hdul:
        data = hdul[index].data
    return data


def ephem_test_from_file(file_name):
    times, signal = np.loadtxt(file_name, unpack=True)
    try:
        result = ecf.find_all(times, signal + 1, mode=5, max_n=80, tess_sectors=False)
    except:
        # an error happened in the following file
        print(file_name)
    return result


def ephem_test_from_csv(file_name):
    signal, times = np.loadtxt(file_name, skiprows=1, delimiter=',', unpack=True)
    try:
        result = ecf.find_all(times, signal, mode=5, max_n=80, tess_sectors=True)
    except:
        # an error happened in the following file
        print(file_name)
    return result


def ephem_from_tic(tic, all_files=None):
    tic_files = [file for file in all_files if f'-{tic:016.0f}-' in file]
    times = np.array([])
    signal = np.array([])
    for file in tic_files:
        tess_data = get_fits_data(file, 1)
        times = np.append(times, tess_data['TIME'])
        signal = np.append(signal, tess_data['PDCSAP_FLUX'])
    
    finites = np.isfinite(signal)
    times = times[finites]
    signal = signal[finites]
    sorter = np.argsort(times)
    times = times[sorter]
    signal = signal[sorter]
    if (len(times) < 3):
        result = [-2, -2, -2, -2, -2]
    else:
        # result = ecf.find_all(times, signal, mode=1, rescale=True, max_n=80, dev_limit=1.8)
        try:
            result = ecf.find_all(times, signal, mode=1, max_n=80, tess_sectors=True)
        except:
            print(tic)
            result = [-1, -1, -1, -1, -1]
    return result[:3]


def conf_from_set(file_name, results_full):
    times, signal = np.loadtxt(file_name, unpack=True)
    t_0, period, conf, sine_like, n_kernel, width_stats, depth_stats, ecl_mid, widths, depths, ratios, flags, flags_pst, ecl_indices, added_snr = results_full
    signal_s, r_derivs, s_derivs = ecf.prepare_derivatives(times, signal, n_kernel)
    if period is None:
        period = -1
    confidence = ecf.eclipse_confidence(times, signal_s, r_derivs[0], period, ecl_indices, ecl_mid, added_snr, widths, depths,
                              flags, flags_pst, return_all=True)
    return confidence