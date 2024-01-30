"""ECLIPSR

This module contains functions to analyse (large) sets of light curves
using parallelisation.

[note] this module is a work in progress, it is recommended to understand and
modify this code to fit your needs before using it.

Code written by: Luc IJspeert
"""

import os
import time
import functools as fct
import numpy as np
import multiprocessing as mp
import astropy.io.fits as fits  # optional functionality

from . import eclipse_finding as ecf
from . import utility as ut


# globals
empty_result = (-1, -1, -1, False, False, 1, np.array([[-1., -1.], [-1., -1.]]), np.array([[-1., -1.], [-1., -1.]]),
                np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
                np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=np.int32))


def get_fits_data(file_name, index=0):
    """Returns the data from a fits file.

    Parameters
    ----------
    file_name: str
        Path to a file containing the light curve data, with
        timestamps, normalised flux, error values as the
        first three columns, respectively.
    index: int
        HDUlist index to grab

    Returns
    -------
    data: numpy.ndarray[float]
        The data portion of the fits file
    """
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    
    with fits.open(file_name, mode='readonly') as hdul:
        data = hdul[index].data
    return data


def ephem_from_file(file_name, delimiter=None):
    """Find the ephemeris for the target light curve

    Parameters
    ----------
    file_name: str
        Path to a file containing the light curve data, with
        timestamps, normalised flux, error values as the
        first three columns, respectively.
    delimiter: None, str
        Column separator for the light curve file

    Returns
    -------
    result: tuple
        Output of the function find_eclipses (mode=1)
    """
    times, signal = np.loadtxt(file_name, delimiter=delimiter, unpack=True)
    times, signal, signal_err = ut.ingest_signal(times, signal + 1, tess_sectors=False)
    try:
        result = ecf.find_eclipses(times, signal, mode=1, max_n=80, tess_sectors=False)
    except:
        print(f'an error happened in the following file: {file_name}')
        result = []
    return result


def analyse_lc_from_file(file_name, delimiter=None, mode=2, max_n=80, tess_sectors=False, save_dir=None):
    """Do all steps of the algorithm for a given light curve file

    Parameters
    ----------
    file_name: str
        Path to a file containing the light curve data, with
        timestamps, normalised flux, error values as the
        first three columns, respectively.
    delimiter: None, str
        Column separator for the light curve file
    mode: int
        Mode of operation: 0, 1, 2 or -1
        See notes for explanation of the modes.
    max_n: int
        Maximum smoothing kernel width in data points
    tess_sectors: bool
        Whether to use TESS sectors to divide up the time series
        or to see it as one continuous piece.
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.

    Returns
    -------
    result: tuple
        Output of the function find_eclipses (mode=2)
    """
    times, signal = np.loadtxt(file_name, delimiter=delimiter, unpack=True)
    times, signal, signal_err = ut.ingest_signal(times, signal + 1, tess_sectors=False)
    
    source_id = os.path.basename(file_name)
    
    try:
        result = ecf.find_eclipses(times, signal, mode=mode, max_n=max_n, tess_sectors=tess_sectors)
    except:
        print(f'an error happened in the following file: {file_name}')
        result = empty_result
    
    if save_dir is not None:
        ut.save_results(result, os.path.join(save_dir, f'{source_id}_eclipsr'), identifier=source_id)
    return result


def analyse_lc_from_tic(tic, all_tic=None, all_files=None, mode=2, max_n=80, save_dir=None):
    """Do all steps of the algorithm for a given TIC number

    Parameters
    ----------
    tic: int
        The TESS Input Catalog (TIC) number for loading/saving the data
        and later reference.
    all_tic: numpy.ndarray[int]
        List of all the TESS TIC numbers corresponding to all_files
        If all files in all_files are for the same tic, this may be None.
    all_files: numpy.ndarray[str]
        List of all the TESS data product '.fits' files. The files
        with the corresponding TIC number are selected.
    mode: int
        Mode of operation: 0, 1, 2 or -1
        See notes for explanation of the modes.
    max_n: int
        Maximum smoothing kernel width in data points
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.

    Returns
    -------
    result: tuple
        Output of the function find_eclipses (mode=2)
    """
    if all_tic is not None:
        tic_files = all_files[all_tic == tic]
    else:
        # assume all files are relevant
        tic_files = all_files

    times = np.array([])
    signal = np.array([])
    qual_flags = np.array([])
    for file in tic_files:
        tess_data = get_fits_data(file, 1)
        times = np.append(times, tess_data['TIME'])
        if ('PDCSAP_FLUX' in tess_data.columns.names):
            signal = np.append(signal, tess_data['PDCSAP_FLUX'])
        elif ('KSPSAP_FLUX' in tess_data.columns.names):
            # signal = np.append(signal, tess_data['KSPSAP_FLUX'])
            # actually use the SAP for MIT-QLP data
            signal = np.append(signal, tess_data['SAP_FLUX'])
        else:
            signal = np.append(signal, tess_data['SAP_FLUX'])
        qual_flags = np.append(qual_flags, tess_data['QUALITY'])
    
    quality = (qual_flags == 0)
    times, signal, signal_err = ut.ingest_signal(times, signal, tess_sectors=True, quality=quality)
    
    if (len(times) < 10):
        result = empty_result
    else:
        try:
            result = ecf.find_eclipses(times, signal, mode=mode, max_n=max_n, tess_sectors=True)
        except:
            print(f'an error happened in the following tic: {tic}')
            result = empty_result
    
    if save_dir is not None:
        ut.save_results(result, os.path.join(save_dir, f'{tic}_eclipsr'), identifier=tic)
    return result


def analyse_set(target_list, function='analyse_lc_from_tic', n_threads=os.cpu_count()-2, **kwargs):
    """Analyse a set of light curves in parallel

    Parameters
    ----------
    target_list: list[str], list[int]
        List of either file names or TIC identifiers to analyse
    function: str
        Name  of the function to use for the analysis
        Choose from [ephem_from_file, analyse_lc_from_file, analyse_lc_from_tic]
    n_threads: int
        Number of threads to use.
        Uses two fewer than the available amount by default.
    **kwargs: dict
        Extra arguments to 'function': refer to each function's
        documentation for a list of all possible arguments.

    Returns
    -------
    results: list
        Output of the function for all targets
    """
    print('not yet')
    fct.partial(eval(function), **kwargs)
    print('worked')
    t1 = time.time()
    with mp.Pool(processes=n_threads) as pool:
        results = pool.map(fct.partial(eval(function), **kwargs), target_list)
    t2 = time.time()
    print(f'Finished analysing set in: {(t2 - t1):1.2} s ({(t2 - t1) / 3600:1.2} h) for {len(target_list)} targets,')
    print(f'using {n_threads} threads ({(t2 - t1) * n_threads / len(target_list):1.2} '
          's average per target single threaded).')
    return results
