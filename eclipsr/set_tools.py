"""ECLIPSR

This module contains functions to analyse (large) sets of light curves
using parallelisation.

-------
Example

The function analyse_set may be used to analyse large numbers of targets in parallel
If TESS data products are used (in the form of fits files containing light curves),
the function 'analyse_lc_from_tic' provides a convenient option. One then gives the
list of TIC numbers as well as a list of all file paths and their associated TIC
numbers.

>>> import eclipsr as ecl
>>> # The keyword arguments will be sent to analyse_lc_from_tic
>>> target_list = [1234567, 12345678, 23456789]
>>> all_tic =     [1234567, 12345678, 23456789, 23456789, 23456789]
>>> all_files = ['path/to/file/tic01234567.fits', 'path/to/file/tic12345678.fits', 'path/to/file/tic23456789_s1.fits',
>>>              'path/to/file/tic23456789_s2.fits', 'path/to/file/tic23456789_s3.fits']
>>> kwargs = {'all_tic': all_tic, 'all_files': all_files, 'mode': 2, 'save_dir': None, 'overwrite': False}
>>> # additional keywords may be provided for 'find_eclipses'
>>> results = ecl.analyse_set(target_list, function='analyse_lc_from_tic', n_threads=os.cpu_count()-2, **kwargs)

If we have reduced light curve files (text files with column time stamps and flux),
we may use 'analyse_lc_from_file', and provide the file names for each light curve
directly in the 'target_list'.

-----------------------------
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
empty_result = (-1, -1, -1, -1 * np.ones(6), False, False, 1, -1 * np.ones((2, 2)), -1 * np.ones((2, 2)),
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
    data: astropy.io.fits.fitsrec.FITS_rec[float], None
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
    data = np.loadtxt(file_name, delimiter=delimiter, unpack=True)
    # check number of columns present
    if (len(data) == 2):
        times, signal = data
        signal_err = np.ones(len(times))
    else:
        times, signal, signal_err = data
    # process the signal
    if np.any(signal < 0):
        signal += 1  # assume the signal varies around 0 instead of 1
    times, signal, signal_err = ut.ingest_signal(times, signal + 1, signal_err, tess_sectors=False)
    try:
        result = ecf.find_eclipses(times, signal, mode=1, max_n=80, tess_sectors=False)
    except:
        print(f'an error occurred with the following file: {file_name}')
        result = []
    return result


def analyse_lc_from_file(file_name, delimiter=None, mode=2, save_dir=None, overwrite=False, **kwargs):
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
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    overwrite: bool
        Whether to respect an existing file or overwrite it
    **kwargs: dict, optional
        Keyword arguments to be passed on to find_eclipses

    Returns
    -------
    result: tuple
        Output of the function find_eclipses (mode=2)
    """
    # check that the file exists
    if not os.path.isfile(file_name):
        print(f'File does not exist: {file_name}')
        result = empty_result
        return result

    data = np.loadtxt(file_name, delimiter=delimiter, unpack=True)
    # check number of columns present
    if (len(data) == 2):
        times, signal = data
        signal_err = np.ones(len(times))
    else:
        times, signal, signal_err = data
    # process the signal
    if np.any(signal < 0):
        signal += 1  # assume the signal varies around 0 instead of 1
    times, signal, signal_err = ut.ingest_signal(times, signal, signal_err, tess_sectors=False)
    # automatically pick an identifier for the save file
    source_id = os.path.basename(file_name)
    # catch any errors that might disrupt the execution
    try:
        result = ecf.find_eclipses(times, signal, mode=mode, **kwargs)
    except:
        print(f'An error occurred with the following file: {file_name}')
        result = empty_result
    
    if save_dir is not None:
        save_name = os.path.join(save_dir, f'{source_id}_eclipsr')
        ut.save_results(result, save_name, identifier=source_id, overwrite=overwrite)
    return result


def analyse_lc_from_tic(tic, all_tic=None, all_files=None, mode=2, save_dir=None, overwrite=False, **kwargs):
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
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    overwrite: bool
        Whether to respect an existing file or overwrite it
    **kwargs: dict, optional
        Keyword arguments to be passed on to find_eclipses

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
    # check that at least one file exists
    if not np.any([os.path.isfile(file) for file in tic_files]):
        print(f'Files do not exist for: {tic}')
        result = empty_result
        return result

    times = np.array([])
    signal = np.array([])
    qual_flags = np.array([])
    for file in tic_files:
        # check that the file exists
        if not os.path.isfile(file):
            continue
        # load in the times and flux
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
    # catch any errors that might disrupt the execution
    if (len(times) < 10):
        result = empty_result
    else:
        try:
            result = ecf.find_eclipses(times, signal, mode=mode, tess_sectors=True, **kwargs)
        except:
            print(f'an error happened in the following tic: {tic}')
            result = empty_result
    
    if save_dir is not None:
        save_name = os.path.join(save_dir, f'{tic}_eclipsr')
        ut.save_results(result, save_name, identifier=tic, overwrite=overwrite)
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
    **kwargs: dict, optional
        Extra arguments to 'function': refer to each function's
        documentation for a list of all possible arguments.

    Returns
    -------
    results: list
        Output of the function for all targets
    """
    fct.partial(eval(function), **kwargs)
    t1 = time.time()
    with mp.Pool(processes=n_threads) as pool:
        results = pool.map(fct.partial(eval(function), **kwargs), target_list)
    t2 = time.time()
    print(f'Finished analysing set in: {(t2 - t1):1.2} s ({(t2 - t1) / 3600:1.2} h) for {len(target_list)} targets,')
    print(f'using {n_threads} threads ({(t2 - t1) * n_threads / len(target_list):1.2} '
          's average per target single threaded).')
    return results
