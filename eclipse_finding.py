"""This module contains functions to find eclipses, measure their properties and
do various miscellaneous things with them (or on arrays in general).

Code written by: Luc IJspeert
"""

import math as m
import numpy as np
import scipy as sp
import scipy.signal
import scipy.spatial
import scipy.special
import numba as nb
import matplotlib.pyplot as plt

import fitshandler as fh


@nb.jit(nopython=True)
def cut_eclipses(times, eclipses):
    """Returns a boolean mask covering up the eclipses.
    Give the eclipse times as a series of two points in time.
    See also: mask_eclipses
    mask_eclipses is a lot faster.
    Can of course be used to cover up any set of two time points.
    """
    mask = np.ones(len(times), dtype=np.bool_)
    for ecl in eclipses:
        mask = mask & ((times < ecl[0]) | (times > ecl[-1]))
    return mask


@nb.jit(nopython=True)
def mask_eclipses(times, eclipses):
    """Returns a boolean mask covering up the eclipses.
    Give the eclipse indices as a series of two indices per eclipse.
    See also: cut_eclipses
    Can of course be used to cover up any set of two indices.
    """
    mask = np.ones(len(times), dtype=np.bool_)
    for ecl in eclipses:
        mask[ecl[0]:ecl[-1] + 1] = False  # include the right point in the mask
    return mask


def rescale_tess(times, signal, bjd_ref=2457000.0, diagnostic_plot=False):
    """Scales different tess sectors by a constant to make them match in amplitude.
    times are in TESS bjd by default, but a different bjd_ref can be given to use
    a different time reference point.
    """
    # the 0.5 offset comes from testing, and the fact that no exact JD were found (just calendar days)
    jd_sectors = np.loadtxt('tess_sectors.dat', usecols=(2, 3)) - bjd_ref
    signal_copy = np.copy(signal)
    mask_sect = [(times > sect[0]) & (times < sect[1]) for sect in jd_sectors]
    # determine the range of the signal
    low = np.zeros([len(jd_sectors)])
    high = np.zeros([len(jd_sectors)])
    averages = np.zeros([len(jd_sectors)])
    threshold = np.zeros([len(jd_sectors)])
    for i, mask in enumerate(mask_sect):
        if np.any(mask):
            masked_s = signal[mask]
            averages[i] = np.average(masked_s)
            low[i] = np.average(masked_s[masked_s < averages[i]])
            # low[i] = np.median(masked_s[masked_s < low[i]])
            # low[i] = np.median(masked_s[masked_s < low[i]])
            high[i] = np.average(masked_s[masked_s > averages[i]])
            # high[i] = np.median(masked_s[masked_s > high[i]])
            # high[i] = np.median(masked_s[masked_s > high[i]])
            threshold[i] = np.average(masked_s[masked_s > high[i]])
            threshold[i] = np.average(masked_s[masked_s > threshold[i]])
    
    difference = high - low
    if np.any(difference != 0):
        min_diff = np.min(difference[difference != 0])
    else:
        min_diff = 0
    threshold = 2 * threshold - averages  # to remove spikes (from e.g. momentum dumps)
    thr_mask = np.ones([len(times)], dtype=bool)
    # adjust the signal so that it has a more uniform range
    for i, mask in enumerate(mask_sect):
        if np.any(mask):
            signal_copy[mask] = (signal[mask] - averages[i]) / difference[i] * min_diff + averages[i]
            thr_mask[mask] = (signal[mask] < threshold[i])
    if diagnostic_plot:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[14, 10])
        ax[0].plot(times, signal)
        ax[1].plot(times, signal_copy)
        for i, sect in enumerate(jd_sectors):
            if np.any(mask_sect[i]):
                ax[0].plot([sect[0], sect[1]], [averages[i], averages[i]], c='tab:orange')
                ax[0].plot([sect[0], sect[1]], [low[i], low[i]], c='tab:red')
                ax[0].plot([sect[0], sect[1]], [high[i], high[i]], c='tab:red')
                ax[0].plot([sect[0], sect[1]], [threshold[i], threshold[i]], c='tab:purple')
                avg = np.average(signal_copy[mask_sect[i]])
                l = np.average(signal_copy[mask_sect[i]][signal_copy[mask_sect[i]] < avg])
                h = np.average(signal_copy[mask_sect[i]][signal_copy[mask_sect[i]] > avg])
                ax[1].plot([sect[0], sect[1]], [avg, avg], c='tab:orange')
                ax[1].plot([sect[0], sect[1]], [l, l], c='tab:red')
                ax[1].plot([sect[0], sect[1]], [h, h], c='tab:red')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.show()
    return signal_copy, thr_mask


@nb.jit(nopython=True)
def fold_time_series(times, period, zero):
    """Fold the given time series over the orbital period to go to function of phase.
    Returns phase array for all timestamps using the provided reference zero point.
    Returned phases are between -0.5 and 0.5
    """
    phases = ((times - zero) / period + 0.5) % 1 - 0.5
    return phases


@nb.jit(nopython=True)
def mark_gaps(a):
    """Mark the two points at either side of gaps in a somewhat-uniformly separated
    series of monotonically ascending numbers (e.g. timestamps).
    Returns a boolean array and the gap widths in units of the smallest step size.
    """
    diff = a[1:] - a[:-1]  # np.diff(a)
    min_d = np.min(diff)
    gap_width = diff / min_d
    gaps = (gap_width > 4)  # gaps that are at least 4 times the minimum time step
    gap_width[np.invert(gaps)] = 1  # set non-gaps to width of 1
    gaps = np.append(gaps, [False])  # add a point back to match length with a
    gaps[1:] = gaps[1:] | gaps[:-1]  # include both points on either side of the gap
    gap_width = np.floor(gap_width[gaps[:-1]]).astype(np.int_)
    if gaps[-1]:
        gap_width = np.append(gap_width, [1])  # need to add an extra item to gap_width
    return gaps, gap_width


def multiple_gaps(gaps):
    """Mark the points that are on both sides next to a gap."""
    multi_gap = np.zeros([len(gaps)], dtype=bool)
    multi_gap[gaps] = True
    multi_gap[:-1] &= gaps[1:]
    multi_gap[1:] &= gaps[:-1]
    return multi_gap


@nb.jit(nopython=True)
def repeat_points_internals(t, n):
    """Makes an array of the number of repetitions to be made in an array (a) before diff
    or convolve is used on it, taking into account gaps in the data.
    It also provides a mask that can remove exactly all the repeated points afterward.
    To be used in conjunction with numpy.repeat().
    Make sure the time-points are somewhat consistently spaced.
    
    example:
    n_repeats, rep_mask = repeat_points_internals(times, signal, n)
    repeated_signal = np.repeat(signal, n_repeats)
    original_signal = repeated_signal[rep_mask]
    np.all(signal == original_signal)
    """
    if (n < 2):
        # no repetitions made
        repetitions = np.ones(len(t), dtype=np.int_)
        repetition_mask = np.ones(len(t), dtype=np.bool_)
    else:
        # get the gap positions and start- and endpoints
        gaps, widths = mark_gaps(t)
        gap_start = (widths != 1)  # the gap points at the start of a gap
        gap_p1 = np.copy(gaps)
        gap_p1[gaps] = np.invert(gap_start)
        gap_p2 = np.copy(gaps)
        gap_p2[gaps] = gap_start
        # prepare some measurements of the width
        widths[widths == 1] = 0
        widths = np.ceil(widths / 2).astype(np.int_)
        max_repeats = np.copy(widths)
        max_repeats[1:] += max_repeats[:-1]
        widths = np.where(widths < n, widths, n)
        max_repeats = np.where(max_repeats < n, max_repeats, n)
        # the number of repetitions is at least 1 and max n
        repetitions = np.ones(len(t), dtype=np.int_)
        repetitions[gaps] = max_repeats
        # start the repetition mask by repeating the (inverted) gap mask
        repetition_mask = np.repeat(np.invert(gaps), repetitions)
        # mark where the positions of the gap edges ('gaps') are in the repetition mask
        new_positions = np.cumsum(repetitions) - 1
        gap_edges = np.zeros(len(repetition_mask), dtype=np.bool_)
        gap_edges[new_positions[gap_p1]] = True
        gap_edges[new_positions[gap_p2] - (widths[gap_start] - 1)] = True
        # remove the points that where originally part of 'a' from the repetition_mask:
        repetition_mask |= gap_edges
        # finally, repeat the start and end of the array as well
        repetitions[0] += (n - 1)
        repetitions[-1] += (n - 1)
        repeat_ends = np.zeros(n - 1, dtype=np.bool_)
        repetition_mask = np.concatenate((repeat_ends, repetition_mask, repeat_ends))
    return repetitions, repetition_mask


@nb.jit(nopython=True)
def smooth(a, n, mask=None):
    """Similar in function to numpy.convolve, but uses a flat kernel (average).
    Can also apply a mask to the output arrays, for if they had repeats in them.
    """
    # kernel = np.full(n, 1 / n)
    # a_smooth = np.convolve(a, kernel, 'same')  # keyword 'same' not supported by numba
    # reduce = len(a_smooth) - max(len(a), n)
    # left = reduce // 2
    # a_smooth = a_smooth[left:-(reduce - left)]
    # below method jitted is about as fast, but at least it it then jitted
    # since the np.convolve is only slowed down by jitting!
    n_2 = n // 2
    n_3 = n % 2
    sum = 0
    a_smooth = np.zeros(len(a))
    for i in range(0, n):
        sum = sum + a[i]
        a_smooth[i] = sum / (i + 1)
    for i in range(n, len(a)):
        sum = sum - a[i - n] + a[i]
        a_smooth[i] = sum / n
    # slide the result backward by half n, and calculate the last few points
    if (n_2 > 1) | (n_3 > 0):
        a_smooth[:-n_2 - n_3 + 1] = a_smooth[n_2 + n_3 - 1:]
    for i in range(-n_2 - n_3 + 1, 0):
        sum = sum - a[i - n_2 - 1]
        a_smooth[i] = sum / (-i + n_2)
    
    if mask is not None:
        a_smooth = a_smooth[mask]
    return a_smooth


@nb.jit(nopython=True)
def smooth_diff(a, n, mask=None):
    """Similar in function to numpy.diff, but also first smooths the input array
    by averaging over n consecutive points.
    Can also apply a mask to the output arrays, for if they had repeats in them.
    Also returns the smoothed a.
    See also: smooth, smooth_derivative
    """
    a_smooth = smooth(a, n, mask=None)
    diff = a_smooth[1:] - a_smooth[:-1]  # np.diff(a_smooth)
    if mask is not None:
        diff = diff[mask[:-1]]
        a_smooth = a_smooth[mask]
    return diff, a_smooth


@nb.jit(nopython=True)
def smooth_derivative(a, dt, n, mask=None):
    """Similar in function to numpy.diff, but also first smooths the input array
    by averaging over n consecutive points and divides by the time-diff, so it
    becomes an actual derivative.
    Can also apply a mask to the output arrays, for if they had repeats in them.
    Also returns the smoothed a.
    See also: smooth, smooth_diff
    """
    diff, a_smooth = smooth_diff(a, n, mask=mask)
    d_dt = diff / dt
    return d_dt, a_smooth


@nb.jit(nopython=True)
def prepare_derivatives(times, signal, n_kernel):
    """Calculate various derivatives of the light curve for the purpose of eclipse finding.
    Retruns all the raw and smooth arrays in vertically stacked groups
    (signal_s, r_derivs, s_derivs)
    [s=smoothed, r=raw]
    """
    diff_t = np.diff(np.append(times, 2 * times[-1] - times[-2]))
    # get the repetition array and the repetition mask
    n_repeats, rep_mask = repeat_points_internals(times, n_kernel)
    # array versions: e=extended, s=smoothed
    signal_e = np.repeat(signal, n_repeats)
    deriv_1, signal_s = smooth_derivative(signal_e, diff_t, n_kernel, rep_mask)
    deriv_1e = np.repeat(deriv_1, n_repeats)
    deriv_2, deriv_1s = smooth_derivative(deriv_1e, diff_t, n_kernel, rep_mask)
    deriv_2e = np.repeat(deriv_2, n_repeats)
    deriv_3, deriv_2s = smooth_derivative(deriv_2e, diff_t, n_kernel, rep_mask)
    deriv_3e = np.repeat(deriv_3, n_repeats)
    deriv_3s = smooth(deriv_3e, n_kernel, rep_mask)
    deriv_13 = - deriv_1s * deriv_3s  # invert the sign to make peaks positive
    deriv_13e = np.repeat(deriv_13, n_repeats)
    deriv_13s = smooth(deriv_13e, n_kernel, rep_mask)
    # return the raw derivs, the smooth derivs and smooth signal
    r_derivs = np.vstack((deriv_1, deriv_2, deriv_3, deriv_13))
    s_derivs = np.vstack((deriv_1s, deriv_2s, deriv_3s, deriv_13s))
    return signal_s, r_derivs, s_derivs


def find_best_n_3(times, signal, min_n=2, max_n=50, dev_limit=0.1, diagnostic_plot=False):
    """Confirms whether eclipses are found in the light curve or not, and
    serves to find the best number of points for smoothing the signal
    in the further analysis (n_kernel).
    This is a brute force routine, so might be suboptimal to run every time.
    On the bright side, similar data sets will have similar optimal n_kernel
    (mainly driven by the amount of points covering eclipse in/egress).
    """
    dev_limit = 1
    # todo: need better function... some just need a bit more
    if (min_n < 2):
        print('min_n = 2 is really the minimum.')
        min_n = 2
    n_range = np.arange(min_n, max_n + max_n % 2)
    roughness = np.zeros([len(n_range)])
    deviation = np.zeros([len(n_range)])
    # normalise the signal
    norm_signal = (signal - np.average(signal))
    norm_signal /= np.median(np.abs(norm_signal))
    # non-smooth derivative
    n_repeats, rep_mask = repeat_points_internals(times, 2)
    signal_e = np.repeat(signal, n_repeats)
    diff_1 = np.diff(signal_e)[rep_mask[:-1]]
    diff_t = np.diff(np.append(times, 2 * times[-1] - times[-2]))
    deriv_1 = diff_1 / diff_t
    med_deriv = np.median(np.abs(deriv_1))
    norm_deriv_1 = (deriv_1 - np.average(deriv_1))
    norm_deriv_1 /= np.std(norm_deriv_1)
    n_repeats, rep_mask = repeat_points_internals(times, 2)
    norm_deriv_1e = np.repeat(norm_deriv_1, n_repeats)
    diff_2 = np.diff(norm_deriv_1e)[rep_mask[:-1]]
    roughness_0 = np.mean(np.abs(diff_2)**2 / 4)
    # go through values of n to get the best one
    for i, n in enumerate(n_range):
        signal_s, r_derivs, s_derivs = prepare_derivatives(times, norm_signal, n)
        deviation[i] = np.mean((norm_signal - signal_s)**2)
        norm_deriv_1 = (r_derivs[0] - np.average(r_derivs[0]))
        norm_deriv_1 /= np.std(norm_deriv_1)
        n_repeats, rep_mask = repeat_points_internals(times, 2)
        norm_deriv_1e = np.repeat(norm_deriv_1, n_repeats)
        diff_2 = np.diff(norm_deriv_1e)[rep_mask[:-1]]
        roughness[i] = np.mean(np.abs(diff_2)**2 / 4)
        # signal_s_2, r_derivs_2, s_derivs_2 = prepare_derivatives(times, norm_signal, n + 1)
        # roughness[i] = np.std(signal_s - signal_s_2) / (med_dt * 24 * 2)
    # smooth the deviation (even numbers have an unfair disadvantage)
    deviation[0] = deviation[1]
    deviation[2::2] = (deviation[1:-1:2] + deviation[3::2]) / 2
    deviation /= np.min(deviation)
    roughness /= roughness_0
    optimize = roughness * deviation
    # todo: maybe take only values where deviation < 0.2
    # deviation = deviation - (deviation[0] - 1)
    check_1 = (deviation < dev_limit)
    # take points into account up to the maximum noise if this is not at index 0
    # if np.any(check_1):
    #     argmax = np.argmax(optimize[check_1])
    #     if (argmax == 0):
    #         argmin = np.argmin(optimize[check_1])
    #     else:
    #         argmin = np.argmin(optimize[:argmax + 1])
    # else:
    #     argmin = 0
    # best_n = n_range[argmin]
    check_1 = (deviation < 1.5)
    check_2 = (roughness < 0.35)
    if np.any(check_1 & check_2):
        # first instance where it goes below roughness recuirement
        best_n = n_range[check_1 & check_2][0]
    elif np.any(check_1):
        # last instance where it is below the deviation limit
        best_n = n_range[check_2][-1]
    else:
        best_n = 2
    
    if diagnostic_plot:
        fig, ax = plt.subplots(figsize=[14, 10])
        if not np.all(check_1):
            ax.plot(n_range[[0, -1]], [dev_limit, dev_limit])
        ax.plot(n_range, deviation, label='deviation')
        ax.plot(n_range, roughness, label='roughness')
        ax.plot(n_range, optimize, label='optimize')
        ax.set_xlabel('n_kernel')
        ax.set_ylabel('statistic')
        plt.tight_layout()
        plt.legend(title=f'n={best_n}')
        plt.show()
    return best_n


def find_best_n(times, signal, min_n=2, max_n=50, dev_limit=1.8, diagnostic_plot=False):
    """Confirms whether eclipses are found in the light curve or not, and
    serves to find the best number of points for smoothing the signal
    in the further analysis (n_kernel).
    This is a brute force routine, so might be suboptimal to run every time.
    On the bright side, similar data sets will have similar optimal n_kernel
    (mainly driven by the amount of points covering eclipse in/egress).
    """
    dev_limit = 2.5
    # todo: need better function... some just need a bit more
    if (min_n < 2):
        print('min_n = 2 is really the minimum.')
        min_n = 2
    n_range = np.arange(min_n, max_n + max_n % 2)
    optimize = np.zeros([len(n_range)])
    deviation = np.zeros([len(n_range)])
    sine_like = np.zeros([len(n_range)], dtype=bool)
    # normalise the signal
    norm_signal = (signal - np.average(signal))
    norm_signal /= np.median(np.abs(norm_signal))
    # non-smooth derivative
    n_repeats, rep_mask = repeat_points_internals(times, 2)
    signal_e = np.repeat(signal, n_repeats)
    diff_1 = np.diff(signal_e)[rep_mask[:-1]]
    diff_t = np.diff(np.append(times, 2 * times[-1] - times[-2]))
    deriv_1 = diff_1 / diff_t
    norm_deriv_1 = (deriv_1 - np.average(deriv_1))
    norm_deriv_1 /= np.std(norm_deriv_1)
    # go through values of n to get the best one
    for i, n in enumerate(n_range):
        signal_s, r_derivs, s_derivs = prepare_derivatives(times, norm_signal, n)
        ecl_indices, added_snr, flags, sine_like[i] = find_eclipses(times, signal, n)
        if (len(added_snr) > 0):
            m_full = np.char.startswith(flags, 'f')  # mask of the full eclipses
            m_left = np.char.startswith(flags, 'lh')
            m_right = np.char.startswith(flags, 'rh')
            l_o = ecl_indices[:, 0]  # left outside
            l_i = ecl_indices[:, 1]  # left inside
            r_i = ecl_indices[:, -2]  # right inside
            r_o = ecl_indices[:, -1]  # right outside
            ecl_mid = np.zeros([len(flags)])
            ecl_mid[m_full] = (times[l_o[m_full]] + times[r_o[m_full]] + times[l_i[m_full]] + times[r_i[m_full]]) / 4
            # take the inner points as next best estimate for half eclipses
            ecl_mid[m_left] = times[l_i[m_left]]
            ecl_mid[m_right] = times[r_i[m_right]]
            if (len(ecl_indices[m_full]) == 0):
                # no full eclipses
                ecl_mask = mask_eclipses(times, ecl_indices)
                height_right = signal_s[ecl_indices[:, 0]] - signal_s[ecl_indices[:, 1]]
                height_left = signal_s[ecl_indices[:, -1]] - signal_s[ecl_indices[:, -2]]
                width_right = times[ecl_indices[:, 1]] - times[ecl_indices[:, 0]]
                width_left = times[ecl_indices[:, -1]] - times[ecl_indices[:, -2]]
                # either right or left is always going to be zero now (only half eclipses)
                slope = (height_right + height_left) / (width_right + width_left)
            else:
                ecl_mask = mask_eclipses(times, ecl_indices[m_full])
                height_right = signal_s[ecl_indices[m_full, 0]] - signal_s[ecl_indices[m_full, 1]]
                height_left = signal_s[ecl_indices[m_full, -1]] - signal_s[ecl_indices[m_full, -2]]
                width_right = times[ecl_indices[m_full, 1]] - times[ecl_indices[m_full, 0]]
                width_left = times[ecl_indices[m_full, -1]] - times[ecl_indices[m_full, -2]]
                slope_right = height_right / width_right
                slope_left = height_left / width_left
                slope = (slope_right + slope_left) / 2
            norm_factor = np.median(np.abs(r_derivs[0][ecl_mask]))
            optimize[i] = np.max(slope) / norm_factor
        else:
            optimize[i] = 0
        deviation[i] = np.mean((norm_signal - signal_s)**2)
    # smooth the deviation (even numbers have an unfair disadvantage)
    deviation[0] = deviation[1]
    deviation[2::2] = (deviation[1:-1:2] + deviation[3::2]) / 2
    deviation /= np.min(deviation)
    # todo: maybe take only values where deviation < 0.2
    # take points into account up to the maximum noise if this is not at index 0
    limit_1 = (deviation < dev_limit)
    limit_2 = (deviation < 100)  # could be set higher still if needed
    if (len(optimize[sine_like & limit_2]) > 2):
        # best_n = n_range[sine_like & limit_2][0]
        optimize *= (1 + sine_like)
        best_n = n_range[np.argmax(optimize)]
    elif np.any(limit_1):
        best_n = n_range[np.argmax(optimize[limit_1])]
    else:
        best_n = 2
    
    if diagnostic_plot:
        fig, ax = plt.subplots(figsize=[14, 10])
        ax.plot(n_range[[0, -1]], [dev_limit, dev_limit])
        ax.plot(n_range, deviation, label='deviation')
        ax.plot(n_range, optimize, label='optimize')
        ax.plot(n_range, sine_like, label='sine_like')
        ax.set_xlabel('n_kernel')
        ax.set_ylabel('statistic')
        plt.tight_layout()
        plt.legend(title=f'n={best_n}')
        plt.show()
    return best_n


def find_best_n_old(times, signal, min_n=2, max_n=50, dev_limit=1.8, diagnostic_plot=False):
    """Confirms whether eclipses are found in the light curve or not, and
    serves to find the best number of points for smoothing the signal
    in the further analysis (n_kernel).
    This is a brute force routine, so might be suboptimal to run every time.
    On the bright side, similar data sets will have similar optimal n_kernel
    (mainly driven by the amount of points covering eclipse in/egress).
    """
    # todo: need better function... some just need a bit more
    if (min_n < 2):
        print('min_n = 2 is really the minimum.')
        min_n = 2
    n_range = np.arange(min_n, max_n)
    noise = np.zeros([len(n_range)], dtype=float)
    deviation = np.zeros([len(n_range)], dtype=float)
    # get a first signal error estimate
    n_repeats, rep_mask = repeat_points_internals(times, 2)
    signal_e = np.repeat(signal, n_repeats)
    diff = np.diff(signal_e)[rep_mask[:-1]]
    err_est = np.median(np.abs(diff))
    # go through values of n to get the best one
    for i, n in enumerate(n_range):
        diff_1, signal_s = smooth_diff(signal_e, n, rep_mask)
        diff_1e = np.repeat(diff_1, n_repeats)
        diff_2, diff_1s = smooth_diff(diff_1e, n, rep_mask)
        noise[i] = np.median(np.abs(diff_1s)) / err_est
        deviation[i] = np.mean(((signal - signal_s) / err_est)**2)
    # set the initial deviation to 1 and check for high values
    deviation = deviation - (deviation[0] - 1)
    check_1 = (deviation < dev_limit)
    # take points into account up to the maximum noise if this is not at index 0
    if np.any(check_1):
        argmax = np.argmax(noise[check_1])
        if (argmax == 0):
            argmin = np.argmin(noise[check_1])
        else:
            argmin = np.argmin(noise[:argmax + 1])
    else:
        argmin = 0
    best_n = n_range[argmin]
    
    if diagnostic_plot:
        fig, ax = plt.subplots(figsize=[14, 10])
        ax.plot(n_range, noise, label='noise')
        ax.plot(n_range, deviation, label='deviation')
        ax.set_xlabel('n_kernel')
        ax.set_ylabel('statistic')
        plt.tight_layout()
        plt.legend(title=f'n={best_n}')
        plt.show()
    return best_n


@nb.jit(nopython=True)
def curve_walker(signal, peaks, slope_sign, no_gaps, mode='up', look_ahead=1):
    """Walk up or down a slope to approach zero or to reach an extremum.
    'peaks' are the starting points, 'signal' is the slope to walk
    mode = 'up': walk in the slope sign direction to reach a maximum
    mode = 'down': walk against the slope sign direction to reach a minimum
    mode = 'up_to_zero'/'down_to_zero': same as above, but approaching zero
        as closely as possible without changing direction.
    If look_ahead is True, two points ahead will be checked for the condition
        as well. Can avoid local minima, but can also jump too far.
    """
    if 'down' in mode:
        slope_sign = -slope_sign
    max_i = len(signal) - 1
    
    def check_edges(indices):
        return (indices > 0) & (indices < max_i)
    
    def check_condition(prev_s, cur_s):
        if 'up' in mode:
            condition = (prev_s < cur_s)
        elif 'down' in mode:
            condition = (prev_s > cur_s)
        else:
            condition = np.zeros(len(cur_s), dtype=np.bool_)
        if 'zero' in mode:
            condition &= np.abs(prev_s) > np.abs(cur_s)
        return condition
    
    # start at the peaks
    prev_i = peaks
    prev_s = signal[prev_i]
    # step in the desired direction (checking the edges of the array)
    check_cur_edges = check_edges(prev_i + slope_sign)
    cur_i = prev_i + slope_sign * check_cur_edges
    cur_s = signal[cur_i]
    # check whether the next point might be closer to zero or lower/higher
    next_points = np.zeros(len(cur_i), dtype=np.bool_)
    if (look_ahead > 1):
        # check additional points ahead
        for i in range(1, look_ahead):
            next_points |= check_condition(prev_s, signal[cur_i + i * slope_sign * check_edges(cur_i + i * slope_sign)])
    # check that we fulfill the condition (also check next points)
    check_cur_slope = check_condition(prev_s, cur_s) | next_points # | next_point | next_point_2
    # additionally, check that we don't cross gaps
    check_gaps = no_gaps[cur_i]
    # combine the checks for the current indices
    check = (check_cur_slope & check_gaps & check_cur_edges)
    # define the indices to be optimized
    cur_i = prev_i + slope_sign * check
    while np.any(check):
        prev_i = cur_i
        prev_s = signal[prev_i]
        # step in the desired direction (checking the edges of the array)
        check_cur_edges = check_edges(prev_i + slope_sign)
        cur_i = prev_i + slope_sign * check_cur_edges
        cur_s = signal[cur_i]
        # check whether the next two points might be lower (and check additional points ahead)
        next_points = np.zeros(len(cur_i), dtype=np.bool_)
        for i in range(1, look_ahead):
            next_points |= check_condition(prev_s, signal[cur_i + i * slope_sign * check_edges(cur_i + i * slope_sign)])
        # and check that we fulfill the condition
        check_cur_slope = check_condition(prev_s, cur_s) | next_points # | next_point | next_point_2
        # additionally, check that we don't cross gaps
        check_gaps = no_gaps[cur_i]
        check = (check_cur_slope & check_gaps & check_cur_edges)
        # finally, make the actual approved steps
        cur_i = prev_i + slope_sign * check
    return cur_i


@nb.jit(nopython=True)
def group_same_peak(deriv_1s, peaks_13):
    """Determine which groups of peaks fall on the same actual peak in the deriv."""
    same_pks = np.zeros(len(peaks_13) - 1, dtype=np.bool_)
    for i, pk in enumerate(peaks_13[:-1]):
        pk1_h = deriv_1s[pk]
        pk2_h = deriv_1s[peaks_13[i + 1]]
        if ((pk1_h > 0) & (pk2_h > 0)):
            pk_h = min(pk1_h, pk2_h)
            all_above = np.all(deriv_1s[pk:peaks_13[i + 1] + 1] > 0.9 * pk_h)
            same_pks[i] = all_above
        elif ((pk1_h < 0) & (pk2_h < 0)):
            pk_h = max(pk1_h, pk2_h)
            all_below = np.all(deriv_1s[pk:peaks_13[i + 1] + 1] < 0.9 * pk_h)
            same_pks[i] = all_below

    groups = []
    last_i = -1
    for i, same in enumerate(same_pks):
        if ((i >= last_i) & same):
            grp = [i]
            grp += ([j for j in range(len(peaks_13)) if (j > i) & np.all(same_pks[i:j])])
            if (len(grp) > 0):
                groups.append(grp)
                last_i = grp[-1]
            else:
                last_i = i
        else:
            continue
    return groups


def mark_eclipses(times, signal_s, s_derivs, r_derivs, n_kernel):
    """Mark the positions of eclipse in/egress
    See: 'prepare_derivatives' to get the input for this function.
    Returns all peak position arrays in a vertically stacked group,
    the added_snr snr measurements and the slope sign (+ ingress, - egress).
    """
    deriv_1s, deriv_2s, deriv_3s, deriv_13s = s_derivs
    deriv_1r, deriv_2r, deriv_3r, deriv_13r = r_derivs
    dt = np.diff(np.append(times, 2 * times[-1] - times[-2]))
    # find the peaks from combined deriv 1*3
    peaks_13, props = sp.signal.find_peaks(deriv_13s, height=np.max(deriv_13s) / 16 / n_kernel)
    pk_13_widths, wh, ipsl, ipsr = sp.signal.peak_widths(deriv_13s, peaks_13, rel_height=0.5)
    pk_13_widths = np.ceil(pk_13_widths / 2).astype(int)
    # check whether multiple peaks_13 were found on a single deriv_1 peak
    groups = group_same_peak(deriv_1s, peaks_13)
    passed_0 = np.ones([len(peaks_13)], dtype=bool)
    for grp in groups:
        highest = np.argmax(deriv_13s[peaks_13[grp]])
        passed_0[grp] = False
        passed_0[grp[highest]] = True
    peaks_13 = peaks_13[passed_0]
    pk_13_widths = pk_13_widths[passed_0]
    # some useful parameters
    max_i = len(deriv_2s) - 1
    n_peaks = len(peaks_13)
    slope_sign = np.sign(deriv_3s[peaks_13]).astype(int)  # sign of the slope in deriv_2s
    neg_slope = (slope_sign == -1)
    # peaks in 1 and 3 are not exactly in the same spot: find the right spot here
    indices = np.arange(n_peaks)
    med_width = np.median(pk_13_widths).astype(int)
    n_repeats = 2 * med_width + 1
    peaks_range = np.repeat(peaks_13, n_repeats).reshape(n_peaks, n_repeats) + np.arange(-med_width, med_width + 1)
    peaks_range = np.clip(peaks_range, 0, max_i)
    peaks_1 = np.argmax(-slope_sign.reshape(n_peaks, 1) * deriv_1s[peaks_range], axis=1)
    peaks_1 = peaks_range[indices, peaks_1]
    peaks_3 = np.argmax(slope_sign.reshape(n_peaks, 1) * deriv_3s[peaks_range], axis=1)
    peaks_3 = peaks_range[indices, peaks_3]
    # need a mask with True everywhere but at the gap positions
    gaps, gap_widths = mark_gaps(times)
    no_gaps = np.invert(gaps)
    # get the position of eclipse in/egress and positions of eclipse bottom
    # -- walk from each peak position towards the positive peak in deriv_2s
    peaks_2_pos = curve_walker(deriv_2s, peaks_13, slope_sign, no_gaps, mode='up', look_ahead=2)
    # -- walk from each peak position towards the negative peak in deriv_2s
    timestep = np.median(np.diff(times))
    if (timestep < 0.003) & (n_kernel > 4):
        look_ahead = 4
    elif (timestep < 0.007) & (n_kernel > 3):
        look_ahead = 3
    else:
        look_ahead = 2
    peaks_2_neg = curve_walker(deriv_2s, peaks_13, slope_sign, no_gaps, mode='down', look_ahead=look_ahead)
    
    # if peaks_2_neg start converging on the same points, signal might be sine-like
    check_converge = np.zeros([n_peaks], dtype=bool)
    check_converge[:-1] = (np.diff(peaks_2_neg) < 3)
    check_converge[1:] |= check_converge[:-1]
    check_converge &= no_gaps[peaks_2_neg]  # don't count gaps as convergence
    sine_like = False
    if (np.sum(check_converge) > n_peaks / 1.2):
        # most of them converge: assume we need to correct all of them
        peaks_2_neg = curve_walker(deriv_2s, peaks_2_pos, slope_sign, no_gaps, mode='down_to_zero', look_ahead=1)
        # indicate sine-like morphology and multiply the noise by half to compensate for dense peak pattern
        sine_like = True
    elif (np.sum(check_converge) > n_peaks / 2.2):
        # only correct the converging ones
        peaks_2_nn = curve_walker(deriv_2s, peaks_2_pos, slope_sign, no_gaps, mode='down_to_zero', look_ahead=1)
        peaks_2_neg[check_converge] = peaks_2_nn[check_converge]
        
    # in/egress peaks (eclipse edges) and bottoms are adjusted a bit
    peaks_edge = np.clip(peaks_2_neg - slope_sign + (n_kernel % 2) - (n_kernel == 2) * neg_slope, 0, max_i)
    peaks_bot = np.clip(peaks_2_pos + (n_kernel % 2), 0, max_i)
    
    # first check for some simple strong conditions on the eclipses (signal inside must be lower)
    passed_1 = np.ones([n_peaks], dtype=bool)
    passed_1 &= (signal_s[peaks_edge] > signal_s[peaks_bot])
    # the peak in 13 should not be right next to a much higher peak (could be sidelobe)
    left = np.clip(peaks_13 - 2, 0, max_i)
    right = np.clip(peaks_13 + 2, 0, max_i)
    passed_1 &= (deriv_13s[left] < 2 * deriv_13s[peaks_13]) & (deriv_13s[right] < 2 * deriv_13s[peaks_13])
    # cut out those that did not pass
    peaks_1 = peaks_1[passed_1]
    peaks_2_neg = peaks_2_neg[passed_1]
    peaks_2_pos = peaks_2_pos[passed_1]
    peaks_edge = peaks_edge[passed_1]
    peaks_bot = peaks_bot[passed_1]
    peaks_3 = peaks_3[passed_1]
    peaks_13 = peaks_13[passed_1]
    slope_sign = slope_sign[passed_1]
    neg_slope = neg_slope[passed_1]
    if (len(peaks_13) == 0):
        # no peaks left after cuts
        added_snr = np.array([])
    else:
        # do some additional more advanced tests for confidence level
        passed_2 = np.ones([len(peaks_13)], dtype=bool)
        
        # define points away from the peaks and a mask for all peaks
        point_outside = (2 * peaks_2_neg - peaks_2_pos).astype(int)
        point_outside = np.clip(point_outside, 0, max_i)
        point_inside = (2 * peaks_2_pos - peaks_2_neg).astype(int)
        point_inside = np.clip(point_inside, 0, max_i)
        peak_pairs = np.column_stack([point_outside, point_inside])
        peak_pairs[neg_slope] = peak_pairs[neg_slope][:, ::-1]
        mask_peaks = mask_eclipses(signal_s, peak_pairs)
        
        # get the estimates for the noise in signal_s
        noise_0 = np.average(np.abs(deriv_1r[mask_peaks] * dt[mask_peaks])) * (1 - 0.5 * sine_like)
        # signal to noise in signal_s: difference in height in/out of eclipse
        snr_0 = (signal_s[peaks_edge] - signal_s[peaks_bot]) / noise_0
        passed_2 &= (snr_0 > 2)
        # get the estimates for the noise in deriv_1s
        noise_1 = np.average(np.abs(deriv_2r[mask_peaks] * dt[mask_peaks])) * (1 - 0.5 * sine_like)
        # signal to noise in deriv_1s: difference in slope
        value_around = np.min([-slope_sign * deriv_1s[point_outside], -slope_sign * deriv_1s[point_inside]], axis=0)
        snr_1 = (-slope_sign * deriv_1s[peaks_1] - value_around) / noise_1
        passed_2 &= (snr_1 > 2)
        # sanity check on the measured slope difference
        slope_check = np.abs(deriv_1s[peaks_1] - deriv_1s[peaks_2_neg]) / noise_1
        passed_2 &= (snr_1 > slope_check)
        # get the estimates for the noise in deriv_2s
        noise_2 = np.average(np.abs(deriv_2s[mask_peaks])) * (1 - 0.5 * sine_like)
        # signal to noise in deriv_2s: peak to peak difference
        snr_2 = (deriv_2s[peaks_2_pos] - deriv_2s[peaks_2_neg]) / noise_2
        passed_2 &= (snr_2 > 1)
        # get the estimates for the noise in deriv_3s
        noise_3 = np.average(np.abs(deriv_3s[mask_peaks])) * (1 - 0.5 * sine_like)
        # signal to noise in deriv_3s: peak height
        value_around = np.min([slope_sign * deriv_3s[point_outside], slope_sign * deriv_3s[point_inside]], axis=0)
        snr_3 = (slope_sign * deriv_3s[peaks_3] - value_around) / noise_3
        passed_2 &= (snr_3 > 1)
        # get the estimates for the noise in deriv_13s
        noise_13 = np.average(np.abs(deriv_13s[mask_peaks])) * (1 - 0.5 * sine_like)
        # signal to noise in deriv_13s: peak height
        value_around = np.min([deriv_13s[point_outside], deriv_13s[peaks_bot]], axis=0)
        snr_13 = (deriv_13s[peaks_13] - value_around) / noise_13
        passed_2 &= (snr_13 > 2)
        
        # do a final check on the total 'eclipse strength'
        added_snr = (snr_0 + snr_1 + snr_2 + snr_3)
        passed_2 &= (added_snr > 10)
        # select the ones that passed
        peaks_1 = peaks_1[passed_2]
        peaks_2_neg = peaks_2_neg[passed_2]
        peaks_2_pos = peaks_2_pos[passed_2]
        peaks_edge = peaks_edge[passed_2]
        peaks_bot = peaks_bot[passed_2]
        peaks_3 = peaks_3[passed_2]
        peaks_13 = peaks_13[passed_2]
        slope_sign = slope_sign[passed_2]
        added_snr = added_snr[passed_2]
    peaks = np.vstack([peaks_1, peaks_2_neg, peaks_2_pos, peaks_edge, peaks_bot, peaks_3, peaks_13])
    return peaks, added_snr, slope_sign, sine_like


@nb.jit(nopython=True)
def local_extremum(a, start, right=True, maximum=True):
    """Walks left or right in a 1D-array to find a local extremum."""
    max_i = len(a) - 1
    step = right - (not right)
    
    def condition(prev, cur):
        if maximum:
            return (prev <= cur)
        else:
            return (prev >= cur)
    
    i = start
    prev = a[i]
    cur = a[i]
    # now check the condition and walk left or right
    while condition(prev, cur):
        prev = a[i]
        i = i + step
        if (i < 0) | (i > max_i):
            break
        cur = a[i]
    # adjust i to the previous point (the extremum)
    i = i - step
    return i


def match_in_egress(times, signal, signal_s, added_snr, peaks_edge, peaks_bot, slope_sign):
    """Match up the best combinations of ingress and egress to form full eclipses.
    This is done by chopping all peaks up into parts with consecutive sets of
    ingresses and egresses (through slope sign), and then matching the most alike ones.
    """
    # define some recurring variables
    max_i = len(times) - 1
    indices = np.arange(len(added_snr))
    t_peaks = times[peaks_edge]
    neg_slope = (slope_sign == -1)
    pos_slope = (slope_sign == 1)
    # depths and widths of the in/egress points
    depths_single = signal_s[peaks_edge] - signal_s[peaks_bot]
    widths_single = np.abs(times[peaks_bot] - times[peaks_edge])
    # find matching combinations to form full eclipses
    used = np.zeros([len(peaks_edge)], dtype=bool)
    full_ecl = []
    for i in indices:
        if (i == indices[-1]) | (not np.any(neg_slope[i:])):
            # last of the peaks... nothing to pair up to. Or: no negative slopes left after i
            used[i] = True
        else:
            # determine where the group of positive and negative slopes starts and ends
            until_1 = indices[i:][neg_slope[i:]][0]
            if (until_1 == indices[-1]) | (not np.any(pos_slope[until_1:])):
                until_2 = until_1 + 1
            else:
                until_2 = indices[until_1:][pos_slope[until_1:]][0]
        if (not used[i]) & (pos_slope[i]):
            # these need to be indices indicating the position in the original list of eclipses
            ingress = indices[i:until_1]
            egress = indices[until_1:until_2]
            # todo: think about how to exclude large gaps
            if ((until_2 - i) > 1):
                # make all combinations of in/egress
                combs = [[p1, p2] for p1 in ingress for p2 in egress]
                if (len(combs) > 0):
                    # for each set of combinations, take the best one
                    d_add = [2 * abs(added_snr[p1] - added_snr[p2]) / (added_snr[p1] + added_snr[p2])
                             for p1, p2 in combs]
                    depths = [(depths_single[p1], depths_single[p2]) for p1, p2 in combs]
                    d_depth = [2 * abs(d1 - d2) / (d1 + d2) for d1, d2 in depths]
                    widths = [(widths_single[p1], widths_single[p2]) for p1, p2 in combs]
                    d_width = [2 * abs(w1 - w2) / (w1 + w2) for w1, w2 in widths]
                    d_time = [(t_peaks[p2] - t_peaks[p1]) for p1, p2 in combs]
                    d_time = [t - min(d_time) for t in d_time]
                    d_stat = [a + t + d + w for a, t, d, w in zip(d_add, d_time, d_depth, d_width)]
                    full_ecl.append(combs[np.argmin(d_stat)])
            used[i:until_2] = True

    full_ecl = np.array(full_ecl)
    not_used = np.ones([len(indices)], dtype=bool)
    if (len(full_ecl) != 0):
        # take the average width of the full_ecl
        avg_added = (added_snr[full_ecl[:, 0]] + added_snr[full_ecl[:, 1]]) / 2
        full_widths = (times[peaks_edge[full_ecl[:, 1]]] - times[peaks_edge[full_ecl[:, 0]]])
        med_width = np.median(full_widths[avg_added >= np.average(avg_added)])
        # and compare the other full_ecl against it.
        passed = (full_widths > 0.1 * med_width) & (full_widths < 2.5 * med_width)
        # check the average in-eclipse level compared to surrounding
        avg_inside = np.zeros([len(full_ecl)])
        avg_outside = np.zeros([len(full_ecl)])
        std_inside = np.zeros([len(full_ecl)])
        std_outside = np.zeros([len(full_ecl)])
        std_s_outside = np.zeros([len(full_ecl)])
        for i, ecl in enumerate(full_ecl):
            pk1 = peaks_edge[ecl[0]]
            pk2 = peaks_edge[ecl[1]]
            avg_inside[i] = np.average(signal[pk1 + 1:pk2 - 1])
            std_inside[i] = np.std(signal[pk1 + 1:pk2 - 1])
            avg_outside[i] = np.average(signal[np.clip(pk1 - (pk2 - pk1) // 2, 0, max_i):pk1]) / 2
            avg_outside[i] += np.average(signal[pk2:np.clip(pk2 + (pk2 - pk1) // 2, 0, max_i)]) / 2
            std_outside[i] = np.std(signal[np.clip(pk1 - (pk2 - pk1) // 2, 0, max_i):pk1]) / 2
            std_outside[i] += np.std(signal[pk2:np.clip(pk2 + (pk2 - pk1) // 2, 0, max_i)]) / 2
            # also determine std of signal_s
            std_s_outside[i] = np.std(signal_s[np.clip(pk1 - (pk2 - pk1) // 2, 0, max_i):pk1]) / 2
            std_s_outside[i] += np.std(signal_s[pk2:np.clip(pk2 + (pk2 - pk1) // 2, 0, max_i)]) / 2
        std = np.max(np.vstack([std_inside, std_outside]), axis=0)
        passed &= (avg_inside < avg_outside - np.max([std, std_s_outside], axis=0))
        full_ecl = full_ecl[passed]
        # also make an array of bool for which peaks where used
        not_used = np.array([False if i in full_ecl else True for i in indices])
    return full_ecl, not_used


def assemble_eclipses(times, signal, signal_s, peaks, added_snr, slope_sign):
    """Goes through the found peaks to assemble the eclipses in a neat array of indices.
    Eclipses are marked by 4 indices each: ingress top and bottom,
    and egress bottom and top (in that order).
    Returns the array of eclipse indices, the added_snr statistic where it is averaged
    for the full eclipses and an array with flags, meaning:
    f (full), lh (left half) and rh (right half)
    """
    if (len(added_snr) == 0):
        # nothing to assemble
        return np.array([]), added_snr, np.array([])
    # define some recurring variables
    peaks_1, peaks_2_neg, peaks_2_pos, peaks_edge, peaks_bot, peaks_3, peaks_13 = peaks
    indices = np.arange(len(peaks_edge))
    neg_slope = (slope_sign == -1)
    pos_slope = (slope_sign == 1)
    # determine snr-categories by selecting peaks in the histogram
    hist, edges = np.histogram(added_snr, bins=np.floor(np.sqrt(len(added_snr))).astype(int))
    i_max_1 = np.argmax(hist)
    i_min_1_l = local_extremum(hist, i_max_1, right=False, maximum=False)
    i_min_1_r = local_extremum(hist, i_max_1, right=True, maximum=False)
    hist2 = np.copy(hist)
    hist2[i_min_1_l:i_min_1_r + 1] = 0
    if np.any(hist2 != 0):
        i_max_2 = np.argmax(hist2)
        i_min_2_l = local_extremum(hist, i_max_2, right=False, maximum=False)
        i_min_2_r = local_extremum(hist, i_max_2, right=True, maximum=False)
        hist3 = np.copy(hist2)
        hist3[i_min_2_l:i_min_2_r + 1] = 0
    else:
        i_max_2 = -1
        hist3 = np.zeros([len(hist2)])
    if np.any(hist3 != 0):
        i_max_3 = np.argmax(hist3)
    else:
        i_max_3 = -1
    # sort the found peaks (they could have been in any order)
    arr_max_i = np.array([i_max_1, i_max_2, i_max_3])
    index_sorter = np.argsort(arr_max_i)
    i_max_1, i_max_2, i_max_3 = arr_max_i[index_sorter]
    # define two dividing lines for each group
    if np.all(edges[arr_max_i] > 20) & np.all(arr_max_i != -1):
        # we have no distinct 'noise' group: choose divider based on histogram height
        hist_sorter = np.argsort(hist[arr_max_i])
        g1_i = arr_max_i[hist_sorter][-1]
        g2_i = arr_max_i[hist_sorter][-2]
        divider_1 = (edges[g1_i] + edges[g2_i]) / 2
        divider_2 = 0
    elif np.all(edges[arr_max_i[arr_max_i != -1]] > 40) & (np.sum(arr_max_i == -1) == 1):
        # we might not have distinct groups: check hist height
        divider_1 = (edges[i_max_2] + edges[i_max_3]) / 2
        divider_2 = 0
        if (np.min(hist[i_max_2:i_max_3]) > max(hist[i_max_2], hist[i_max_3]) * 2 / 3):
            divider_1 = 0
    elif (i_max_1 == -1) & (i_max_2 == -1):
        # only one group found
        divider_1 = 0
        divider_2 = 0
    elif (i_max_1 == -1):
        # two groups found
        divider_1 = (edges[i_max_2] + edges[i_max_3]) / 2
        divider_2 = 0
    else:
        # three groups found, including a 'noise' group
        divider_1 = (edges[i_max_3] + edges[i_max_2]) / 2
        divider_2 = (edges[i_max_2] + edges[i_max_1]) / 2

    # keep track of which category the eclipses belong to
    cat = np.zeros([indices[-1] + 1], dtype='<U2')
    # split up into a high, middle and low snr group
    group_high = (added_snr >= divider_1)
    group_mid = (added_snr >= divider_2) & (added_snr < divider_1)
    if (np.average(added_snr[group_mid]) > 100):
        # if both groups are high snr, should be fine (and better) to handle them simultaneously
        group_high |= group_mid
    group_low = (added_snr < divider_2)
    # match peaks form the highest group
    full_ecl_gh, unused_gh = match_in_egress(times, signal, signal_s, added_snr[group_high], peaks_edge[group_high],
                                             peaks_bot[group_high], slope_sign[group_high])
    if (len(full_ecl_gh) > 0):
        full_ecl = indices[group_high][full_ecl_gh]
    else:
        full_ecl = np.zeros([0, 2], dtype=int)
    cat[group_high] = 'g1'
    # match peaks form the middle group
    if np.any(group_mid):
        group_mid[group_high] |= unused_gh  # put any non-used peaks in the next group
        full_ecl_gm, unused_gm = match_in_egress(times, signal, signal_s, added_snr[group_mid], peaks_edge[group_mid],
                                                 peaks_bot[group_mid], slope_sign[group_mid])
        if (len(full_ecl_gm) > 0):
            full_ecl = np.vstack([full_ecl, indices[group_mid][full_ecl_gm]])
        cat[group_mid] = 'g2'
        group_low[group_mid] |= unused_gm  # put any non-used peaks in the next group
    else:
        group_low[group_high] |= unused_gh  # put any non-used peaks in the next group
    # match peaks form the lowest group
    if np.any(group_low):
        full_ecl_gl, unused_gl = match_in_egress(times, signal, signal_s, added_snr[group_low], peaks_edge[group_low],
                                                 peaks_bot[group_low], slope_sign[group_low])
        if (len(full_ecl_gl) > 0):
            full_ecl = np.vstack([full_ecl, indices[group_low][full_ecl_gl]])
        cat[group_low] = 'g3'
    
    # check overlapping eclipses
    mean_snr = (added_snr[full_ecl[:, 0]] + added_snr[full_ecl[:, 1]]) / 2
    overlap = np.zeros([len(mean_snr)], dtype=bool)
    i_full_ecl = np.arange(len(full_ecl))
    for i, ecl in enumerate(full_ecl):
        cond1 = ((ecl[0] > full_ecl[:, 0]) & (ecl[0] < full_ecl[:, 1]))
        cond2 = ((ecl[1] > full_ecl[:, 0]) & (ecl[1] < full_ecl[:, 1]))
        if np.any(cond1 | cond2):
            i_overlap = np.append([i], i_full_ecl[cond1 | cond2])
            snr_vals = mean_snr[i_overlap]
            overlap[i_overlap] = (snr_vals != np.max(snr_vals))
    full_ecl = full_ecl[np.invert(overlap)]
    # finally, construct the eclipse indices array
    if (len(full_ecl) != 0):
        ecl_indices = np.zeros([indices[-1] + 1, 4], dtype=int)
        ecl_indices[pos_slope, 0] = peaks_edge[pos_slope]
        ecl_indices[pos_slope, 1] = peaks_bot[pos_slope]
        ecl_indices[pos_slope, 2] = peaks_bot[pos_slope]
        ecl_indices[pos_slope, 3] = peaks_bot[pos_slope]
        ecl_indices[neg_slope, 0] = peaks_bot[neg_slope]
        ecl_indices[neg_slope, 1] = peaks_bot[neg_slope]
        ecl_indices[neg_slope, 2] = peaks_bot[neg_slope]
        ecl_indices[neg_slope, 3] = peaks_edge[neg_slope]
        ecl_indices[full_ecl[:, 0], 2] = ecl_indices[full_ecl[:, 1], 2]
        ecl_indices[full_ecl[:, 0], 3] = ecl_indices[full_ecl[:, 1], 3]
        ecl_indices = np.delete(ecl_indices, full_ecl[:, 1], axis=0)
        flags = np.zeros([indices[-1] + 1], dtype='<U3')
        flags[slope_sign == 1] = 'lh-'
        flags[slope_sign == -1] = 'rh-'
        flags[full_ecl[:, 0]] = 'f-'
        flags = np.char.add(flags, cat)  # append the categories to the flags
        flags = np.delete(flags, full_ecl[:, 1])
        added_snr[full_ecl[:, 0]] = (added_snr[full_ecl[:, 0]] + added_snr[full_ecl[:, 1]]) / 2
        added_snr = np.delete(added_snr, full_ecl[:, 1])
    else:
        ecl_indices = np.array([])
        flags = np.array([])
    # check whether eclipses consist of just one data point
    keep = []
    for ecl in ecl_indices:
        ecl_signal = signal[ecl[0]:ecl[-1] + 1]
        max_signal = np.max(ecl_signal)
        depth = max_signal - np.min(ecl_signal)
        n_below = len(ecl_signal[ecl_signal < max_signal - depth / 2])
        keep.append(n_below > 1)
    ecl_indices = ecl_indices[keep]
    added_snr = added_snr[keep]
    flags = flags[keep]
    return ecl_indices, added_snr, flags


def plot_marker_diagnostics(times, signal, signal_s, s_derivs, peaks, ecl_indices, flags):
    """Plot the signal and derivatives with the eclipse points marked."""
    deriv_1s, deriv_2s, deriv_3s, deriv_13s = s_derivs
    peaks_1, peaks_2_neg, peaks_2_pos, peaks_edge, peaks_bot, peaks_3, peaks_13 = peaks
    plot_height = np.max(signal) + 0.02 * (np.max(signal) - np.min(signal))
    
    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=[14, 10])
    ax[0].plot(times, signal, label='raw')
    ax[0].plot(times, signal_s, label='smoothed')
    ax[0].scatter(times[peaks_1], signal[peaks_1], label='peak marker', c='tab:orange')
    ax[0].scatter(times[peaks_edge], signal[peaks_edge], label='outside', c='tab:red')
    ax[0].scatter(times[peaks_bot], signal[peaks_bot], label='inside', c='tab:green')
    for i, ecl in enumerate(ecl_indices):
        full = (flags[i].startswith('f'))
        colour = 'tab:red' * full + 'tab:purple' * (not full)
        ax[0].plot(times[ecl[[0, -1]]], [plot_height, plot_height], c=colour, marker='|')
    ax[0].plot([], [], c='tab:red', marker='|', label='full eclipses')
    ax[0].plot([], [], c='tab:purple', marker='|', label='eclipse halves')
    ax[1].plot(times, deriv_1s, label='first deriv')
    ax[1].scatter(times[peaks_1], deriv_1s[peaks_1], label='peak marker', c='tab:orange')
    ax[1].scatter(times[peaks_2_neg], deriv_1s[peaks_2_neg], label='outside', c='tab:red')
    ax[1].scatter(times[peaks_2_pos], deriv_1s[peaks_2_pos], label='inside', c='tab:green')
    ax[2].plot(times, deriv_2s, label='second deriv')
    ax[2].scatter(times[peaks_1], deriv_2s[peaks_1], label='peak marker', c='tab:orange')
    ax[2].scatter(times[peaks_2_neg], deriv_2s[peaks_2_neg], label='outside', c='tab:red')
    ax[2].scatter(times[peaks_2_pos], deriv_2s[peaks_2_pos], label='inside', c='tab:green')
    ax[3].plot(times, deriv_3s, label='third deriv')
    ax[3].scatter(times[peaks_3], deriv_3s[peaks_3], label='peak marker', c='tab:orange')
    ax[3].scatter(times[peaks_2_neg], deriv_3s[peaks_2_neg], label='outside', c='tab:red')
    ax[3].scatter(times[peaks_2_pos], deriv_3s[peaks_2_pos], label='inside', c='tab:green')
    ax[4].plot(times, deriv_13s, label='deriv 1*3')
    ax[4].scatter(times[peaks_13], deriv_13s[peaks_13], label='peak marker', c='tab:orange')
    ax[4].scatter(times[peaks_2_neg], deriv_13s[peaks_2_neg], label='outside', c='tab:red')
    ax[4].scatter(times[peaks_2_pos], deriv_13s[peaks_2_pos], label='inside', c='tab:green')
    for i in range(5):
        ax[i].legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    return


def find_eclipses(times, signal, n_kernel, diagnostic_plot=False):
    """Finds the eclipses in a light curve, regardless of other variability.
    n_kernel is the number of points for smoothing.
    """
    # do the derivatives
    signal_s, r_derivs, s_derivs = prepare_derivatives(times, signal, n_kernel)
    # find the likely eclipse in/egresses and put them together
    peaks, added_snr, slope_sign, sine_like = mark_eclipses(times, signal_s, s_derivs, r_derivs, n_kernel)
    ecl_indices, added_snr, flags = assemble_eclipses(times, signal, signal_s, peaks, added_snr, slope_sign)
    
    if diagnostic_plot:
        plot_marker_diagnostics(times, signal, signal_s, s_derivs, peaks, ecl_indices, flags)
    return ecl_indices, added_snr, flags, sine_like


def measure_eclipses(times, signal, ecl_indices, flags):
    """Get the eclipse midpoints, widths and depths.
    Eclipse depths are averaged between the in and egress side for full eclipses. In high
    noise cases, providing the smoothed light curve can give more accurate eclipse depths.
    Eclipse widths for half eclipses are estimated as twice the width of half the eclipse.
    The eclipse midpoints for half eclipses are taken to be the lowest measured point.
    
    A measure for flat-bottom-ness is also given (ratio between measured
    eclipse width and width at the bottom. Be aware that a ratio of zero does
    not mean that it is not a flat-bottomed eclipse per se. On the other hand,
    a non-zero ratio is a strong indication that there is a flat bottom.
    """
    # prepare some arrays
    m_full = np.char.startswith(flags, 'f')  # mask of the full eclipses
    m_left = np.char.startswith(flags, 'lh')
    m_right = np.char.startswith(flags, 'rh')
    l_o = ecl_indices[:, 0]  # left outside
    l_i = ecl_indices[:, 1]  # left inside
    r_i = ecl_indices[:, -2]  # right inside
    r_o = ecl_indices[:, -1]  # right outside
    # calculate the widths
    widths_bottom = times[r_i[m_full]] - times[l_i[m_full]]
    widths = np.zeros([len(flags)])
    widths[m_full] = times[r_o[m_full]] - times[l_o[m_full]]
    widths[m_left] = times[l_i[m_left]] - times[l_o[m_left]]
    widths[m_right] = times[r_o[m_right]] - times[r_i[m_right]]
    # calculate the ratios: a measure for how 'flat-bottomed' it is
    ratios = np.zeros([len(flags)])
    ratios[m_full] = widths_bottom / widths[m_full]
    # calculate the depths
    depths_l = signal[l_o] - signal[l_i]  # can be zero
    depths_r = signal[r_o] - signal[r_i]  # can be zero
    denom = 2 * m_full + 1 * m_left + 1 * m_right  # should give 2's and 1's (only!)
    depths = (depths_l + depths_r) / denom
    # determine the eclipse midpoints, and estimate them for half eclipses
    ecl_mid = np.zeros([len(flags)])
    ecl_mid[m_full] = (times[l_o[m_full]] + times[r_o[m_full]] + times[l_i[m_full]] + times[r_i[m_full]]) / 4
    # take the inner points as next best estimate for half eclipses
    ecl_mid[m_left] = times[l_i[m_left]]
    ecl_mid[m_right] = times[r_i[m_right]]
    return ecl_mid, widths, depths, ratios


@nb.jit(nopython=True)
def construct_range(t_0, period, domain, p_min=0.1):
    """More elaborate arange algorithm.
    t_0: the fixed point in the range, from where the pattern builds outward.
    period: denotes the step size.
    domain: two values (array-like) that give the borders of the pattern.
    p_min: minimum period value; default is 0.1 (day).
    """
    if (period < p_min):
        n_before = np.ceil((domain[0] - t_0) / p_min)
        n_after = np.floor((domain[1] - t_0) / p_min)
    else:
        n_before = np.ceil((domain[0] - t_0) / period)
        n_after = np.floor((domain[1] - t_0) / period)
    n_range = np.arange(n_before, n_after + 1).astype(np.int_)
    points = t_0 + period * n_range
    return points, n_range


def pattern_test_old(ecl_mid, added_snr, widths, time_frame, ecl_0=None, p_max=None, p_step=None):
    """Test for the presence of a regular pattern in a set of eclipse midpoints.
    ecl_mid: measured eclipse positions
    added_snr: measured eclipse significance levels
    widths: eclipse widths
    p_min, p_max, p_step: period range to search and step size.
    The absolute lower limit is 0.001 (assumed to be in days).
    """
    # set the maximum period and reference eclipse, if not given
    n_ecl = len(ecl_mid)
    ecl_i = np.arange(n_ecl)
    if (p_max is None) | (ecl_0 is None):
        snr_ref = max(np.mean(added_snr), 0.5 * np.max(added_snr))
        high_snr = (added_snr > snr_ref)
        if ecl_0 is None:
            ecl_0 = ecl_i[high_snr][0]
        if (p_max is None) & (len(ecl_mid[high_snr]) > 1):
            p_max = 3 * np.median(np.diff(ecl_mid[high_snr]))
        elif p_max is None:
            p_max = 3 * np.median(np.diff(ecl_mid))
    if (p_max == 0):
        p_max = 3 * np.median(np.diff(ecl_mid))
    p_max = max(0.002, p_max)
    t_0 = ecl_mid[ecl_0]
    # set the minimum period if not given
    p_min = 0.95 * np.min(np.abs(t_0 - ecl_mid[ecl_i != ecl_0]))
    if (p_max < p_min):
        p_min = 0.01 * p_max
    p_min = max(0.001, p_min)
    # set the period step if not given
    if p_step is None:
        p_step = 1 / (25 * np.ptp(ecl_mid))
    p_step = min(p_step, 0.01 * p_max)
    # make the period grid and the k-dimensional lookup tree
    periods = np.arange(p_min, p_max, p_step)
    ecl_tree = sp.spatial.cKDTree(ecl_mid.reshape(-1, 1))
    # fill the goodness of fit array
    gof = np.zeros([len(periods)])
    for i, p in enumerate(periods):
        pattern, n_range = construct_range(t_0, p, time_frame)
        if (len(pattern) != 0):
            # get nearest neighbour (distance and index) in ecl_mid for each pattern time
            d_nn, i_nn = ecl_tree.query(pattern.reshape(-1, 1), k=1)
            # the (indices of the) set of data points with an associated pattern period
            ecl_included = np.unique(i_nn)
            # determine which of the nearest neighbours is the closest neighbour
            n_range_0 = n_range - n_range[0]  # range starting at zero
            i_cn = [n_range_0[i_nn == i][np.argmin(d_nn[i_nn == i])] for i in ecl_included]
            # calculate which closest neighbours are less than the ecl width away
            cn_w = (d_nn[i_cn] / widths[ecl_included] < 0.5)
            # calculate the goodness of fit
            gof[i] = len(added_snr[ecl_included][cn_w]) / len(pattern) * np.sum(added_snr[ecl_included][cn_w])
            gof[i] -= np.sum(d_nn[i_cn][cn_w])
        else:
            gof[i] = 0
    return periods, gof, ecl_tree


def extract_pattern_old(ecl_mid, widths, ecl_0, ecl_period, time_frame, ecl_tree):
    """Get the indices of the eclipses matching the pattern."""
    pattern, n_range = construct_range(ecl_mid[ecl_0], ecl_period, time_frame)
    d_nn, i_nn = ecl_tree.query(pattern.reshape(-1, 1), k=1)
    ecl_included = np.unique(i_nn)
    # determine which of the nearest neighbours is the closest neighbour
    n_range_0 = n_range - n_range[0]  # range starting at zero
    i_cn = [n_range_0[i_nn == i][np.argmin(d_nn[i_nn == i])] for i in ecl_included]
    # calculate which closest neighbours are less than the ecl width away and adjust ecl_included
    cn_w = (d_nn[i_cn] / widths[ecl_included] < 0.5)
    return ecl_included[cn_w]


@nb.jit(nopython=True)
def pattern_test(ecl_mid, added_snr, widths, time_frame, ecl_0=None, p_max=None, p_step=None):
    """Test for the presence of a regular pattern in a set of eclipse midpoints.
    ecl_mid: measured eclipse positions
    added_snr: measured eclipse significance levels
    widths: eclipse widths
    p_min, p_max, p_step: period range to search and step size.
    The absolute lower limit is 0.001 (assumed to be in days).
    """
    # set the maximum period and reference eclipse, if not given
    n_ecl = len(ecl_mid)
    ecl_i = np.arange(n_ecl)
    if (p_max is None) | (ecl_0 is None):
        snr_ref = max(np.mean(added_snr), 0.5 * np.max(added_snr))
        high_snr = (added_snr > snr_ref)
        if ecl_0 is None:
            ecl_0 = ecl_i[high_snr][0]
        if (p_max is None) & (len(ecl_mid[high_snr]) > 1):
            p_max = 3 * np.median(np.diff(ecl_mid[high_snr]))
        elif p_max is None:
            p_max = 3 * np.median(np.diff(ecl_mid))
    if (p_max == 0):
        p_max = 3 * np.median(np.diff(ecl_mid))
    p_max = max(0.002, p_max)
    t_0 = ecl_mid[ecl_0]
    # set the minimum period if not given
    p_min = 0.95 * np.min(np.abs(t_0 - ecl_mid[ecl_i != ecl_0]))
    if (p_max < p_min):
        p_min = 0.01 * p_max
    p_min = max(0.001, p_min)
    # set the period step if not given
    if p_step is None:
        p_step = 1 / (25 * np.ptp(ecl_mid))
    p_step = min(p_step, 0.01 * p_max)
    # make the period grid and the k-dimensional lookup tree
    periods = np.arange(p_min, p_max, p_step)
    # fill the goodness of fit array
    gof = np.zeros(len(periods))
    for i, p in enumerate(periods):
        pattern, n_range = construct_range(t_0, p, time_frame)
        if (len(pattern) != 0):
            # get nearest neighbour in pattern for each ecl_mid by looking to the left and right of the sorted position
            i_nn = np.searchsorted(pattern, ecl_mid)
            i_nn[i_nn == len(pattern)] = len(pattern) - 1
            closest = np.abs(pattern[i_nn] - ecl_mid) < np.abs(ecl_mid - pattern[i_nn - 1])
            i_nn = i_nn * closest + (i_nn - 1) * np.invert(closest)
            # get the distances to nearest neighbours
            d_nn = np.abs(pattern[i_nn] - ecl_mid)
            # calculate which closest neighbours are less than the ecl width away
            cn_w = (d_nn / widths < 0.5)
            # calculate the goodness of fit
            gof[i] = len(added_snr[cn_w]) / len(pattern) * np.sum(added_snr[cn_w]) - np.sum(d_nn[cn_w])
        else:
            gof[i] = 0
    return periods, gof


@nb.jit(nopython=True)
def extract_pattern(ecl_mid, widths, ecl_0, ecl_period, time_frame):
    """Get the indices of the eclipses matching the pattern."""
    pattern, n_range = construct_range(ecl_mid[ecl_0], ecl_period, time_frame)
    # get nearest neighbour in pattern for each ecl_mid by looking to the left and right of the sorted position
    i_nn = np.searchsorted(pattern, ecl_mid)
    i_nn[i_nn == len(pattern)] = len(pattern) - 1
    closest = np.abs(pattern[i_nn] - ecl_mid) < np.abs(ecl_mid - pattern[i_nn - 1])
    i_nn = i_nn * closest + (i_nn - 1) * np.invert(closest)
    # get the distances to nearest neighbours
    d_nn = np.abs(pattern[i_nn] - ecl_mid)
    # calculate which closest neighbours are less than the ecl width away and adjust ecl_included
    cn_w = (d_nn / widths < 0.5)
    return np.arange(len(ecl_mid))[cn_w]


def measure_grouping(periods, ecl_mid, added_snr):
    """Measures how closely the phase folded eclipses are grouped in phase space,
    as well as in the added_snr statistic.
    Folds the times of eclipse midpoints by a given set of periods and measures the
    median absolute deviation (MAD) of the phases.
    Returns the phases and the total MADs.
    """
    ldtype = np.float32  # halve the space needed (local-dtype)
    # prepare the period array
    n_periods = len(periods)
    n_ecl = len(ecl_mid)
    periods = periods.reshape(-1, 1)
    added_snr = np.tile(added_snr, n_periods).reshape(n_periods, n_ecl)
    # put the zero point at a specific place between the eclipses
    zero_point = (ecl_mid[0] + periods / 4)
    phases = fold_time_series(ecl_mid, periods, zero=zero_point).astype(ldtype)
    group_1 = (phases <= 0)
    group_2 = (phases > 0)
    n_g1 = np.sum(group_1, axis=-1, dtype=np.int32)
    n_g2 = np.sum(group_2, axis=-1, dtype=np.int32)
    not_empty = (n_g2 != 0)
    n_p_ne = np.sum(not_empty)
    # maximize the number of points that fall on a straight line in phase
    avg_1 = np.average(phases, weights=group_1, axis=-1)
    avg_1 = avg_1.reshape(n_periods, 1)
    on_line = np.sum(np.abs(phases * group_1 - avg_1) < 0.01, axis=-1, dtype=ldtype)
    avg_2 = np.average(phases[not_empty], weights=group_2[not_empty], axis=-1)
    avg_2 = avg_2.reshape(n_p_ne, 1)
    on_line[not_empty] += np.sum(np.abs(phases[not_empty] * group_2[not_empty] - avg_2) < 0.01, axis=-1, dtype=ldtype)
    on_line /= (n_g1 + n_g2)
    on_line += 0.01 * (on_line == 0)
    # calculate deviations in phase
    dev_1 = np.sum(np.abs(phases * group_1 - avg_1), axis=-1, dtype=ldtype)
    phase_dev = dev_1 / n_g1
    dev_2 = np.sum(np.abs(phases[not_empty] * group_2[not_empty] - avg_2), axis=-1, dtype=ldtype)
    phase_dev[not_empty] = (dev_1[not_empty] + dev_2) / (n_g1[not_empty] + n_g2[not_empty])
    phase_dev /= on_line
    # calculate the deviations in the added_snr
    avg_1 = np.average(added_snr, weights=group_1, axis=-1)
    avg_1 = avg_1.reshape(n_periods, 1)
    dev_1 = np.sum(np.abs(added_snr * group_1 - avg_1), axis=-1) / 100
    snr_dev = dev_1 / n_g1
    avg_2 = np.average(added_snr[not_empty], weights=group_2[not_empty], axis=-1)
    avg_2 = avg_2.reshape(n_p_ne, 1)
    dev_2 = np.sum(np.abs(added_snr[not_empty] * group_2[not_empty] - avg_2), axis=-1) / 100
    snr_dev[not_empty] = (dev_1[not_empty] + dev_2) / (n_g1[not_empty] + n_g2[not_empty])
    snr_dev /= on_line
    return phases, phase_dev, snr_dev


def measure_phase_dev(periods, ecl_mid, added_snr):
    """Measures how closely the phase folded eclipses are grouped in phase space,
    as well as in the added_snr statistic.
    Folds the times of eclipse midpoints by a given set of periods and measures the
    median absolute deviation (MAD) of the phases.
    Returns the phases and the total MADs.
    """
    ldtype = np.float32  # halve the space needed (local-dtype)
    # prepare the period array
    n_periods = len(periods)
    n_ecl = len(ecl_mid)
    periods = periods.reshape(-1, 1)
    added_snr = np.tile(added_snr, n_periods).reshape(n_periods, n_ecl)
    # put the zero point at a specific place between the eclipses
    zero_point = ecl_mid[0]
    phases = fold_time_series(ecl_mid, periods, zero=zero_point).astype(ldtype)
    n_ecl = len(ecl_mid)
    # calculate deviations in phase
    avg_1 = np.average(phases, axis=-1)
    avg_1 = avg_1.reshape(n_periods, 1)
    dev_1 = np.sum(np.abs(phases - avg_1), axis=-1, dtype=ldtype)
    phase_dev = dev_1 / n_ecl
    return phases, phase_dev


def test_separation(variable, group_1, group_2, phase=False):
    """Simple test to see whether the variable is split into separate
    distributions or not using the phases. If no variable is given of
    which to measure the separation, the phases are compared instead.
    """
    n_g1 = len(variable[group_1])
    n_g2 = len(variable[group_2])
    if phase:
        # test if the phases are separated by something else than 0.5
        variable[group_1] += 0.5
    
    if (n_g2 < 3) | (n_g1 < 3):
        # no separation if there is nothing to separate,
        # or cannot say anything about distribution of 1 or 2 points
        separate = False
    elif (n_g1 + n_g2 < 8):
        # for very low numbers, use 50 percent difference as criterion
        g1_avg = np.average(variable[group_1])
        g2_avg = np.average(variable[group_2])
        separate = (max(g1_avg, g2_avg) > 1.5 * min(g1_avg, g2_avg))
    else:
        g1_avg = np.average(variable[group_1])
        g2_avg = np.average(variable[group_2])
        g1_std = np.std(variable[group_1])
        g2_std = np.std(variable[group_2])
        std = max(g1_std, g2_std)
        separate = (abs(g1_avg - g2_avg) > std)
    return separate


def determine_primary(group_1, group_2, depths, widths, added_snr):
    n_g1 = np.sum(group_1)
    n_g2 = np.sum(group_2)
    if ((n_g1 != 0) & (n_g2 != 0)):
        if test_separation(depths, group_1, group_2):
            # separation by eclipse depth
            primary_g1 = (np.average(depths[group_1]) > np.average(depths[group_2]))
        elif test_separation(added_snr, group_1, group_2):
            # separation by added_snr
            primary_g1 = (np.average(added_snr[group_1]) > np.average(added_snr[group_2]))
        elif test_separation(widths, group_1, group_2):
            # separation by eclipse width
            primary_g1 = (np.average(widths[group_1]) > np.average(widths[group_2]))
        else:
            # no clear separation or separate_p: take the group with highest average added_snr
            primary_g1 = (np.average(added_snr[group_1]) > np.average(added_snr[group_2]))
    else:
        if ((n_g1 == 0) & (n_g2 != 0)):
            primary_g1 = False  # cannot discern between p and s
        else:
            primary_g1 = True  # cannot discern between p and s
    return primary_g1


def estimate_period(ecl_mid, widths, depths, added_snr, flags):
    """Determines the time of the midpoint of the first primary eclipse (t0)
    and the eclipse (orbital if possible) period. Also returns an array of flags
    with a 'p' for primary and 's' for secondary for each eclipse.
    Flag 't' means either a rejected feature in the light curve with high SNR,
    or a potential tertiary eclipse.
    """
    n_ecl = len(ecl_mid)
    ecl_i = np.arange(n_ecl)
    m_full = np.char.startswith(flags, 'f')
    n_full_ecl = np.sum(m_full)
    # first establish an estimate of the period
    if (n_ecl < 2):
        # no eclipses or a single eclipse... return None
        ecl_period = None
        ecl_included = []
    elif (n_ecl == 2):
        # only two eclipses... only one guess possible
        ecl_period = abs(ecl_mid[1] - ecl_mid[0])
        ecl_included = np.arange(len(ecl_mid))
        if (ecl_period < 0.01):
            # chance that it overlaps
            ecl_period = None
            ecl_included = []
    else:
        # sort the eclipses first
        ecl_sorter = np.argsort(ecl_mid)
        ecl_mid = ecl_mid[ecl_sorter]
        widths = widths[ecl_sorter]
        depths = depths[ecl_sorter]
        added_snr = added_snr[ecl_sorter]
        m_full = m_full[ecl_sorter]
        # define the time domain to search
        time_frame = np.array([np.min(ecl_mid), np.max(ecl_mid)])
        time_padding = 0.05 * (time_frame[1] - time_frame[0]) / len(ecl_mid)
        time_frame[0] -= time_padding
        time_frame[1] += time_padding
        # set the reference zero-point eclipse
        snr_ref = max(np.average(added_snr), 0.5 * np.max(added_snr))
        ecl_i = np.arange(len(ecl_mid))
        high_snr = (added_snr > snr_ref)
        ecl_0 = ecl_i[added_snr >= np.median(added_snr[high_snr])][0]
        if (len(ecl_mid[high_snr]) > 1):
            p_max = 3 * np.median(np.diff(ecl_mid[high_snr]))
            if (p_max < 0.001):
                p_max = 3 * np.median(np.diff(ecl_mid))
        else:
            p_max = 3 * np.median(np.diff(ecl_mid))
        if (p_max > 0.001):
            # determine the best period
            height = depths / np.min(depths)  # normalised for better results
            periods, gof = pattern_test(ecl_mid, height, widths / 2, time_frame, ecl_0=ecl_0, p_max=p_max)
            best = np.argmax(gof)
            p_best = periods[best]
            ecl_period = p_best
            # fig, ax = plt.subplots()
            # ax.plot(periods, gof)
            # get the eclipse indices of those matching the pattern
            ecl_included = extract_pattern(ecl_mid, widths, ecl_0, ecl_period, time_frame)
            # refine the period (using all included eclipses)
            dp = 3 * np.average(np.diff(periods))
            period_range = np.arange(p_best - dp, p_best + dp, dp / 300)
            all_phases, phase_dev = measure_phase_dev(period_range, ecl_mid[ecl_included], added_snr[ecl_included])
            ecl_period = period_range[np.argmin(phase_dev)]
            # ax.plot(period_range, phase_dev)
            # plt.show()
        else:
            ecl_period = None
            ecl_included = []
    # now better determine the primaries and secondaries and other/tertiaries
    if ecl_period is not None:
        # try double the eclipse period to look for secondaries
        t_0 = ecl_mid[ecl_included][0]
        if (len(ecl_included) > 2):
            phases = fold_time_series(ecl_mid[ecl_included], ecl_period * 2, zero=(t_0 + ecl_period / 2))
            # define groups (potentially prim/sec)
            g1_bool = (phases < 0)
            g1 = ecl_included[g1_bool]
            g2_bool = (phases > 0)
            g2 = ecl_included[g2_bool]
            # test separation, first in phase, then in depth, then in added_snr and finally width
            if (n_full_ecl > 3):
                separate_p = test_separation(phases, g1_bool, g2_bool, phase=True)
            else:
                separate_p = False
            separate_d = test_separation(depths, g1, g2)
            separate_s = test_separation(added_snr, g1, g2)
            separate_w = test_separation(widths, g1, g2)
            double_p_sep = separate_p | separate_d | separate_s | separate_w
            # see if doubling the period gives us two significantly different groups
            if double_p_sep:
                ecl_period = 2 * ecl_period
        else:
            double_p_sep = False
        if not double_p_sep:
            # Try finding eclipses in between the eclipses (possibly at a different phase).
            g1 = ecl_included
            g2 = []
            not_included = np.setxor1d(ecl_i, ecl_included)
            if (len(not_included) > 0):
                phases = fold_time_series(ecl_mid, ecl_period, zero=(t_0 + ecl_period / 4))
                g1_avg_phase = np.average(phases[g1])
                hist, edges = np.histogram(phases[not_included], bins=50)
                not_g1 = (edges[:-1] < g1_avg_phase - 0.04) | (edges[1:] > g1_avg_phase + 0.04)
                if np.any(not_g1):
                    hist_max = np.argmax(hist[not_g1])
                    in_bin = (phases >= edges[:-1][not_g1][hist_max]) & (phases <= edges[1:][not_g1][hist_max])
                    avg_phase = np.average(phases[in_bin])
                    g2 = not_included[(np.abs(phases[not_included] - avg_phase) < 0.02)]
        # refine the groups by selecting a small phase delta (giving full eclipses an advantage)
        # beware eclipses that are within the phase delta but less than the period away
        phases = fold_time_series(ecl_mid, ecl_period, zero=(t_0 + ecl_period / 4))
        g1_avg_phase = np.average(phases[g1])
        g1_bool = (np.abs(phases - g1_avg_phase) < 0.02)
        g1_bool[m_full] |= (np.abs(phases[m_full] - g1_avg_phase) < 0.04)
        # check for eclipses less than a period away (we only want to add any missing eclipses)
        for i in g1:
            g1_bool[g1_bool] &= (np.abs(ecl_mid[g1_bool] - ecl_mid[i]) > 0.5 * ecl_period)
            g1_bool[i] = True
        g1 = ecl_i[g1_bool]
        g2_avg_phase = np.average(phases[g2])
        g2_bool = (np.abs(phases - g2_avg_phase) < 0.02)
        g2_bool[m_full] |= (np.abs(phases[m_full] - g2_avg_phase) < 0.04)
        for i in g2:
            g2_bool[g2_bool] &= (np.abs(ecl_mid[g2_bool] - ecl_mid[i]) > 0.5 * ecl_period)
            g2_bool[i] = True
        g2 = ecl_i[g2_bool]
        # check for eclipses that include wide gaps (or are too narrow)
        g1_avg_w = np.average(widths[g1])
        g1 = g1[(widths[g1] > 0.4 * g1_avg_w) & (widths[g1] < 1.9 * g1_avg_w)]
        g2_avg_w = np.average(widths[g2])
        g2 = g2[(widths[g2] > 0.4 * g1_avg_w) & (widths[g2] < 1.9 * g2_avg_w)]
        # determine which group is primary/secondary if possible (from the full eclipses)
        # check if group 2 is too small
        if (len(g2) < max(2, 0.05 * len(g1))):
            # more likely noise
            g2 = []
            primary_g1 = True
        else:
            primary_g1 = determine_primary(g1, g2, depths, widths, added_snr)
        # make the primary/secondary/tertiary flags_pst
        flags_pst = np.zeros([len(ecl_mid)], dtype=str)
        flags_pst[g1] = 'p' * primary_g1 + 's' * (not primary_g1)
        flags_pst[g2] = 'p' * (not primary_g1) + 's' * primary_g1
        flags_pst[np.setxor1d(ecl_i, np.append(g1, g2).astype(int))] = 't'
        # if only two primaries selected, check their likeness
        primaries = (flags_pst == 'p')
        if (len(added_snr[primaries]) == 2):
            prim_ecl_i = ecl_i[primaries]
            if np.any(added_snr[primaries] < added_snr[primaries][::-1] / 2):
                # if they are too different, revoke all prim/sec and the period
                flags_pst[primaries] = 't'
                flags_pst[flags_pst == 's'] = 't'
                ecl_period = None
        # finally, put the t_zero on the first full primary eclipse
        primaries = (flags_pst == 'p')
        full_primaries = m_full & primaries
        if np.any(full_primaries):
            t_zero = ecl_mid[full_primaries][0]
        elif np.any(primaries):
            t_zero = ecl_mid[primaries][0]
        else:
            t_zero = None
            ecl_period = None
    else:
        t_zero = None
        flags_pst = np.full_like(ecl_mid, 't', dtype=str)
    return t_zero, ecl_period, flags_pst


def plot_period_diagnostics(times, signal, signal_s, ecl_indices, ecl_mid, widths, depths, flags, flags_pst, period):
    """Plot the signal, mark primary and secondary eclipses and plot the period."""
    full = np.char.startswith(flags, 'f')
    prim = np.char.startswith(flags_pst, 'p')
    sec = np.char.startswith(flags_pst, 's')
    tert = np.char.startswith(flags_pst, 't')
    if (len(ecl_indices) != 0):
        ecl_mask = mask_eclipses(times, ecl_indices[:, [0, -1]])
        ecl_bottom_mask = mask_eclipses(times, ecl_indices[:, [1, -2]])
    else:
        ecl_mask = np.zeros([len(times)], dtype=bool)
        ecl_bottom_mask = np.zeros([len(times)], dtype=bool)
    if period is not None:
        t_0 = ecl_mid[prim][0]
        period_array = np.arange(t_0, times[0], -period)[::-1]
        period_array = np.append(period_array, np.arange(t_0, times[-1], period))
        phases = fold_time_series(ecl_mid, period, (t_0 + period / 4))
    else:
        phases = np.zeros([len(ecl_mid)])
    s_minmax = (np.max(signal) - np.min(signal))
    height_1 = np.max(signal) + 0.02 * s_minmax
    height_2 = height_1 + 0.02 * s_minmax
    
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[14, 10])
    ax[0].scatter(ecl_mid, phases, c='tab:blue', marker='o', label='eclipse midpoints')
    ax[0].scatter(ecl_mid[full], phases[full], c='tab:red', marker='o', label='full eclipses')
    if np.any(prim):
        prim_avg = np.average(phases[prim])
        ax[0].plot(times[[0, -1]], [prim_avg, prim_avg], c='grey', linestyle='--')
    if np.any(sec):
        sec_avg = np.average(phases[sec])
        ax[0].plot(times[[0, -1]], [sec_avg, sec_avg], c='grey', linestyle='--')
    ax[0].scatter([], [], c='tab:orange', marker='X', label='eclipse widths')
    ax[0].scatter([], [], c='tab:green', marker='P', label='eclipse depths')
    ax[0].set_ylim(-0.55, 0.55)
    w_ax = ax[0].twiny()
    d_ax = ax[0].twiny()
    w_ax.spines['top'].set_position(('axes', 1.15))
    w_ax.scatter(widths, phases, c='tab:orange', marker='X', label='eclipse widths')
    d_ax.scatter(depths, phases, c='tab:green', marker='P', label='eclipse depths')
    w_ax.set_xlabel('eclipse width (units of time)')
    d_ax.set_xlabel('eclipse depth (units of signal)')
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('phase')
    if period is not None:
        ax[0].legend(title=f'period = {period:1.4f}')
    else:
        ax[0].legend()
    ax[1].scatter(times[ecl_mask], signal[ecl_mask])
    ax[1].scatter(times[np.invert(ecl_mask)], signal[np.invert(ecl_mask)], label='eclipses')
    ax[1].scatter(times[np.invert(ecl_bottom_mask)], signal[np.invert(ecl_bottom_mask)], label='eclipse bottoms')
    ax[1].plot(times, signal_s, marker='.', c='tab:brown')
    if period is not None:
        ax[1].plot(period_array, np.full_like(period_array, height_1), c='k', marker='|')
    ax[1].scatter(ecl_mid[prim], np.full_like(ecl_mid[prim], height_2), c='tab:red', marker='^', label='primaries')
    ax[1].scatter(ecl_mid[sec], np.full_like(ecl_mid[sec], height_2), c='tab:purple', marker='s', label='secondaries')
    ax[1].scatter(ecl_mid[tert], np.full_like(ecl_mid[tert], height_2), c='tab:pink', marker='x',
                  label='tertiaries/other')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('signal')
    ax[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    return


def find_ephemeris(times, signal, n_kernel, diagnostic_plot=False):
    """Finds t0 and the eclipse period from a set of eclipses.
    See also: find_eclipses
    """
    # do the derivatives
    signal_s, r_derivs, s_derivs = prepare_derivatives(times, signal, n_kernel)
    # find the likely eclipse in/egresses and put them together
    peaks, added_snr, slope_sign, sine_like = mark_eclipses(times, signal_s, s_derivs, r_derivs, n_kernel)
    ecl_indices, added_snr, flags = assemble_eclipses(times, signal, signal_s, peaks, added_snr, slope_sign)
    # take some measurements
    ecl_mid, widths, depths, ratios = measure_eclipses(times, signal_s, ecl_indices, flags)
    # if possible, find the period and flags for p/s/t
    t_0, period, flags_pst = estimate_period(ecl_mid, widths, depths, added_snr, flags)
    if diagnostic_plot:
        plot_period_diagnostics(times, signal, signal_s, ecl_indices, ecl_mid, widths, depths, flags, flags_pst, period)
    return t_0, period, flags_pst


def found_ratio(times, ecl_mid, prim, sec, period, n_found):
    if period is None:
        n_possible = 1
    elif (period > 0):
        gaps, gap_widths = mark_gaps(times)
        gap_times = np.concatenate([[times[0]], times[gaps], [times[-1]]])
        # primaries
        if np.any(prim):
            t_0 = ecl_mid[prim][0]
            possible_prim = np.arange(t_0, times[0], -period)[::-1]
            possible_prim = np.append(possible_prim, np.arange(t_0, times[-1], period))
            # count only possible eclipses where there is data coverage
            mask = np.zeros([len(possible_prim)], dtype=bool)
            for gt1, gt2 in zip(gap_times[:-1], gap_times[1:]):
                if np.any((times > gt1) & (times < gt2)):
                    # there is coverage between these points
                    mask |= ((possible_prim > gt1) & (possible_prim < gt2))
            possible_prim = possible_prim[mask]
        else:
            possible_prim = []
        # secondaries
        if np.any(sec):
            t_0 = ecl_mid[sec][0]
            possible_sec = np.arange(t_0, times[0], -period)[::-1]
            possible_sec = np.append(possible_sec, np.arange(t_0, times[-1], period))
            # count only possible eclipses where there is data coverage
            mask = np.zeros([len(possible_sec)], dtype=bool)
            for gt1, gt2 in zip(gap_times[:-1], gap_times[1:]):
                if np.any((times > gt1) & (times < gt2)):
                    # there is coverage between these points
                    mask |= ((possible_sec > gt1) & (possible_sec < gt2))
            possible_sec = possible_sec[mask]
        else:
            possible_sec = []
        # combine into one number
        n_possible = len(possible_prim) + len(possible_sec)
    else:
        n_possible = 1
    # use the number of theoretical eclipses to get the ratio
    if (n_possible != 0):
        found_ratio = n_found / n_possible
    else:
        found_ratio = 1
    # transform this ratio a bit for usefulness
    if (found_ratio > 1):
        found_ratio = 1 / found_ratio
    found_ratio = 0.5 * found_ratio + 0.5
    return found_ratio


@nb.jit(nopython=True)
def normalised_slope(times, signal_s, deriv_1r, ecl_indices, ecl_mask, prim_sec, m_full):
    """Calculate the average slope of the eclipses and normalise it by
    the median derivative of the light curve outside the eclipses.
    """
    mask = prim_sec & m_full
    if (len(ecl_indices[mask]) == 0):
        # no full eclipses
        height_right = signal_s[ecl_indices[prim_sec, 0]] - signal_s[ecl_indices[prim_sec, 1]]
        height_left = signal_s[ecl_indices[prim_sec, -1]] - signal_s[ecl_indices[prim_sec, -2]]
        width_right = times[ecl_indices[prim_sec, 1]] - times[ecl_indices[prim_sec, 0]]
        width_left = times[ecl_indices[prim_sec, -1]] - times[ecl_indices[prim_sec, -2]]
        # either right or left is always going to be zero now (only half eclipses)
        slope = (height_right + height_left) / (width_right + width_left)
    else:
        height_right = signal_s[ecl_indices[mask, 0]] - signal_s[ecl_indices[mask, 1]]
        height_left = signal_s[ecl_indices[mask, -1]] - signal_s[ecl_indices[mask, -2]]
        width_right = times[ecl_indices[mask, 1]] - times[ecl_indices[mask, 0]]
        width_left = times[ecl_indices[mask, -1]] - times[ecl_indices[mask, -2]]
        slope_right = height_right / width_right
        slope_left = height_left / width_left
        right_min = (slope_right < slope_left)
        slope = slope_right * right_min + slope_left * (np.invert(right_min))
        # slope = np.min([height_right / width_right, height_left / width_left], axis=0)
    norm_factor = np.median(np.abs(deriv_1r[ecl_mask]))
    return np.mean(slope) / norm_factor


@nb.jit(nopython=True)
def normalised_symmetry(times, signal, ecl_indices):
    if (len(ecl_indices) == 0):
        symmetry = 1
    else:
        height_right = signal[ecl_indices[:, 0]] - signal[ecl_indices[:, 1]]
        height_left = signal[ecl_indices[:, -1]] - signal[ecl_indices[:, -2]]
        width_right = times[ecl_indices[:, 1]] - times[ecl_indices[:, 0]]
        width_left = times[ecl_indices[:, -1]] - times[ecl_indices[:, -2]]
        height = np.abs(height_right - height_left) / ((height_right + height_left) / 2)
        slope = np.abs(height_right / width_right - height_left / width_left)
        slope /= ((height_right / width_right + height_left / width_left) / 2)
        symmetry = np.mean(slope) * np.mean(height) / 0.7
        if (symmetry < 0.5):
            # draw out the part around 0 to get some better differentiation
            symmetry *= 2 * (1 - symmetry)
    symmetry = 0.5 / (0.5 + symmetry)
    return symmetry


@nb.jit(nopython=True)
def normalised_equality(added_snr, depths, widths, prim, sec):
    """Calculate the deviations"""
    n_prim = len(added_snr[prim])
    n_sec = len(added_snr[sec])
    if (n_prim > 1):
        avg_a_p = np.mean(added_snr[prim])
        dev_a_p = np.sum((np.abs(added_snr[prim] - avg_a_p) / avg_a_p)**2)
        avg_d_p = np.mean(depths[prim])
        dev_d_p = np.sum((np.abs(depths[prim] - avg_d_p) / avg_d_p)**2)
        avg_w_p = np.mean(widths[prim])
        dev_w_p = np.sum((np.abs(widths[prim] - avg_w_p) / avg_w_p)**2)
        if (n_sec > 1):
            avg_a_s = np.mean(added_snr[sec])
            dev_a_s = np.sum((np.abs(added_snr[sec] - avg_a_s) / avg_a_s)**2)
            avg_d_s = np.mean(depths[sec])
            dev_d_s = np.sum((np.abs(depths[sec] - avg_d_s) / avg_d_s)**2)
            avg_w_s = np.mean(widths[sec])
            dev_w_s = np.sum((np.abs(widths[sec] - avg_w_s) / avg_w_s)**2)
        else:
            dev_a_s = 0
            dev_d_s = 0
            dev_w_s = 0
            n_sec = 0  # adjust to zero for the final calculation
        dev_a = (dev_a_p + dev_a_s) / (n_prim + n_sec - 1)
        dev_d = (dev_d_p + dev_d_s) / (n_prim + n_sec - 1)
        dev_w = (dev_w_p + dev_w_s) / (n_prim + n_sec - 1)
        equality = dev_a * dev_d * dev_w / 0.1
        if (equality < 0.5):
            # draw out the part around 0 to get some better differentiation
            equality *= 2 * (1 - equality)
    else:
        equality = 1
    equality = 0.5 / (0.5 + equality)
    return equality


def eclipse_confidence(times, signal_s, deriv_1r, period, ecl_indices, ecl_mid, added_snr, widths, depths,
                       flags, flags_pst, return_all=False):
    """Determine a number that expresses the confidence that we have found actual eclipses.
    Below ... is probably a false positive, above ... is quite probably and EB.
    """
    if (len(ecl_mid) != 0):
        primaries = (flags_pst == 'p')
        secondaries = (flags_pst == 's')
        prim_sec = primaries | secondaries
        m_full = np.char.startswith(flags, 'f')
        ecl_mask = mask_eclipses(times, ecl_indices[prim_sec])
        n_found = len(ecl_mid[prim_sec])
        if (n_found != 0):
            if np.any(primaries):
                avg_p = np.average(added_snr[primaries])
            else:
                avg_p = 0
            if np.any(secondaries):
                avg_s = np.average(added_snr[secondaries])
            else:
                avg_s = 0
            # convert the added_snr to a value between 0 and 1
            ratio_0 = np.arctan((avg_p + avg_s) / 40) * 2 / (np.pi)
            # the number of ecl vs number of theoretically visible ones
            ratio_1 = found_ratio(times, ecl_mid, primaries, secondaries, period, n_found)
            # slope of the eclipses - higher is more likely an actual eclipse
            ratio_2 = normalised_slope(times, signal_s, deriv_1r, ecl_indices, ecl_mask, primaries, m_full)
            # if np.any(secondaries):
            #     ratio_2 += normalised_slope(times, signal_s, deriv_1r, ecl_indices, ecl_mask, secondaries, m_full)
            ratio_2 = np.arctan(ratio_2 / 2.5) * 2 / (np.pi)
            # the more unequal the eclipse in/egress are, the lower the confidence
            ratio_3 = normalised_symmetry(times, signal_s, ecl_indices[m_full & prim_sec])
            # if eclipse depth varies a lot - might just be pulsations
            ratio_4 = normalised_equality(added_snr, depths, widths, primaries, secondaries)
            # penalty for not having any full eclipses
            penalty = 1 - 0.5 * (not np.any(m_full & prim_sec))
            confidence = ratio_0 * ratio_2 * np.sqrt(ratio_1**2 + ratio_2**2 + ratio_3**2 + ratio_4**2) / 2
            confidence *= penalty
        else:
            # still have the possibility of a single eclipse (so without period)
            ratio_0 = np.arctan(np.max(added_snr) / 600) * 2 / (np.pi)
            ratio_2 = normalised_slope(times, signal_s, deriv_1r, ecl_indices, ecl_mask, np.invert(prim_sec), m_full)
            ratio_2 = np.arctan(ratio_2 / 2.5) * 2 / (np.pi)
            penalty = 1 - 0.5 * (not np.any(m_full))
            confidence = ratio_0 * ratio_2 * penalty
            ratio_1, ratio_3, ratio_4 = -0.1, -0.1, -0.1
    else:
        # no eclipses identified
        confidence = -1
        ratio_0, ratio_1, ratio_2, ratio_3, ratio_4 = -0.1, -0.1, -0.1, -0.1, -0.1
        penalty = 0
    if return_all:
        return confidence, ratio_0, ratio_1, ratio_2, ratio_3, ratio_4, penalty
    return confidence


def eclipse_stats(flags_pst, widths, depths):
    """Measures the average width and depth for the primary and for the
    secondary eclipses, plus the standard deviations.
    """
    prim = (flags_pst == 'p')
    sec = (flags_pst == 's')
    width_p = [np.average(widths[prim]), np.std(widths[prim])]
    depth_p = [np.average(depths[prim]), np.std(depths[prim])]
    if np.any(sec):
        width_s = [np.average(widths[sec]), np.std(widths[sec])]
        depth_s = [np.average(depths[sec]), np.std(depths[sec])]
    else:
        width_s = [-1, -1]
        depth_s = [-1, -1]
    width_stats = np.column_stack([width_p, width_s])
    depth_stats = np.column_stack([depth_p, depth_s])
    return width_stats, depth_stats


def find_all(times, signal, mode=1, rescale=True, max_n=80, dev_limit=1.8):
    """Find the eclipses, ephemeris and the statistics about the eclipses.
    There are several modes of operation:
    1: Only find and return the ephemeris and confidence level
    2: Find and return the ephemeris, confidence level and individual eclipse
        midpoints, widths, depths and bottom ratios, plus p/s/t flags
    3: Same as 2, but also returns all eclipse indices and the added_snr statistic
    4: Find and return the ephemeris, confidence level and collective stats
        about the eclipse widths and depths (mean and std)
    5: Return everything
    -1: Turn on diagnostic plots and return everything
    """
    # set diagnostic plots on or off
    dp = False
    if (mode == -1):
        dp = True
    # rescale the different (TESS) sectors
    if rescale:
        signal, thr_mask = rescale_tess(times, signal, diagnostic_plot=dp)
        times = times[thr_mask]
        signal = signal[thr_mask]
    # find the best number of smoothing points
    n_kernel = find_best_n(times, signal, max_n=max_n, dev_limit=dev_limit, diagnostic_plot=dp)
    # do the derivatives
    signal_s, r_derivs, s_derivs = prepare_derivatives(times, signal, n_kernel)
    # get the likely eclipse indices from the derivatives
    peaks, added_snr, slope_sign, sine_like = mark_eclipses(times, signal_s, s_derivs, r_derivs, n_kernel)
    ecl_indices, added_snr, flags = assemble_eclipses(times, signal, signal_s, peaks, added_snr, slope_sign)
    if dp:
        plot_marker_diagnostics(times, signal, signal_s, s_derivs, peaks, ecl_indices, flags)
    # check if any where found
    if (len(ecl_indices) != 0):
        # take some measurements and find the period if possible
        ecl_mid, widths, depths, ratios = measure_eclipses(times, signal_s, ecl_indices, flags)
        t_0, period, flags_pst = estimate_period(ecl_mid, widths, depths, added_snr, flags)
        if dp:
            plot_period_diagnostics(times, signal, signal_s, ecl_indices, ecl_mid, widths, depths,
                                    flags, flags_pst, period)
    else:
        ecl_mid, widths, depths, ratios = np.array([[], [], [], []])
        t_0, period, flags_pst = -1, -1, np.array([])
    # check if any primary eclipses were found
    if (len(flags_pst) != 0):
        # determine the confidence level
        conf = eclipse_confidence(times, signal_s, r_derivs[0], period, ecl_indices, ecl_mid,
                                  added_snr, widths, depths, flags, flags_pst)
    else:
        conf = -1
    # check if the w/d statistics need to be calculated
    if (conf != -1) & (mode in [4, 5, -1]):
        # determine the collective characteristics
        width_stats, depth_stats = eclipse_stats(flags_pst, widths, depths)
    else:
        width_stats = np.array([[-1, -1], [-1, -1]])
        depth_stats = np.array([[-1, -1], [-1, -1]])
    # depending on the mode, return (part of) the results
    if (mode == 2):
        return t_0, period, conf, ecl_mid, widths, depths, ratios, flags_pst
    elif (mode == 3):
        return t_0, period, conf, ecl_mid, widths, depths, ratios, flags_pst, ecl_indices, added_snr
    elif (mode == 4):
        return t_0, period, conf, sine_like, width_stats, depth_stats
    elif (mode in [5, -1]):
        return t_0, period, conf, sine_like, n_kernel, width_stats, depth_stats, \
               ecl_mid, widths, depths, ratios, flags, flags_pst, ecl_indices, added_snr
    else:
        # mode == 1 or anything not noted above
        return t_0, period, conf, sine_like, n_kernel


def ephem_test_from_file(file_name):
    times, signal = np.loadtxt(file_name, unpack=True)
    try:
        result = find_all(times, signal, mode=5, rescale=False, max_n=80, dev_limit=1.8)
    except:
        print(file_name)
    return result


def ephem_from_tic(tic, all_files=None):
    tic_files = [file for file in all_files if f'-{tic:016.0f}-' in file]
    times = np.array([])
    signal = np.array([])
    for file in tic_files:
        tess_data = fh.get_data(file, 1)
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
        # result = find_all(times, signal, mode=1, rescale=True, max_n=80, dev_limit=1.8)
        try:
            result = find_all(times, signal, mode=1, rescale=True, max_n=80, dev_limit=1.8)
        except:
            print(tic)
            result = [-1, -1, -1, -1, -1]
    return result[:3]


def conf_from_set(file_name, results_full):
    times, signal = np.loadtxt(file_name, unpack=True)
    t_0, period, conf, sine_like, n_kernel, width_stats, depth_stats, ecl_mid, widths, depths, ratios, flags, flags_pst, ecl_indices, added_snr = results_full
    signal_s, r_derivs, s_derivs = prepare_derivatives(times, signal, n_kernel)
    if period is None:
        period = -1
    confidence = eclipse_confidence(times, signal_s, r_derivs[0], period, ecl_indices, ecl_mid, added_snr, widths, depths,
                              flags, flags_pst, return_all=True)
    return confidence


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    