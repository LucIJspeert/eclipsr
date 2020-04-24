"""This module contains functions to find eclipses, measure their properties and
do various miscellaneous things with them (or on arrays in general).

Code written by: Luc IJspeert
"""

import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt


def cut_eclipses(times, eclipses):
    """Returns a boolean mask covering up the eclipses.
    Give the eclipse times as a series of two points in time.
    See also: mask_eclipses
    """
    mask = np.ones_like(times, dtype=bool)
    for ecl in eclipses:
        mask = mask & ((times < ecl[0]) | (times > ecl[1]))
    return mask


def mask_eclipses(times, eclipses):
    """Returns a boolean mask covering up the eclipses.
    Give the eclipse indices as a series of two indices per eclipse.
    See also: cut_eclipses
    """
    mask = np.ones_like(times, dtype=bool)
    for ecl in eclipses:
        mask[ecl[0]:ecl[-1] + 1] = False  # include the right point in the mask
    return mask


def fold_time_series(times, period, zero):
    """Fold the given time series over the orbital period to go to function of phase.
    Returns phase array for all timestamps using the provided reference zero point.
    Returned phases are between -0.5 and 0.5
    """
    phases = ((times - zero) / period + 0.5) % 1 - 0.5
    return phases


def repeat_points_internals(a, n, mask=None):
    """Makes the array of repetitions and the repetition mask that can remove
    exactly the repeated points afterward. This function can be used in case
    more repeat_points are done on the same array (these stay the same then).
    See also: repeat_points
    """
    # start making the repetition mask by repeating the gaps marked in 'mask'
    if mask is None:
        mask = np.zeros_like(a, dtype=bool)
        mask[[0, -1]] = True
    repetitions = np.ones_like(a, dtype=int)
    repetitions += mask.astype(int) * (n - 1)
    repetition_mask = np.repeat(np.invert(mask), repetitions)
    # mark where the positions of the gap edges ('mask') are in the repetition mask
    new_positions = np.cumsum(repetitions) - 1
    gap_edges = np.zeros_like(repetition_mask, dtype=bool)
    gap_edges[new_positions[mask][::2]] = True
    gap_edges[new_positions[mask][1::2] - (n - 1)] = True
    # remove the points that where originally part of 'a' from the repetition_mask:
    repetition_mask |= gap_edges
    return repetitions, repetition_mask


def repeat_points(a, n, mask=None, repetitions=None, rep_mask=None):
    """Similar in function to numpy.repeat(), except the input now is a boolean mask
    marking the points to be repeated n times in a (minimum is n=2).
    If mask=None, only the first and last points of the array are repeated.
    Also returns a boolean mask to use on the repeated array and get back the original
    series of points.
    If 'repetitions' and 'rep_mask' are provided, these are used instead of calculating them.
    Providing 'mask' is then not necessary.
    """
    if repetitions is None:
        repetitions, repetition_mask = repeat_points_internals(a, n, mask=mask)
    else:
        repetition_mask = rep_mask
    a_ext = np.repeat(a, repetitions)
    return a_ext, repetition_mask


def smooth(a, n, mask=None):
    """Similar in function to numpy.convolve, but uses a flat kernel (average).
    Can also apply a mask to the output arrays, for if they had repeats in them.
    """
    kernel = np.full(n, 1 / n)
    a_smooth = np.convolve(a, kernel, 'same')
    if mask is not None:
        a_smooth = a_smooth[mask]
    return a_smooth


def smooth_diff(a, n, mask=None):
    """Similar in function to numpy.diff, but also first smooths the input array
    by averaging over n consecutive points.
    Can also apply a mask to the output arrays, for if they had repeats in them.
    Also returns the smoothed a.
    See also: smooth, smooth_derivative
    """
    a_smooth = smooth(a, n, mask=None)
    diff = np.diff(a_smooth)
    if mask is not None:
        diff = diff[mask[:-1]]
        a_smooth = a_smooth[mask]
    return diff, a_smooth


def smooth_derivative(a, dt, n, mask=None):
    """Similar in function to numpy.diff, but also first smooths the input array
    by averaging over n consecutive points and divides by the time-diff, so it
    becomes an act.
    Can also apply a mask to the output arrays, for if they had repeats in them.
    Also returns the smoothed a.
    See also: smooth, smooth_diff
    """
    diff, a_smooth = smooth_diff(a, n, mask=mask)
    d_dt = diff/dt
    return d_dt, a_smooth


def mark_gaps(a, mark_ends=True):
    """Mark the two points at either side of gaps in a somewhat-uniformly separated
    series of numbers (e.g. timestamps). Returns a boolean array.
    Start and end of the array are also marked by default.
    """
    diff = np.diff(a)
    min_d = np.min(diff)
    gaps = (diff > 4 * min_d)  # gaps that are at least 4 times the minimum time step
    gaps[1:] |= gaps[:-1]  # include both points on either side of the gap
    if mark_ends:
        gaps[0] = True  # also include the first and last points
        gaps = np.append(gaps, [True])  # include the last point as well
    else:
        gaps = np.append(gaps, [False])
    return gaps


def prepare_derivatives(times, signal, gaps, n_points):
    """Calculate various derivatives of the light curve for the purpose of eclipse finding.
    Retruns all the raw and smooth arrays in stacked groups
    (r_derivs, signal_s, s_derivs, deriv_13s)
    [s=smoothed, r=raw]
    """
    diff_t = np.append(np.diff(times), 2 * times[-1] - times[-2])
    # get the repetition array and the repetition mask
    n_repeats, rep_mask = repeat_points_internals(signal, n_points, mask=gaps)
    # array versions: e=extended, s=smoothed
    signal_e, rep_mask = repeat_points(signal, n_points, repetitions=n_repeats, rep_mask=rep_mask)
    deriv_1, signal_s = smooth_derivative(signal_e, diff_t, n_points, rep_mask)
    deriv_1e, rep_mask = repeat_points(deriv_1, n_points, repetitions=n_repeats, rep_mask=rep_mask)
    deriv_2, deriv_1s = smooth_derivative(deriv_1e, diff_t, n_points, rep_mask)
    deriv_2e, rep_mask = repeat_points(deriv_2, n_points, repetitions=n_repeats, rep_mask=rep_mask)
    deriv_3, deriv_2s = smooth_derivative(deriv_2e, diff_t, n_points, rep_mask)
    deriv_3e, rep_mask = repeat_points(deriv_3, n_points, repetitions=n_repeats, rep_mask=rep_mask)
    # deriv_3s = smooth(deriv_3e, n_points, rep_mask)
    deriv_4, deriv_3s = smooth_derivative(deriv_3e, diff_t, n_points, rep_mask)
    deriv_4e, rep_mask = repeat_points(deriv_4, n_points, repetitions=n_repeats, rep_mask=rep_mask)
    deriv_4s = smooth(deriv_4e, n_points, rep_mask)
    deriv_13 = - deriv_1s * deriv_3s  # invert the sign to make peaks positive
    deriv_13e, rep_mask = repeat_points(deriv_13, n_points, repetitions=n_repeats, rep_mask=rep_mask)
    deriv_13s = smooth(deriv_13e, n_points, rep_mask)
    # return the raw derivs, the smooth derivs and smooth signal
    r_derivs = np.vstack([deriv_1, deriv_2, deriv_3, deriv_13, deriv_4])
    s_derivs = np.vstack([deriv_1s, deriv_2s, deriv_3s, deriv_4s])
    return signal_s, r_derivs, s_derivs, deriv_13s
    

def deriv_noise_level(deriv_13s, n_points):
    """Calculate the noise level in and array centered around zero
    with a number of high peaks that are not noise.
    Designed for the deriv_13s in eclipse finding.
    """
    # todo: try to use std_outside as estimator
    # take the absolute value of the minimum and use that as upper cut off limit
    abs_min = np.abs(np.min(deriv_13s))
    mask_1 = (deriv_13s < abs_min)
    noise_1 = np.average(np.abs(deriv_13s[mask_1]))
    # now use this first noise level for another cut off take a weighted average
    mask_2 = (deriv_13s > noise_1) & (deriv_13s < abs_min)
    weights = 1 * mask_1 + 3 * mask_2
    if (np.sum(weights) == 0):
        # nothing has passed the conditions
        noise_13s = np.max(np.abs(deriv_13s))
    else:
        noise_13s = np.average(np.abs(deriv_13s), weights=weights)
        # if noise_13s is below almost all points in deriv_13s, it is smoothed too much
        ratio = np.sum(deriv_13s > noise_13s) / len(deriv_13s)
        noise_13s += ratio**n_points * np.max(np.abs(deriv_13s))
    return noise_13s


def measure_eclipse_presence(times, signal, gaps, n_points):
    """Measures various parameters to determine the presence of eclipses
    and their prominence in the data given a value for n_points.
    Gives an estimate of the expected number of eclipses, the number of
    eclipses found, the approximate SNR and the deviation of the signal
    from the smoothing.
    See also: confirm_eclipses
    """
    signal_s, r_derivs, s_derivs, deriv_13s = prepare_derivatives(times, signal, gaps, n_points)
    # this chi squared-like estimator is for measuring the deviation from the original signal
    deviation = np.sum((signal - signal_s)**2)
    noise = deriv_noise_level(deriv_13s, n_points)
    peaks, props = sp.signal.find_peaks(deriv_13s, height=np.max(deriv_13s) / 2)
    n_peaks = len(peaks)
    if (n_peaks == 0):
        # no peaks found at all: don't go any further
        n_confirmed = 0
        pk_signal = 0
    else:
        gaps_i = np.arange(len(gaps))[gaps]
        inserter = np.searchsorted(peaks, gaps_i)
        peaks_gaps = np.insert(peaks, inserter, gaps_i)
        # peaks_gaps = np.sort(np.append(gaps_i, peaks))  # todo: is faster, but doesn't give a nice gap_mask
        gap_mask = np.zeros_like(peaks_gaps, dtype=bool)
        gap_mask[inserter + np.arange(len(inserter))] = True  # position of the gap indices in peaks_gaps
        # exclude intervals that are gaps, or next to gaps
        gapped_intervals = (gap_mask[:-1] | gap_mask[1:])
        condition_0 = np.invert(gapped_intervals)
        interval_range = peaks_gaps[1:] - peaks_gaps[:-1]  # number of points covering each interval
        # take half the interval range, but the full range in case of gaps
        half_range = np.ceil(interval_range / (1 + np.invert(gap_mask[1:] | gap_mask[:-1]))).astype(int)
        between = [signal[i:j + 1] for i, j in zip(peaks_gaps[:-1], peaks_gaps[1:])]
        out_left = [signal[max(i, 0):j] for i, j in zip(peaks_gaps[:-1] - half_range, peaks_gaps[:-1])]
        out_right = [signal[i + 1:j + 2] for i, j in zip(peaks_gaps[1:], peaks_gaps[1:] + half_range)]
        in_avg = np.array([np.average(cut) for cut in between])
        in_std = np.array([np.std(cut) for cut in between])
        out_left_avg = np.array([np.average(cut) if (len(cut) > 0) else 0 for cut in out_left])
        out_right_avg = np.array([np.average(cut) if (len(cut) > 0) else 0 for cut in out_right])
        out_avg = (out_left_avg + out_right_avg) / 2
        condition_1 = (in_avg + 2 * in_std < out_avg)
        # check whether the amount of smoothing is acceptable compared to interval length
        condition_2 = (interval_range > n_points / 2)
        # finally, calculate the number of intervals adhering to the conditions
        condition = condition_0 & condition_1 & condition_2 & np.invert(gap_mask)[:-1]
        n_confirmed = np.sum(condition / (1 + gapped_intervals))
        if (n_confirmed > 0):
            # pk_signal = np.average(deriv_13s[peaks[condition][0]])
            pk_signal = np.average(deriv_13s[peaks_gaps[:-1][condition]])
        else:
            pk_signal = np.max(deriv_13s[peaks])
    peak_snr = pk_signal / noise
    return n_peaks, n_confirmed, peak_snr, deviation


def confirm_eclipses(times, signal, min_n=2, max_n=30, plot_diagnostics=False):
    """Confirms whether eclipses are found in the light curve or not, and
    serves to find the best number of points for smoothing the signal
    in the further analysis (n_points).
    This is a brute force routine, so might be suboptimal to run every time.
    On the bright side, similar data sets will have similar optimal n_points
    (mainly driven by the amount of points covering eclipse in/egress).
    """
    if (min_n < 2):
        print('min_n = 2 is really the minimum.')
        min_n = 2
    gaps = mark_gaps(times)
    n_range = np.arange(min_n, max_n)
    n_peaks = np.zeros_like(n_range, dtype=int)
    n_confirmed = np.zeros_like(n_range, dtype=int)
    data_snr = np.zeros_like(n_range, dtype=float)
    deviation = np.zeros_like(n_range, dtype=float)
    for n in n_range:
        i = n - min_n
        n_peaks[i], n_confirmed[i], data_snr[i], deviation[i] = measure_eclipse_presence(times, signal, gaps, n)
    # todo: see if any snr from individual peaks can help
    # we want to maximize snr and minimize deviation from the original signal
    maximize = data_snr / deviation
    optimal_i = np.argmax(maximize)
    best_n = n_range[optimal_i]
    best_snr = data_snr[optimal_i]
    n_found = n_confirmed[optimal_i]
    if (n_found == 0):
        # no confirmed eclipses: minimize snr!
        n_found_zero = (n_confirmed == 0)
        minimize = data_snr * deviation
        optimal_i = np.argmin(minimize[n_found_zero])
        best_n = n_range[n_found_zero][optimal_i]
        best_snr = data_snr[n_found_zero][optimal_i]

    if plot_diagnostics:
        fig, ax = plt.subplots()
        ax.plot(n_range, n_confirmed / np.max(n_confirmed), label='n_confirmed')
        ax.plot(n_range, data_snr / np.max(data_snr), label='data_snr')
        ax.plot(n_range, deviation / np.max(deviation), label='deviation')
        if n_found:
            ax.plot(n_range, maximize / np.max(maximize), label='data_snr / deviation')
        else:
            ax.plot(n_range, minimize / np.max(minimize), label='data_snr * deviation')
        plt.plot([best_n, best_n], [0, 1], label=f'best n (n found = {n_found})')
        ax.set_xlabel('n_points')
        ax.set_ylabel('normalized statistics')
        plt.tight_layout()
        plt.legend()
        plt.show()
    return n_found, best_n, best_snr


def slope_walker(signal, peaks, slope_sign, no_gaps, walk_up=True):
    """Walk up or down a slope.
    Walk_up = True: walk in the slope sign direction
    Walk_up = False: walk against the slope sign direction
    """
    if not walk_up:
        slope_sign = -slope_sign
    max_i = len(signal) - 1
    
    def check_edges(indices):
        return (indices > 0) & (indices < max_i)
    
    def check_slope(prev_s, cur_s):
        if walk_up:
            return (prev_s < cur_s)
        else:
            return (prev_s > cur_s)
    
    # start at the peaks
    prev_i = peaks
    prev_s = signal[prev_i]
    # step in the desired direction
    check_cur_edges = check_edges(prev_i + slope_sign)
    cur_i = prev_i + slope_sign * check_cur_edges
    cur_s = signal[cur_i]
    # check whether the next two points might be lower
    next_point = check_slope(prev_s, signal[cur_i + slope_sign * check_edges(cur_i + slope_sign)])
    next_point_2 = check_slope(prev_s, signal[cur_i + 2 * slope_sign * check_edges(cur_i + 2 * slope_sign)])
    # check that we only walk down the slope (also check next point
    check_cur_slope = check_slope(prev_s, cur_s) | next_point | next_point_2
    # additionally, check that we don't cross gaps
    check_gaps = no_gaps[cur_i]
    # combine the checks for the current indices
    check = (check_cur_slope & check_gaps & check_cur_edges)
    # define the indices to be optimized
    cur_i = prev_i + slope_sign * check
    while np.any(check):
        prev_i = cur_i
        prev_s = signal[prev_i]
        # step in the desired direction
        check_cur_edges = check_edges(prev_i + slope_sign)
        cur_i = prev_i + slope_sign * check_cur_edges
        cur_s = signal[cur_i]
        # check whether the next two points might be lower
        next_point = check_slope(prev_s, signal[cur_i + slope_sign * check_edges(cur_i + slope_sign)])
        next_point_2 = check_slope(prev_s, signal[cur_i + 2 * slope_sign * check_edges(cur_i + 2 * slope_sign)])
        # and check that we only walk down the slope
        check_cur_slope = check_slope(prev_s, cur_s) | next_point | next_point_2
        # additionally, check that we don't cross gaps
        check_gaps = no_gaps[cur_i]
        check = (check_cur_slope & check_gaps & check_cur_edges)
        # finally, make the actual approved steps
        cur_i = prev_i + slope_sign * check
    return cur_i


def mark_eclipses(signal_s, deriv_13s, s_derivs, gaps, n_points):
    """Mark the positions of eclipse in/egress
    The snr is used as a significance criterion in finding the eclipses.
    See: 'prepare_derivatives' to get the other input for this function.
    Returns arrays of indices for the initial found in/egress positions
    plus the in/egress start and end points
    and finally the sign of the slope of deriv 3 at the peaks.
    """
    deriv_1s, deriv_2s, deriv_3s, deriv_4s = s_derivs[[0, 1, 2, 3]]
    # find the peaks from combined deriv 1*3
    peaks, props = sp.signal.find_peaks(deriv_13s, height=np.max(deriv_13s) / 16 / n_points)
    slope_sign = np.sign(deriv_3s[peaks]).astype(int)  # sign of the slope in deriv_2s
    max_i = len(deriv_2s) - 1
    no_gaps = np.invert(gaps)  # need True everywhere but at the gap positions
    # get the position of eclipse in/egress and positions of eclipse bottom
    # -- walk from each peak position towards the positive peak in deriv_2s
    peaks_pos = slope_walker(deriv_2s, peaks, slope_sign, no_gaps, walk_up=True)
    # -- walk from each peak position towards the negative peak in deriv_2s
    peaks_neg = slope_walker(deriv_2s, peaks, slope_sign, no_gaps, walk_up=False)
    
    # tests to see if the eclipses are really eclipses
    confirmed = np.ones_like(peaks, dtype=bool)  # whether the peaks have passed the conditions
    
    # if peaks_neg start converging on the same points, we need some extra analysis
    check_converge = np.zeros_like(peaks, dtype=bool)
    check_converge[:-1] = (np.diff(peaks_neg) < n_points)
    check_converge[1:] |= check_converge[:-1]
    check_converge &= no_gaps[peaks_neg]  # don't count gaps as convergence
    if (np.sum(check_converge) > len(peaks) / 2.5):
        # when using deriv_4s, the following adjustment is needed
        peaks -= (n_points % 2)
        peaks_neg -= (n_points % 2)
        peaks_pos -= (n_points % 2)
        # -- walk from each peak position towards the maximum in deriv_4s
        prev = deriv_4s[peaks]
        cur_index = peaks - slope_sign
        cur = deriv_4s[cur_index]
        check_slope = (prev < cur)  # check that we only walk up the slope
        check_pkn = (slope_sign * cur_index > slope_sign * peaks_neg)  # check that we don't cross peaks_neg
        check = (check_slope & check_pkn & check_converge)
        peaks_neg_2 = peaks - slope_sign * check  # the indices to be optimized
        while np.any(check):
            prev = deriv_4s[peaks_neg_2]
            # check that we don't go over the array index limits
            cur_index = peaks_neg_2 - slope_sign
            check_edges = (cur_index > 0) & (cur_index < max_i)
            cur_index = peaks_neg_2 - slope_sign * check
            cur = deriv_4s[cur_index]
            # and check that we only walk down the slope
            check_slope = (prev < cur)
            # additionally, check that we don't cross gaps
            check_pkn = (slope_sign * cur_index > slope_sign * peaks_neg)
            check = (check_edges & check_slope & check_pkn & check_converge)
            peaks_neg_2 -= slope_sign * check
        # -- now walk from each peak position towards the next minimum in deriv_4s
        prev = deriv_4s[peaks]
        cur_index = peaks - slope_sign
        cur = deriv_4s[cur_index]
        check_slope = (prev > cur)  # check that we only walk up the slope
        check_pkn = (slope_sign * cur_index > slope_sign * peaks_neg)  # check that we don't cross peaks_neg
        check = (check_slope & check_pkn & check_converge)
        peaks_neg_3 = peaks - slope_sign * check  # the indices to be optimized
        while np.any(check):
            prev = deriv_4s[peaks_neg_3]
            # check that we don't go over the array index limits
            cur_index = peaks_neg_3 - slope_sign
            check_edges = (cur_index > 0) & (cur_index < max_i)
            cur_index = peaks_neg_3 - slope_sign * check
            cur = deriv_4s[cur_index]
            # and check that we only walk down the slope
            check_slope = (prev > cur)
            # additionally, check that we don't cross gaps
            check_pkn = (slope_sign * cur_index > slope_sign * peaks_neg)
            check = (check_edges & check_slope & check_pkn & check_converge)
            peaks_neg_3 -= slope_sign * check
        # the peaks that have not reached peaks_neg are accepted, the others are not
        accept_crit = (peaks_neg_3 != peaks_neg) & check_converge
        reject_crit = (peaks_neg_3 == peaks_neg) & check_converge
        peaks_neg[accept_crit] = peaks_neg_2[accept_crit]
        confirmed[reject_crit] = False
        # revert the adjustments to the peaks
        peaks += (n_points % 2)
        peaks_neg += (n_points % 2)
        peaks_pos += (n_points % 2)
    
    # define some useful stuff
    neg_slope = (slope_sign == -1)
    pos_slope = (slope_sign == 1)
    right_side = np.column_stack([peaks_pos[neg_slope], peaks_neg[neg_slope]])
    left_side = np.column_stack([peaks_neg[pos_slope], peaks_pos[pos_slope]])
    right_cut = [[int(1.5*j - 0.5*i + 1), int(2.5*j - 1.5*i + 2)] for i, j in right_side]
    left_cut = [[int(2.5*i - 1.5*j - 1), int(1.5*i - 0.5*j)] for i, j in left_side]
    
    # peak to peak difference in deriv_2
    sdt_2_out = np.zeros_like(peaks, dtype=float)
    sdt_2_out[neg_slope] = [np.std(deriv_2s[i:j]) for i, j in right_cut]
    sdt_2_out[pos_slope] = [np.std(deriv_2s[i:j]) for i, j in left_cut]
    finites = np.isfinite(sdt_2_out)
    sdt_2_out[np.invert(finites)] = np.median(sdt_2_out[finites])
    pktpk_2 = (deriv_2s[peaks_pos] - deriv_2s[peaks_neg]) / (np.average(sdt_2_out) + sdt_2_out)
    confirmed &= (pktpk_2 > 3)  # safe restriction: 3, harsh restriction: 6
    
    # peak height in deriv_13s
    sdt_13_out = np.zeros_like(peaks, dtype=float)
    sdt_13_out[neg_slope] = [np.std(deriv_13s[i:j]) for i, j in right_cut]
    sdt_13_out[pos_slope] = [np.std(deriv_13s[i:j]) for i, j in left_cut]
    finites = np.isfinite(sdt_13_out)
    sdt_13_out[np.invert(finites)] = np.median(sdt_13_out[finites])
    peaks_13 = deriv_13s[peaks] / (np.average(sdt_13_out) + sdt_13_out)
    confirmed &= (peaks_13 > 2)  # safe restriction: 2, harsh restriction: 6
    
    # add 1 if n_points is odd (it seems to work)
    peaks += (n_points % 2)
    peaks_neg += (n_points % 2)
    peaks_pos += (n_points % 2)
    
    # define some useful stuff
    neg_slope = (slope_sign == -1)
    pos_slope = (slope_sign == 1)
    right_side = np.column_stack([peaks_pos[neg_slope], peaks_neg[neg_slope]])
    left_side = np.column_stack([peaks_neg[pos_slope], peaks_pos[pos_slope]])
    right_cut = [[int(1.5*j - 0.5*i + 1), int(2.5*j - 1.5*i + 2)] for i, j in right_side]
    left_cut = [[int(2.5*i - 1.5*j - 1), int(1.5*i - 0.5*j)] for i, j in left_side]
    point_outside = (2 * peaks_neg - peaks_pos).astype(int)
    point_outside = np.clip(point_outside, 0, max_i)
    
    # difference in signal height in/out of eclipse
    std_0_out = np.zeros_like(peaks, dtype=float)
    std_0_out[neg_slope] = [np.std(signal_s[j:2*j - i + 1]) for i, j in right_side]
    std_0_out[pos_slope] = [np.std(signal_s[2*i - j:i + 1]) for i, j in left_side]
    finites = np.isfinite(std_0_out)
    std_0_out[np.invert(finites)] = np.median(std_0_out[finites])
    signal_difference = (signal_s[peaks_neg] - signal_s[peaks_pos]) / (np.average(std_0_out) + std_0_out)
    confirmed &= (signal_difference > 3)
    
    # difference in slope (deriv_1s)
    std_1_out = np.zeros_like(peaks, dtype=float)
    std_1_out[neg_slope] = [np.std(deriv_1s[i:j]) for i, j in right_cut]
    std_1_out[pos_slope] = [np.std(deriv_1s[i:j]) for i, j in left_cut]
    finites = np.isfinite(std_1_out) & (std_1_out != 0)
    std_1_out[np.invert(finites)] = np.median(std_1_out[finites])
    slope_difference = np.abs(deriv_1s[peaks] - deriv_1s[point_outside]) / (np.average(std_1_out) + std_1_out)
    confirmed &= (slope_difference > 2)  # safe restriction: 2, stronger restriction: 3
    
    # sanity check on the measured slope difference
    slope_check = np.abs(deriv_1s[peaks] - deriv_1s[peaks_neg]) / (np.average(std_1_out) + std_1_out)
    confirmed &= (slope_difference > slope_check)
    
    added_strength = (signal_difference + slope_difference + pktpk_2 + peaks_13)
    
    # select the ones that passed
    peaks = peaks[confirmed]
    peaks_neg = peaks_neg[confirmed]
    peaks_pos = peaks_pos[confirmed]
    slope_sign = slope_sign[confirmed]
    
    # plt.plot(times, signal / np.max(signal) * np.max(signal_difference))
    # plt.plot(times, deriv_1s / np.max(deriv_1s) * np.max(slope_difference))
    # plt.plot(times, deriv_2s / np.max(deriv_2s) * np.max(pktpk_2))
    # plt.plot(times, deriv_3s / np.max(deriv_3s) * np.max(signal_difference))
    # plt.plot(times, deriv_13s / np.max(deriv_13s) * np.max(added_strength))
    # for i, j in zip(peaks_pos[neg_slope], peaks_neg[neg_slope]):
        # ts = times[j:2*j - i + 1]
        # ts = times[int(1.5 * j - 0.5 * i + 1):int(2.5 * j - 1.5 * i + 2)]
        # ts = times[int(1.5*i - 0.5*j):int(1.5*j - 0.5*i + 1)]
        # plt.scatter(ts, np.zeros_like(ts))
        # ts = times[i:j + 1]
        # plt.scatter(ts, np.zeros_like(ts))
    # for i, j in zip(peaks_neg[pos_slope], peaks_pos[pos_slope]):
        # ts = times[2*i - j:i + 1]
        # ts = times[int(2.5 * i - 1.5 * j - 1):int(1.5 * i - 0.5 * j)]
        # ts = times[int(1.5*i - 0.5*j):int(1.5*j - 0.5*i + 1)]
        # plt.scatter(ts, np.zeros_like(ts))
        # ts = times[i:j + 1]
        # plt.scatter(ts, np.zeros_like(ts))
    # plt.scatter(times[peaks], signal_difference, label='signal difference')
    # plt.scatter(times[point_outside], slope_difference, label='slope difference')
    # plt.scatter(times[peaks], peaks_13, label='peaks_13')
    # plt.scatter(times[peaks], pktpk_2, label='pktpk_2')
    # plt.scatter(times[peaks], added_strength, label='added')
    # plt.scatter(times[peaks_pos], np.zeros_like(peaks_pos), label='peaks_pos')
    # plt.scatter(times[peaks_neg], np.zeros_like(peaks_neg), label='peaks_neg')
    # plt.scatter(times[point_outside], np.zeros_like(point_outside), label='point_outside')
    # plt.plot(times[[0, -1]], [3, 3], label='snr')
    # plt.plot(times[[0, -1]], [6, 6], label='snr')
    # plt.legend()
    return peaks, peaks_neg, peaks_pos, slope_sign


def assemble_eclipses(times, peaks, peaks_neg, peaks_pos, slope_sign, gaps):
    """Goes through the found peaks in combined deriv 1*3, the sign of
    the slope in deriv 2 and the gap indices, to assemble the eclipses
    in a neat array of indices.
    Eclipses are marked by 4 indices each: ingress top and bottom,
    and egress bottom and top (in that order).
    times is used to check for complete eclipses in case of gapped data.
    Returns the array of eclipse indices and one with flags, meaning:
    f (full), fg (full with gap), lh (left half) and rh (right half)
    Also returns an 'error' array which gives an indication of the error
    (in the time domain) on the eclipse position given as index values.
    """
    # stitch the right peaks together to get the eclipses
    used = np.zeros_like(peaks, dtype=bool)
    gaps_i = np.arange(len(gaps))[gaps]
    flags = []
    ecl_indices = []
    for i, (pk, pkp, pkn) in enumerate(zip(peaks, peaks_pos, peaks_neg)):
        if not used[i]:
            # # check whether there is a gap between pk and pkn
            # if (pk > pkn):
            #     gap_between_pk = gaps_i[(gaps_i > pkn) & (gaps_i < pk)]
            # elif (pk < pkn):
            #     gap_between_pk = gaps_i[(gaps_i > pkn) & (gaps_i < pk)]
            
            if (slope_sign[i] == -1):
                # we have a loose eclipse end --> must be gap to the left
                gap_i = gaps_i[gaps_i < pk][-1]
                ecl_indices.append([gap_i, gap_i, pkp, pkn])
                flags.append('rh')
                used[i] = True
            elif (slope_sign[i] == 1):
                # might have a full eclipse or a loose eclipse beginning
                if (i == (len(peaks) - 1)):
                    # end of data set --> we have a loose eclipse beginning
                    gap_i = gaps_i[gaps_i > pk][0]
                    ecl_indices.append([pkn, pkp, gap_i, gap_i])
                    flags.append('lh')
                    used[i] = True
                elif ((slope_sign[i] == 1) & (slope_sign[i + 1] == 1)):
                    # we have a loose eclipse beginning --> must be a gap to the right
                    gap_i = gaps_i[gaps_i > pk][0]
                    ecl_indices.append([pkn, pkp, gap_i, gap_i])
                    flags.append('lh')
                    used[i] = True
                elif ((slope_sign[i] == 1) & (slope_sign[i + 1] == -1)):
                    # we might have a full eclipse (but check for gap in between)
                    gap_i = gaps_i[(gaps_i > pk) & (gaps_i < peaks[i + 1])]
                    if (len(gap_i) > 0):
                        # found gap(s) --> probably have two loose parts
                        t_ecl = times[peaks_neg[i + 1]] - times[pkn]
                        t_peak = times[pkp] - times[pkn]
                        if (t_ecl < 10 * t_peak):
                            # maybe it still is one eclipse
                            ecl_indices.append([pkn, pkp, peaks_pos[i + 1], peaks_neg[i + 1]])
                            flags.append('fg')
                            used[i] = True
                            used[i + 1] = True
                        else:
                            # more likely two loose parts (so only extract the first half now)
                            gap_i = gaps_i[gaps_i > pk][0]
                            ecl_indices.append([pkn, pkp, gap_i, gap_i])
                            flags.append('lh')
                            used[i] = True
                    else:
                        #  have one whole eclipse
                        ecl_indices.append([pkn, pkp, peaks_pos[i + 1], peaks_neg[i + 1]])
                        flags.append('f')
                        used[i] = True
                        used[i + 1] = True
            else:
                # something went wrong, this is more likely not an eclipse
                used[i] = True
    # todo: check if eclipses are loose halves where there are no gaps...
    # todo: also, remove this (below) again
    flags = np.array(flags)
    ecl_indices = np.array(ecl_indices)[(flags == 'f') | (flags == 'fg')]
    flags = flags[(flags == 'f') | (flags == 'fg')]
    return np.array(ecl_indices), np.array(flags)


def plot_eclipse_diagnostics(times, signal, signal_s, s_derivs, deriv_13s, r_derivs, peaks, peaks_neg, peaks_pos):
    """Plot the signal and derivatives with the eclipse points marked."""
    deriv_1s, deriv_2s, deriv_3s, deriv_4s = s_derivs[[0, 1, 2, 3]]
    deriv_1, deriv_2, deriv_3 = r_derivs[[0, 1, 2]]
    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=[14, 10])
    ax[0].plot(times, signal, label='raw')
    ax[0].plot(times, signal_s, label='smoothed')
    ax[0].scatter(times[peaks], signal[peaks], label='eclipse marker', c='tab:orange')
    ax[0].scatter(times[peaks_neg], signal[peaks_neg], label='outside', c='tab:red')
    ax[0].scatter(times[peaks_pos], signal[peaks_pos], label='inside', c='tab:green')
    ax[1].plot(times, deriv_1s, label='first deriv')
    ax[1].scatter(times[peaks], deriv_1s[peaks], label='eclipse marker', c='tab:orange')
    ax[1].scatter(times[peaks_neg], deriv_1s[peaks_neg], label='outside', c='tab:red')
    ax[1].scatter(times[peaks_pos], deriv_1s[peaks_pos], label='inside', c='tab:green')
    ax[2].plot(times, deriv_2s, label='second deriv')
    ax[2].scatter(times[peaks], deriv_2s[peaks], label='eclipse marker', c='tab:orange')
    ax[2].scatter(times[peaks_neg], deriv_2s[peaks_neg], label='outside', c='tab:red')
    ax[2].scatter(times[peaks_pos], deriv_2s[peaks_pos], label='inside', c='tab:green')
    ax[3].plot(times, deriv_3s, label='third deriv')
    ax[3].scatter(times[peaks], deriv_3s[peaks], label='eclipse marker', c='tab:orange')
    ax[3].scatter(times[peaks_neg], deriv_3s[peaks_neg], label='outside', c='tab:red')
    ax[3].scatter(times[peaks_pos], deriv_3s[peaks_pos], label='inside', c='tab:green')
    ax[4].plot(times, deriv_13s, label='deriv 1*3')
    ax[4].scatter(times[peaks], deriv_13s[peaks], label='eclipse marker', c='tab:orange')
    ax[4].scatter(times[peaks_neg], deriv_13s[peaks_neg], label='outside', c='tab:red')
    ax[4].scatter(times[peaks_pos], deriv_13s[peaks_pos], label='inside', c='tab:green')
    # ax[4].plot(times, deriv_4s, label='deriv 4')
    # ax[4].scatter(times[peaks], deriv_4s[peaks], label='eclipse marker', c='tab:orange')
    # ax[4].scatter(times[peaks_neg], deriv_4s[peaks_neg], label='outside', c='tab:red')
    # ax[4].scatter(times[peaks_pos], deriv_4s[peaks_pos], label='inside', c='tab:green')
    for i in range(5):
        ax[i].legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    return


def find_eclipses(times, signal, n_points, plot_diagnostics=False):
    """Finds the eclipses in a light curve, regardless of other variability.
    Provide the time stamps, signal (flux), number of points for smoothing,
    
    A SNR of 20 is a safe value in most cases, for raw data. In case the data
    has already seen sone form of averaging, reducing the SNR is a good
    strategy to find low amplitude eclipses that are otherwise not found.
    """
    gaps = mark_gaps(times)
    signal_s, r_derivs, s_derivs, deriv_13s = prepare_derivatives(times, signal, gaps, n_points)
    
    peaks, peaks_neg, peaks_pos, slope_sign = mark_eclipses(signal_s, deriv_13s, s_derivs, gaps, n_points)
    ecl_indices, flags = assemble_eclipses(times, peaks, peaks_neg, peaks_pos, slope_sign, gaps)
    
    if plot_diagnostics:
        plot_eclipse_diagnostics(times, signal, signal_s, s_derivs, deriv_13s, r_derivs, peaks, peaks_neg, peaks_pos)
    return ecl_indices, flags


def measure_eclipses(times, signal, ecl_indices, flags):
    """Get the eclipse midpoints, widths and depths.
    Widths return zero in case of half eclipses, while depths are given for both
    full and half eclipses (just not averaged for the latter). The eclipse
    midpoints for half eclipses are estimated from the average eclipse width.
    
    A measure for flat-bottom-ness is also given (ratio between measured
    eclipse width and width at the bottom. Be aware that a ratio of zero does
    not mean that it is not a flat-bottomed eclipse per se. On the other hand,
    a non-zero ratio is a strong indication that there is a flat bottom.
    """
    # prepare some arrays
    m_full = (flags == 'f') | (flags == 'fg')  # mask of the full eclipses
    m_left = (flags == 'lh')
    m_right = (flags == 'rh')
    l_o = ecl_indices[:, 0]  # left outside
    l_i = ecl_indices[:, 1]  # left inside
    r_i = ecl_indices[:, -2]  # right inside
    r_o = ecl_indices[:, -1]  # right outside
    # calculate the widths
    widths_bottom = times[r_i[m_full]] - times[l_i[m_full]]
    widths = np.zeros_like(flags, dtype=float)
    widths[m_full] = times[r_o[m_full]] - times[l_o[m_full]]
    # calculate the ratios: a measure for how 'flat-bottomed' it is
    ratios = np.zeros_like(flags, dtype=float)
    ratios[m_full] = widths_bottom / widths[m_full]
    # calculate the depths
    depths_l = signal[l_o] - signal[l_i]  # can be zero
    depths_r = signal[r_o] - signal[r_i]  # can be zero
    denom = 2 * m_full + 1 * m_left + 1 * m_right  # should give 2's and 1's (only!)
    depths = (depths_l + depths_r) / denom
    # determine the eclipse midpoints, and estimate them for half eclipses
    ecl_mid = np.zeros_like(flags, dtype=float)
    ecl_mid[m_full] = (times[l_o[m_full]] + times[r_o[m_full]] + times[l_i[m_full]] + times[r_i[m_full]]) / 4
    if np.any(m_full):
        avg_width = np.average(widths[m_full])
        ecl_mid[m_left] = times[l_o[m_left]] + avg_width / 2
        ecl_mid[m_right] = times[r_o[m_right]] - avg_width / 2
    else:
        # if we only have half eclipses, take the inner points as next best estimate
        ecl_mid[m_left] = times[l_i[m_left]]
        ecl_mid[m_right] = times[r_i[m_right]]
    return ecl_mid, widths, depths, ratios


def measure_grouping(period, ecl_mid, indices, p_err, widths, w_err, depths, d_err):
    """Measures how closely the phase folded eclipses are grouped in phase space.
    Folds the times of eclipse midpoints by a given period and measures the
    median absolute deviation of the phases. Returns the total MAD and the phases.
    """
    # put the zero point at a specific place between the eclipses
    zero_point = (ecl_mid[0] + period / 4)
    phases = fold_time_series(ecl_mid, period, zero=zero_point)
    group_1 = indices[phases <= 0]
    group_2 = indices[phases > 0]
    n_g1 = len(group_1)
    n_g2 = len(group_2)
    empty = (n_g2 == 0)
    # define the denominators, avoiding infinities
    denom_1 = (n_g1 - 1) + 0.1 * (n_g1 == 1)
    if not empty:
        denom_2 = (n_g1 - 1) + 0.1 * (n_g1 == 1)

    avg_1 = np.average(phases[group_1])
    phase_dev = np.sum(((phases[group_1] - avg_1) / p_err)**2) / denom_1
    if not empty:
        avg_2 = np.average(phases[group_2])
        phase_dev += np.sum(((phases[group_2] - avg_2) / p_err)**2) / denom_2
        phase_dev /= 2

    # avg_1 = np.average(widths[group_1])
    # std_1 = np.std(widths[group_1])
    # width_sep = avg_1 / std_1
    # if not empty:
    #     avg_2 = np.average(widths[group_2])
    #     std_2 = np.std(widths[group_2])
    #     width_sep = np.abs(avg_1 - avg_2) / (std_1 + std_2)
    #
    # avg_1 = np.average(depths[group_1])
    # std_1 = np.std(depths[group_1])
    # depth_sep = avg_1 / std_1
    # if not empty:
    #     avg_2 = np.average(depths[group_2])
    #     std_2 = np.std(depths[group_2])
    #     depth_sep = np.abs(avg_1 - avg_2) / (std_1 + std_2)
    # todo: add this when widths are a bit better (so that short periods don't get large separation
    return phase_dev, phases


def test_separation(phases, widths, depths, indices):
    """Simple test to see whether the widths and depths are in separate
    distributions or not using the phases."""
    group_1 = indices[phases <= 0]
    group_2 = indices[phases > 0]
    n_g1 = len(group_1)
    n_g2 = len(group_2)
    if (n_g2 == 0) | (n_g1 == 1) | (n_g2 == 1):
        # no separation if there is nothing to separate,
        # or cannot say anything about distribution of 1 point
        separate_w, separate_d = False, False
    elif (n_g1 + n_g2 < 8):
        # for low numbers, use 5 percent difference as criterion
        g1_avg_w = np.average(widths[group_1])
        g2_avg_w = np.average(widths[group_2])
        separate_w = (max(g1_avg_w, g2_avg_w) > 1.05 * min(g1_avg_w, g2_avg_w))
        g1_avg_d = np.average(depths[group_1])
        g2_avg_d = np.average(depths[group_2])
        separate_d = (max(g1_avg_d, g2_avg_d) > 1.05 * min(g1_avg_d, g2_avg_d))
    else:
        g1_avg_w = np.average(widths[group_1])
        g2_avg_w = np.average(widths[group_2])
        g1_std_w = np.std(widths[group_1])
        g2_std_w = np.std(widths[group_2])
        std_w = min(g1_std_w, g2_std_w)
        if (std_w == 0):
            std_w = max(g1_std_w, g2_std_w)
        separate_w = (abs(g1_avg_w - g2_avg_w) > 3 * std_w)
        g1_avg_d = np.average(depths[group_1])
        g2_avg_d = np.average(depths[group_2])
        g1_std_d = np.std(depths[group_1])
        g2_std_d = np.std(depths[group_2])
        std_d = min(g1_std_d, g2_std_d)
        if (std_d == 0):
            std_d = max(g1_std_d, g2_std_d)
        separate_d = (abs(g1_avg_d - g2_avg_d) > 3 * std_d)
    return separate_w, separate_d


def measure_period(times, signal, ecl_mid, widths, depths):
    """Determines the time of the midpoint of the first primary eclipse (t0)
    and the eclipse (orbital if possible) period. Also returns an array of flags
    with a 'p' for primary and 's' for secondary for each eclipse.
    """
    n_ecl = len(ecl_mid)
    if (n_ecl < 2):
        # no eclipses or a single eclipse... return None
        t_zero, ecl_period, flags, stats = None, None, np.array([]), np.array([])
        stats = np.array([])
    elif (n_ecl == 2):
        # only two eclipses... only one guess possible
        ecl_period = ecl_mid[1] - ecl_mid[0]
        if (depths[0] > 1.1 * depths[1]):
            t_zero = ecl_mid[0]
            flags = ['p', 's']
        elif (1.1 * depths[0] < depths[1]):
            t_zero = ecl_mid[1]
            flags = ['s', 'p']
        else:
            # if the depth is within ten percent, make no distinction
            t_zero = ecl_mid[0]
            flags = ['p', 'p']
        stats = np.array([])
    else:
        # it is assumed that relatively few false positive eclipses where given
        t_between = np.diff(ecl_mid)
        t_between_long = t_between[:-1] + t_between[1:]
        # calculate some useful qtt's from these timings
        med_short = np.median(t_between)
        med_long = np.median(t_between_long)
        split = (med_long - med_short) / 2
        mask_short = (t_between > med_short - split) & (t_between < med_short + split)
        avg_short = np.average(t_between[mask_short])
        std_short = np.std(t_between[mask_short])
        mask_long = (t_between_long > med_long - split) & (t_between_long < med_long + split)
        avg_long = np.average(t_between_long[mask_long])
        std_long = np.std(t_between_long[mask_long])
        # determine where to search for the best period
        n_bins_short = np.ceil(np.sqrt(n_ecl - 1))
        n_bins_long = np.ceil(np.sqrt(n_ecl - 2))
        edges_short = np.linspace(avg_short - 3 * std_short, avg_short + 3 * std_short, n_bins_short)
        hist_short, edges_short = np.histogram(t_between, bins=edges_short)
        edges_long = np.linspace(avg_long - 3 * std_long, avg_long + 3 * std_long, n_bins_long)
        hist_long, edges_long = np.histogram(t_between_long, bins=edges_long)
        # plt.step(edges_short[:-1], hist_short)
        # plt.step(edges_long[:-1], hist_long)
        # put a certain total number of points in all bins, according to density
        n_short = np.ceil(1000 * hist_short / (n_ecl - 1)).astype(int)
        n_long = np.ceil(1000 * hist_long / (n_ecl - 2)).astype(int)
        # define all possible periods that are going to be examined
        periods_short = [np.linspace(p1, p2, n) for p1, p2, n in zip(edges_short[:-1], edges_short[1:], n_short)]
        periods_short = np.concatenate(periods_short)
        periods_long = [np.linspace(p1, p2, n) for p1, p2, n in zip(edges_long[:-1], edges_long[1:], n_long)]
        periods_long = np.concatenate(periods_long)
        all_periods = np.append(periods_short, periods_long)
        # establish some error estimates
        time_err = np.median(np.diff(times))
        mp_err = min(time_err, std_short)
        w_err = time_err / 2
        d_err = np.median(np.abs(np.diff(signal))) / 2
        # measure the chi-squareds of the phases, widths and depths
        indices = np.arange(n_ecl)
        phase_chi2 = np.zeros_like(all_periods, dtype=float)
        # width_chi2 = np.zeros_like(all_periods, dtype=float)
        # depth_chi2 = np.zeros_like(all_periods, dtype=float)
        all_phases = np.zeros([len(all_periods), n_ecl], dtype=float)
        for i, p in enumerate(all_periods):
            phase_chi2[i], all_phases[i] = measure_grouping(p, ecl_mid, indices, mp_err/p,
                                                            widths, w_err, depths, d_err)
        plt.scatter(all_periods, phase_chi2)
        # plt.scatter(all_periods, width_chi2)
        # plt.scatter(all_periods, depth_chi2)
        
        i_short = np.sum(n_short)  # index where the long periods start
        # get the best period from both period intervals and some comparison outcomes
        argmin_short = np.argmin(phase_chi2[:i_short])
        best_p_short = periods_short[argmin_short]
        best_chi2_short = phase_chi2[:i_short][argmin_short]
        best_phases_short = all_phases[:i_short][argmin_short]
        argmin_long = np.argmin(phase_chi2[i_short:])
        best_p_long = periods_long[argmin_long]
        best_chi2_long = phase_chi2[i_short:][argmin_long]
        best_phases_long = all_phases[i_short:][argmin_long]
        sep_w_long, sep_d_long = test_separation(best_phases_long, widths, depths, indices)
        # decide on the best period (long or short)
        if ((best_chi2_long < best_chi2_short) | sep_d_long | sep_w_long):
            # long period has lowest dev or have separation of depths or widths in long period
            ecl_period = best_p_long
            phases = best_phases_long
            separate_w = sep_w_long
            separate_d = sep_d_long
        else:
            # take the best fitting short period
            ecl_period = best_p_short
            phases = best_phases_short
            separate_w = False
            separate_d = False

        # and finally determine primary/secondary if possible
        group_1 = indices[phases <= 0]
        group_2 = indices[phases > 0]
        if (len(group_2) == 0):
            primary_g1 = True  # cannot discern between p and s
        elif separate_d:
            # separation by eclipse depth
            primary_g1 = (np.average(depths[group_1]) > np.average(depths[group_2]))
        elif separate_w:
            # separation by eclipse width
            primary_g1 = (np.average(widths[group_1]) > np.average(widths[group_2]))
        else:
            # no clear separation: take the first eclipse as primary
            primary_g1 = (group_1[0] < group_2[0])
        # make the p/s flags
        flags = np.zeros(n_ecl, dtype=str)
        flags[group_1] = 'p' * primary_g1 + 's' * (not primary_g1)
        flags[group_2] = 'p' * (not primary_g1) + 's' * primary_g1
        # put t_zero on the first primary eclipse
        t_zero = ecl_mid[flags == 'p'][0]
        # assemble the statistics about eclipse width and depth (primary first)
        w_stats_1 = [np.average(widths[group_1]), np.std(widths[group_1])]
        w_stats_2 = [np.average(widths[group_2]), np.std(widths[group_2])]
        d_stats_1 = [np.average(depths[group_1]), np.std(depths[group_1])]
        d_stats_2 = [np.average(depths[group_2]), np.std(depths[group_2])]
        if primary_g1:
            stats = np.column_stack([w_stats_1, w_stats_2, d_stats_1, d_stats_2])
        else:
            stats = np.column_stack([w_stats_2, w_stats_1, d_stats_2, d_stats_1])
    return t_zero, ecl_period, flags, stats


def plot_period_diagnostics(times, signal, ecl_indices, ecl_mid, flags_ps, t_0, period, widths, depths):
    """Plot the signal, mark primary and secondary eclipses and plot the period."""
    ecl_mask = mask_eclipses(times, ecl_indices[:, [0, -1]])
    ecl_bottom_mask = mask_eclipses(times, ecl_indices[:, [1, -2]])
    period_array = np.arange(t_0, times[-1], period)
    plot_height = np.max(signal) + 0.02 * (np.max(signal) - np.min(signal))
    phases = fold_time_series(ecl_mid, period, (t_0 + period / 4))
    
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[14, 10])
    ax[0].scatter(ecl_mid, phases, c='tab:blue', marker='o', label='eclipse midpoints')
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
    ax[0].legend()
    ax[1].scatter(times[ecl_mask], signal[ecl_mask])
    ax[1].scatter(times[np.invert(ecl_mask)], signal[np.invert(ecl_mask)], label='eclipses')
    ax[1].scatter(times[np.invert(ecl_bottom_mask)], signal[np.invert(ecl_bottom_mask)], label='eclipse bottoms')
    ax[1].plot(period_array, np.full_like(period_array, plot_height), c='k', marker='|')
    for i, flag in enumerate(flags_ps):
        marker = '^' * (flag == 'p') + 's' * (flag == 's')
        colour = 'tab:red' * (flag == 'p') + 'tab:purple' * (flag == 's')
        ax[1].scatter(ecl_mid[i], plot_height + 0.02 * (np.max(signal) - np.min(signal)), c=colour, marker=marker)
    ax[1].scatter([], [], c='tab:red', marker='^', label='primaries')
    ax[1].scatter([], [], c='tab:purple', marker='s', label='secondaries')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('signal')
    ax[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    return


def extrapolate_eclipses(period, t_0, time_frame):
    """Calculate where eclipses are expected to happen from the period and t_zero.
    Give the time interval of where to extrapolate eclipses (as two time points).
    """
    t_before = time_frame[0] - t_0
    n_start = np.ceil(t_before / period).astype(int)
    n_end = n_start + np.floor((time_frame[1] - time_frame[0]) / period).astype(int)
    eclipses = t_0 + period * np.arange(n_start, n_end)
    return eclipses

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    