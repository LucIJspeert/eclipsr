"""This module contains functions to find eclipses, measure their properties and
do various miscellaneous things with them (or on arrays in general).

Code written by: Luc IJspeert
"""

import numpy as np
import scipy as sp
import scipy.signal
import scipy.special
import matplotlib.pyplot as plt


def cut_eclipses(times, eclipses):
    """Returns a boolean mask covering up the eclipses.
    Give the eclipse times as a series of two points in time.
    See also: mask_eclipses
    Can of course be used to cover up any set of two time points.
    """
    mask = np.ones_like(times, dtype=bool)
    for ecl in eclipses:
        mask = mask & ((times < ecl[0]) | (times > ecl[1]))
    return mask


def mask_eclipses(times, eclipses):
    """Returns a boolean mask covering up the eclipses.
    Give the eclipse indices as a series of two indices per eclipse.
    See also: cut_eclipses
    Can of course be used to cover up any set of two indices.
    """
    mask = np.ones_like(times, dtype=bool)
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
    low = np.zeros(len(jd_sectors))
    high = np.zeros(len(jd_sectors))
    averages = np.zeros(len(jd_sectors))
    for i, mask in enumerate(mask_sect):
        if np.any(mask):
            masked_s = signal[mask]
            averages[i] = np.average(masked_s)
            low[i] = np.average(masked_s[masked_s < averages[i]])
            low[i] = np.average(masked_s[masked_s < low[i]])
            low[i] = np.average(masked_s[masked_s < low[i]])
            high[i] = np.average(masked_s[masked_s > averages[i]])
            high[i] = np.average(masked_s[masked_s > high[i]])
            high[i] = np.average(masked_s[masked_s > high[i]])
    
    difference = high - low
    max_diff = np.max(difference)
    threshold = 2 * high - averages  # to remove spikes (from e.g. momentum dumps)
    thr_mask = np.ones_like(times, dtype=bool)
    # adjust the signal so that it has a more uniform range
    for i, mask in enumerate(mask_sect):
        if np.any(mask):
            signal_copy[mask] = (signal[mask] - averages[i]) / difference[i] * max_diff + averages[i]
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
                ax[0].plot([sect[0], sect[1]], [2 * high[i] - averages[i], 2 * high[i] - averages[i]], c='tab:purple')
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


def fold_time_series(times, period, zero):
    """Fold the given time series over the orbital period to go to function of phase.
    Returns phase array for all timestamps using the provided reference zero point.
    Returned phases are between -0.5 and 0.5
    """
    phases = ((times - zero) / period + 0.5) % 1 - 0.5
    return phases


def mark_gaps(a):
    """Mark the two points at either side of gaps in a somewhat-uniformly separated
    series of monotonically ascending numbers (e.g. timestamps).
    Returns a boolean array and the gap widths in units of the smallest step size.
    """
    diff = np.diff(a)
    min_d = np.min(diff)
    gap_width = diff / min_d
    gaps = (gap_width > 4)  # gaps that are at least 4 times the minimum time step
    gap_width[np.invert(gaps)] = 1  # set non-gaps to width of 1
    gaps = np.append(gaps, [False])  # add a point back to match length with a
    gaps[1:] |= gaps[:-1]  # include both points on either side of the gap
    gap_width = np.floor(gap_width[gaps[:-1]]).astype(int)
    if gaps[-1]:
        gap_width = np.append(gap_width, [1])  # need to add an extra item to gap_width
    return gaps, gap_width


def multiple_gaps(gaps):
    """Mark the points that are on both sides next to a gap."""
    multi_gap = np.zeros_like(gaps)
    multi_gap[gaps] = True
    multi_gap[:-1] &= gaps[1:]
    multi_gap[1:] &= gaps[:-1]
    return multi_gap


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
        repetitions, repetition_mask = np.ones_like(t, dtype=int), np.ones_like(t, dtype=bool)
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
        widths = np.ceil(widths / 2).astype(int)
        max_repeats = np.copy(widths)
        max_repeats[1:] += max_repeats[:-1]
        widths = np.where(widths < n, widths, n)
        max_repeats = np.where(max_repeats < n, max_repeats, n)
        # the number of repetitions is at least 1 and max n
        repetitions = np.ones_like(t, dtype=int)
        repetitions[gaps] = max_repeats
        # start the repetition mask by repeating the (inverted) gap mask
        repetition_mask = np.repeat(np.invert(gaps), repetitions)
        # mark where the positions of the gap edges ('gaps') are in the repetition mask
        new_positions = np.cumsum(repetitions) - 1
        gap_edges = np.zeros_like(repetition_mask, dtype=bool)
        gap_edges[new_positions[gap_p1]] = True
        gap_edges[new_positions[gap_p2] - (widths[gap_start] - 1)] = True
        # remove the points that where originally part of 'a' from the repetition_mask:
        repetition_mask |= gap_edges
        # finally, repeat the start and end of the array as well
        repetitions[[0, -1]] += (n - 1)
        repeat_ends = np.zeros(n - 1, dtype=bool)
        repetition_mask = np.concatenate([repeat_ends, repetition_mask, repeat_ends])
    return repetitions, repetition_mask


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
    becomes an actual derivative.
    Can also apply a mask to the output arrays, for if they had repeats in them.
    Also returns the smoothed a.
    See also: smooth, smooth_diff
    """
    diff, a_smooth = smooth_diff(a, n, mask=mask)
    d_dt = diff/dt
    return d_dt, a_smooth


def prepare_derivatives(times, signal, n_points):
    """Calculate various derivatives of the light curve for the purpose of eclipse finding.
    Retruns all the raw and smooth arrays in vertically stacked groups
    (signal_s, r_derivs, s_derivs)
    [s=smoothed, r=raw]
    """
    diff_t = np.diff(np.append(times, 2 * times[-1] - times[-2]))
    # get the repetition array and the repetition mask
    n_repeats, rep_mask = repeat_points_internals(times, n_points)
    # array versions: e=extended, s=smoothed
    np.repeat(signal, n_repeats)
    signal_e = np.repeat(signal, n_repeats)
    deriv_1, signal_s = smooth_derivative(signal_e, diff_t, n_points, rep_mask)
    deriv_1e = np.repeat(deriv_1, n_repeats)
    deriv_2, deriv_1s = smooth_derivative(deriv_1e, diff_t, n_points, rep_mask)
    deriv_2e = np.repeat(deriv_2, n_repeats)
    deriv_3, deriv_2s = smooth_derivative(deriv_2e, diff_t, n_points, rep_mask)
    deriv_3e = np.repeat(deriv_3, n_repeats)
    deriv_3s = smooth(deriv_3e, n_points, rep_mask)
    deriv_13 = - deriv_1s * deriv_3s  # invert the sign to make peaks positive
    deriv_13e = np.repeat(deriv_13, n_repeats)
    deriv_13s = smooth(deriv_13e, n_points, rep_mask)
    # return the raw derivs, the smooth derivs and smooth signal
    r_derivs = np.vstack([deriv_1, deriv_2, deriv_3, deriv_13])
    s_derivs = np.vstack([deriv_1s, deriv_2s, deriv_3s, deriv_13s])
    return signal_s, r_derivs, s_derivs


def find_best_n(times, signal, min_n=2, max_n=40, dev_limit=1.8, diagnostic_plot=False):
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
    n_range = np.arange(min_n, max_n)
    noise = np.zeros_like(n_range, dtype=float)
    deviation = np.zeros_like(n_range, dtype=float)
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
        deviation[i] = np.sum(((signal - signal_s) / err_est)**2) / len(signal)
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
        ax.plot(n_range, noise, label='deviation')
        ax.plot(n_range, deviation, label='deviation')
        plt.plot([n_range[0], n_range[-1]], [dev_limit, dev_limit])
        ax.set_xlabel('n_points')
        ax.set_ylabel('statistic')
        plt.tight_layout()
        plt.legend()
        plt.show()
    return best_n


def curve_walker(signal, peaks, slope_sign, no_gaps, mode='up', look_ahead=True):
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
        elif 'zero' in mode:
            condition = np.full_like(cur_s, dtype=bool)
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
    if look_ahead:
        # check an additional point ahead
        next_point = check_condition(prev_s, signal[cur_i + slope_sign * check_edges(cur_i + slope_sign)])
        next_point_2 = check_condition(prev_s, signal[cur_i + 2 * slope_sign * check_edges(cur_i + 2 * slope_sign)])
    else:
        next_point = np.zeros_like(cur_i, dtype=bool)
        next_point_2 = np.zeros_like(cur_i, dtype=bool)
    # check that we fulfill the condition (also check next points)
    check_cur_slope = check_condition(prev_s, cur_s) | next_point | next_point_2
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
        # check whether the next two points might be lower
        next_point = check_condition(prev_s, signal[cur_i + slope_sign * check_edges(cur_i + slope_sign)])
        if look_ahead:
            # check an additional point ahead
            next_point_2 = check_condition(prev_s, signal[cur_i + 2 * slope_sign * check_edges(cur_i + 2 * slope_sign)])
        else:
            next_point_2 = np.zeros_like(next_point, dtype=bool)
        # and check that we fulfill the condition
        check_cur_slope = check_condition(prev_s, cur_s) | next_point | next_point_2
        # additionally, check that we don't cross gaps
        check_gaps = no_gaps[cur_i]
        check = (check_cur_slope & check_gaps & check_cur_edges)
        # finally, make the actual approved steps
        cur_i = prev_i + slope_sign * check
    return cur_i


def mark_eclipses(times, signal_s, s_derivs, r_derivs, n_points):
    """Mark the positions of eclipse in/egress
    See: 'prepare_derivatives' to get the input for this function.
    Returns all peak position arrays in a vertically stacked group,
    the added_snr snr measurements and the slope sign (+ ingress, - egress).
    """
    deriv_1s, deriv_2s, deriv_3s, deriv_13s = s_derivs
    deriv_1r, deriv_2r, deriv_3r, deriv_13r = r_derivs
    dt = np.diff(np.append(times, 2 * times[-1] - times[-2]))
    # find the peaks from combined deriv 1*3
    peaks_13, props = sp.signal.find_peaks(deriv_13s, height=np.max(deriv_13s) / 16 / n_points)
    pk_13_widths, wh, ipsl, ipsr = sp.signal.peak_widths(deriv_13s, peaks_13, rel_height=0.5)
    pk_13_widths = np.ceil(pk_13_widths / 2).astype(int)
    # fig, ax = plt.subplots(nrows=2, sharex=True)
    # ax[0].plot(times, deriv_13s)
    # ax[0].scatter(times[peaks_13], deriv_13s[peaks_13], c='tab:orange')
    
    slope_sign = np.sign(deriv_3s[peaks_13]).astype(int)  # sign of the slope in deriv_2s
    neg_slope = (slope_sign == -1)
    max_i = len(deriv_2s) - 1
    n_peaks = len(peaks_13)
    # need a mask with True everywhere but at the gap positions
    gaps, gap_widths = mark_gaps(times)
    no_gaps = np.invert(gaps)
    # get the position of eclipse in/egress and positions of eclipse bottom
    # -- walk from each peak position towards the positive peak in deriv_2s
    peaks_2_pos = curve_walker(deriv_2s, peaks_13, slope_sign, no_gaps, mode='up', look_ahead=True)
    # -- walk from each peak position towards the negative peak in deriv_2s
    peaks_2_neg = curve_walker(deriv_2s, peaks_13, slope_sign, no_gaps, mode='down', look_ahead=True)
    # if peaks_2_neg start converging on the same points, signal might be sine-like
    check_converge = np.zeros_like(peaks_13, dtype=bool)
    check_converge[:-1] = (np.diff(peaks_2_neg) < 3)
    check_converge[1:] |= check_converge[:-1]
    check_converge &= no_gaps[peaks_2_neg]  # don't count gaps as convergence
    noise_factor = 1
    if (np.sum(check_converge) > n_peaks / 1.2):
        # most of them converge: assume we need to correct all of them
        peaks_2_neg = curve_walker(deriv_2s, peaks_2_pos, slope_sign, no_gaps, mode='down_to_zero', look_ahead=False)
        noise_factor = 0.5  # multiply the noise by half to compensate for dense peak pattern
    elif (np.sum(check_converge) > n_peaks / 2.2):
        # only correct the converging ones
        peaks_2_nn = curve_walker(deriv_2s, peaks_2_pos, slope_sign, no_gaps, mode='down_to_zero', look_ahead=False)
        peaks_2_neg[check_converge] = peaks_2_nn[check_converge]
    
    # peaks in 1 and 3 are not exactly in the same spot: find the right spot here
    indices = np.arange(len(peaks_13))
    peaks_range = np.column_stack([peaks_13 - 1, peaks_13, peaks_13 + 1])
    if (n_points < 20):
        peaks_range = np.column_stack([peaks_13 - 1, peaks_13, peaks_13 + 1])
    elif (n_points < 50):
        peaks_range = np.column_stack([peaks_13 - 2, peaks_13 - 1, peaks_13, peaks_13 + 1, peaks_13 + 2])
    else:
        peaks_range = np.column_stack([peaks_13 - 3, peaks_13 - 2, peaks_13 - 1,
                                       peaks_13, peaks_13 + 1, peaks_13 + 2, peaks_13 + 3])
    med_width = np.median(pk_13_widths).astype(int)
    n_repeats = 2 * med_width + 1
    peaks_range = np.repeat(peaks_13, n_repeats).reshape(n_peaks, n_repeats) + np.arange(-med_width, med_width + 1)
    peaks_range = np.clip(peaks_range, 0, max_i)
    # todo: work this out better
    peaks_1 = np.argmax(-slope_sign.reshape(n_peaks, 1) * deriv_1s[peaks_range], axis=1)
    peaks_1 = peaks_range[indices, peaks_1]
    peaks_3 = np.argmax(slope_sign.reshape(n_peaks, 1) * deriv_3s[peaks_range], axis=1)
    peaks_3 = peaks_range[indices, peaks_3]
    # in/egress peaks (eclipse edges) and bottoms are adjusted a bit
    peaks_edge = np.clip(peaks_2_neg - slope_sign + (n_points % 2) - (n_points == 2) * neg_slope, 0, max_i)
    peaks_bot = np.clip(peaks_2_pos + (n_points % 2), 0, max_i)
    
    # first check for some simple strong conditions on the eclipses
    passed_1 = np.ones_like(peaks_13, dtype=bool)
    passed_1 &= (signal_s[peaks_2_neg] > signal_s[peaks_2_pos])
    passed_1 &= (signal_s[peaks_2_neg] >= signal_s[peaks_13])
    passed_1 &= (signal_s[peaks_13] >= signal_s[peaks_2_pos])
    # ax[0].scatter(times[peaks_13][passed_1], deriv_13s[peaks_13][passed_1], c='tab:red')
    # the slope should be steeper than the bottom of the eclipse
    # passed_1 &= (-slope_sign * deriv_1s[peaks_1] >= -slope_sign * deriv_1s[peaks_2_pos])
    # todo: remove or improve the above statement (or need better peak finding)
    # ax[1].plot(times, deriv_1s)
    # ax[1].scatter(times[peaks_1], -slope_sign * deriv_1s[peaks_1], c='tab:green', marker='+')
    # ax[1].scatter(times[peaks_2_pos], -slope_sign * deriv_1s[peaks_2_pos], c='tab:green', marker='x')
    # ax[1].scatter(times[peaks_bot], -slope_sign * deriv_1s[peaks_bot], c='tab:green', marker='o')
    # ax[0].scatter(times[peaks_13][passed_1], deriv_13s[peaks_13][passed_1], c='tab:green')
    # the peak in 13 should not be right next to a higher peak
    left = np.clip(peaks_13 - 2, 0, max_i)
    right = np.clip(peaks_13 + 2, 0, max_i)
    passed_1 &= (deriv_13s[left] < deriv_13s[peaks_13]) & (deriv_13s[right] < deriv_13s[peaks_13])
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
    # do some additional more advanced tests for confidence level
    # ax[0].scatter(times[peaks_13], deriv_13s[peaks_13], c='tab:purple')
    # plt.show()
    passed_2 = np.ones_like(peaks_13, dtype=bool)
    
    # define points away from the peaks and a mask for all peaks
    point_outside = (2 * peaks_2_neg - peaks_2_pos).astype(int)
    point_outside = np.clip(point_outside, 0, max_i)
    point_inside = (2 * peaks_2_pos - peaks_2_neg).astype(int)
    point_inside = np.clip(point_inside, 0, max_i)
    peak_pairs = np.column_stack([point_outside, point_inside])
    peak_pairs[neg_slope] = peak_pairs[neg_slope][:, ::-1]
    mask_peaks = mask_eclipses(signal_s, peak_pairs)
    # fig, ax = plt.subplots()
    # ax.plot(times, deriv_13s / np.max(deriv_13s))
    
    # get the estimates for the noise in signal_s
    noise_0 = np.average(np.abs(deriv_1r[mask_peaks] * dt[mask_peaks])) * noise_factor
    # signal to noise in signal_s: difference in height in/out of eclipse
    snr_0 = (signal_s[peaks_edge] - signal_s[peaks_bot]) / noise_0
    # ax.scatter(times[peaks_13][passed_2], snr_0[passed_2], c='tab:orange')
    passed_2 &= (snr_0 > 2)
    # get the estimates for the noise in deriv_1s
    noise_1 = np.average(np.abs(deriv_2r[mask_peaks] * dt[mask_peaks])) * noise_factor
    # signal to noise in deriv_1s: difference in slope
    value_around = np.min([-slope_sign * deriv_1s[point_outside], -slope_sign * deriv_1s[point_inside]], axis=0)
    snr_1 = (-slope_sign * deriv_1s[peaks_1] - value_around) / noise_1
    # ax.scatter(times[peaks_13][passed_2], snr_1[passed_2], c='tab:red')
    passed_2 &= (snr_1 > 2)
    # sanity check on the measured slope difference
    slope_check = np.abs(deriv_1s[peaks_1] - deriv_1s[peaks_2_neg]) / noise_1
    # slope_check = np.abs(deriv_1s[peaks_1] - deriv_1s[peaks_2_neg]) - noise_1
    passed_2 &= (snr_1 > slope_check)
    # get the estimates for the noise in deriv_2s
    noise_2 = np.average(np.abs(deriv_2s[mask_peaks])) * noise_factor
    # signal to noise in deriv_2s: peak to peak difference
    snr_2 = (deriv_2s[peaks_2_pos] - deriv_2s[peaks_2_neg]) / noise_2
    # ax.scatter(times[peaks_13][passed_2], snr_2[passed_2], c='tab:green')
    passed_2 &= (snr_2 > 1)
    # get the estimates for the noise in deriv_3s
    noise_3 = np.average(np.abs(deriv_3s[mask_peaks])) * noise_factor
    # signal to noise in deriv_3s: peak height
    value_around = np.min([slope_sign * deriv_3s[point_outside], slope_sign * deriv_3s[point_inside]], axis=0)
    snr_3 = (slope_sign * deriv_3s[peaks_3] - value_around) / noise_3
    # ax.scatter(times[peaks_13][passed_2], snr_3[passed_2], c='tab:purple')
    passed_2 &= (snr_3 > 1)
    # get the estimates for the noise in deriv_13s
    noise_13 = np.average(np.abs(deriv_13s[mask_peaks])) * noise_factor
    # signal to noise in deriv_13s: peak height
    value_around = np.min([deriv_13s[point_outside], deriv_13s[peaks_bot]], axis=0)
    snr_13 = (deriv_13s[peaks_13] - value_around) / noise_13
    # ax.scatter(times[peaks_13][passed_2], snr_13[passed_2], c='tab:pink')
    passed_2 &= (snr_13 > 2)
    
    # do a final check on the total 'eclipse strength'
    added_snr = (snr_0 + snr_1 + snr_2 + snr_3)
    # ax.scatter(times[peaks_13][passed_2], added_snr[passed_2], c='tab:grey')
    # plt.show()
    passed_2 &= (added_snr > 10)
    
    # select the ones that passed
    peaks_1 = peaks_1[passed_2]
    peaks_2_neg = peaks_2_neg[passed_2]
    peaks_2_pos = peaks_2_pos[passed_2]
    peaks_edge = peaks_edge[passed_2]
    peaks_bot = peaks_bot[passed_2]
    peaks_3 = peaks_3[passed_2]
    peaks_13 = peaks_13[passed_2]
    peaks = np.vstack([peaks_1, peaks_2_neg, peaks_2_pos, peaks_edge, peaks_bot, peaks_3, peaks_13])
    slope_sign = slope_sign[passed_2]
    added_snr = added_snr[passed_2]
    return peaks, added_snr, slope_sign


def local_extremum(a, start, right=True, maximum=True):
    """Walks left or right in a 1D-array to find a local extremum."""
    max_i = len(a) - 1
    
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
        i = i + right - (not right)
        if (i < 0) | (i > max_i):
            break
        prev = np.copy(cur)
        cur = a[i]
    # adjust i to the previous point (the extremum)
    i = i - right + (not right)
    return i


def match_in_egress(times, signal, signal_s, added_snr, peaks_edge, peaks_bot, slope_sign):
    """Match up the best combinations of ingress and egress to form full eclipses.
    This is done by chopping all peaks up into parts with consecutive sets of
    ingresses and egresses (through slope sign).
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
    used = np.zeros_like(peaks_edge, dtype=bool)
    full_ecl = []
    for i, pk in enumerate(peaks_edge):
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
    not_used = np.ones_like(indices, dtype=bool)
    if (len(full_ecl) != 0):
        # take the average width of the full_ecl
        avg_added = (added_snr[full_ecl[:, 0]] + added_snr[full_ecl[:, 1]]) / 2
        full_widths = (times[peaks_edge[full_ecl[:, 1]]] - times[peaks_edge[full_ecl[:, 0]]])
        med_width = np.median(full_widths[avg_added >= np.average(avg_added)])
        # and compare the other full_ecl against it.
        passed = (full_widths > 0.5 * med_width) & (full_widths < 2.0 * med_width)
        # check the average in-eclipse level compared to surrounding
        avg_inside = np.zeros(len(full_ecl))
        avg_outside = np.zeros(len(full_ecl))
        std_inside = np.zeros(len(full_ecl))
        std_outside = np.zeros(len(full_ecl))
        std_s_outside = np.zeros(len(full_ecl))
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
        passed &= ((avg_inside < avg_outside - std) | (avg_inside < avg_outside - 3 * std_s_outside))
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
    # determine snr-categories
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
        hist3 = np.zeros_like(hist2)
    if np.any(hist3 != 0):
        i_max_3 = np.argmax(hist3)
    else:
        i_max_3 = -1
    
    arr_max_i = np.array([i_max_1, i_max_2, i_max_3])
    index_sorter = np.argsort(arr_max_i)
    i_max_1, i_max_2, i_max_3 = arr_max_i[index_sorter]

    if np.all(edges[arr_max_i] > 20) & np.all(arr_max_i != -1):
        hist_sorter = np.argsort(hist[arr_max_i])
        g1_i = arr_max_i[hist_sorter][-1]
        g2_i = arr_max_i[hist_sorter][-2]
        divider_1 = (edges[g1_i] + edges[g2_i]) / 2
        divider_2 = 0
    elif (i_max_1 == -1) & (i_max_2 == -1):
        divider_1 = 0
        divider_2 = -1
    elif (i_max_1 == -1):
        divider_1 = (edges[i_max_2] + edges[i_max_3]) / 2
        divider_2 = -1
    else:
        g3_i, g2_i, g1_i = arr_max_i[index_sorter]
        divider_1 = (edges[g1_i] + edges[g2_i]) / 2
        divider_2 = (edges[g2_i] + edges[g3_i]) / 2
    
    # plt.hist(added_snr, bins=np.floor(np.sqrt(len(added_snr))).astype(int))
    # plt.plot(edges[:-1], hist2)
    # plt.plot(edges[:-1], hist3)
    # plt.plot([divider_1, divider_1], [0, np.max(hist)])
    # plt.plot([divider_2, divider_2], [0, np.max(hist)])

    # keep track of which category the eclipses belong to
    cat = np.zeros(indices[-1] + 1, dtype='<U2')
    # split up into a high, middle and low snr group
    group_high = (added_snr >= divider_1) #(added_snr >= edges[i_min_1])
    group_mid = (added_snr >= divider_2) & (added_snr < divider_1)
    if (divider_2 != -1):
        group_low = (added_snr < divider_2)
    else:
        group_mid = np.zeros_like(added_snr, dtype=bool)
        group_low = (added_snr < divider_1)
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
    overlap = np.zeros_like(mean_snr, dtype=bool)
    i_full_ecl = np.arange(len(full_ecl))
    for i, ecl in enumerate(full_ecl):
        cond1 = ((ecl[0] > full_ecl[:, 0]) & (ecl[0] < full_ecl[:, 1]))
        cond2 = ((ecl[1] > full_ecl[:, 0]) & (ecl[1] < full_ecl[:, 1]))
        if np.any(cond1 | cond2):
            i_overlap = np.append([i], i_full_ecl[cond1 | cond2])
            snr_vals = mean_snr[i_overlap]
            remove = (snr_vals != np.max(snr_vals))
            overlap[i_overlap] = remove
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
        flags = np.zeros(indices[-1] + 1, dtype='<U3')
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


def find_eclipses(times, signal, n_points, diagnostic_plot=False):
    """Finds the eclipses in a light curve, regardless of other variability.
    n_points is the number of points for smoothing.
    """
    # do the derivatives
    signal_s, r_derivs, s_derivs = prepare_derivatives(times, signal, n_points)
    # find the likely eclipse in/egresses and put them together
    peaks, added_snr, slope_sign = mark_eclipses(times, signal_s, s_derivs, r_derivs, n_points)
    ecl_indices, added_snr, flags = assemble_eclipses(times, signal, signal_s, peaks, added_snr, slope_sign)
    
    if diagnostic_plot:
        plot_marker_diagnostics(times, signal, signal_s, s_derivs, peaks, ecl_indices, flags)
    return ecl_indices, added_snr, flags


def measure_eclipses(times, signal, ecl_indices, flags):
    """Get the eclipse midpoints, widths and depths.
    Widths return zero in case of half eclipses, while depths are given for both
    full and half eclipses (just not averaged for the latter). In high noise cases,
    providing the smoothed light curve can give more accurate eclipse depths.
    The eclipse midpoints for half eclipses are estimated from the average eclipse width.
    
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
    periods = periods.reshape(n_periods, 1)
    added_snr = np.tile(added_snr, n_periods).reshape([n_periods, n_ecl])
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
    avg_1 = avg_1.reshape([n_periods, 1])
    on_line = np.sum(np.abs(phases * group_1 - avg_1) < 0.01, axis=-1, dtype=ldtype)
    avg_2 = np.average(phases[not_empty], weights=group_2[not_empty], axis=-1)
    avg_2 = avg_2.reshape([n_p_ne, 1])
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
    avg_1 = avg_1.reshape([n_periods, 1])
    dev_1 = np.sum(np.abs(added_snr * group_1 - avg_1), axis=-1) / 100
    snr_dev = dev_1 / n_g1
    avg_2 = np.average(added_snr[not_empty], weights=group_2[not_empty], axis=-1)
    avg_2 = avg_2.reshape([n_p_ne, 1])
    dev_2 = np.sum(np.abs(added_snr[not_empty] * group_2[not_empty] - avg_2), axis=-1) / 100
    snr_dev[not_empty] = (dev_1[not_empty] + dev_2) / (n_g1[not_empty] + n_g2[not_empty])
    snr_dev /= on_line
    return phases, phase_dev, snr_dev


def test_separation(phases, variable=None):
    """Simple test to see whether the variable is split into separate
    distributions or not using the phases. If no variable is given of
    which to measure the separation, the phases are compared instead.
    """
    group_1 = (phases <= 0)
    group_2 = (phases > 0)
    n_g1 = len(group_1)
    n_g2 = len(group_2)
    if variable is None:
        # test if the phases are separated by something else than 0.5
        variable = np.copy(phases)
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
        separate = (abs(g1_avg - g2_avg) > 2 * std)
    return separate


def determine_primary(full, group_1, group_2, g_full_1, g_full_2, phases, depths, widths, added_snr):
    n_gf1 = np.sum(g_full_1)
    n_gf2 = np.sum(g_full_2)
    n_g1 = np.sum(group_1)
    n_g2 = np.sum(group_2)
    if ((n_gf1 != 0) & (n_gf2 != 0)):
        if test_separation(phases, depths[full]):
            # separation by eclipse depth
            primary_g1 = (np.average(depths[full][g_full_1]) > np.average(depths[full][g_full_2]))
        elif test_separation(phases, added_snr[full]):
            # separation by added_snr
            primary_g1 = (np.average(added_snr[full][g_full_1]) > np.average(added_snr[full][g_full_2]))
        elif test_separation(phases, widths[full]):
            # separation by eclipse width
            primary_g1 = (np.average(widths[full][g_full_1]) > np.average(widths[full][g_full_2]))
        else:
            # no clear separation or separate_p: take the group with highest average added_snr
            primary_g1 = (np.average(added_snr[full][g_full_1]) > np.average(added_snr[full][g_full_2]))
    elif ((n_g1 != 0) & (n_g2 != 0)):
        if test_separation(phases, depths[group_1 | group_2]):
            # separation by eclipse depth
            primary_g1 = (np.average(depths[group_1]) > np.average(depths[group_2]))
        elif test_separation(phases, added_snr[group_1 | group_2]):
            # separation by added_snr
            primary_g1 = (np.average(added_snr[group_1]) > np.average(added_snr[group_2]))
        elif test_separation(phases, widths[group_1 | group_2]):
            # separation by eclipse width
            primary_g1 = (np.average(widths[group_1]) > np.average(widths[group_2]))
        else:
            # no clear separation or separate_p: take the group with highest average added_snr
            primary_g1 = (np.average(added_snr[group_1]) > np.average(added_snr[group_2]))
    else:
        if ((n_gf1 == 0) & (n_gf2 != 0)) | ((n_g1 == 0) & (n_g2 != 0)):
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
    m_full = np.char.startswith(flags, 'f')
    # m_full_g1 = (flags == 'f-g1')
    m_full_g12 = (flags == 'f-g1') | (flags == 'f-g2')
    m_half = np.invert(m_full)
    n_full_ecl = np.sum(m_full)
    n_full_g12_ecl = np.sum(m_full_g12)
    # first establish an estimate of the period
    if (n_full_ecl < 2):
        # no eclipses or a single eclipse... return None
        ecl_period = None
        full = m_full
        half = m_half
    elif (n_full_ecl == 2):
        # only two eclipses... only one guess possible
        ecl_period = abs(ecl_mid[m_full][1] - ecl_mid[m_full][0])
        full = m_full
        half = m_half
    elif (n_full_g12_ecl == 2):
        # only two high signal eclipses... use those
        ecl_period = abs(ecl_mid[m_full_g12][1] - ecl_mid[m_full_g12][0])
        full = m_full_g12
        half = np.invert(full)
    else:
        if (np.sum(m_full_g12) > 2):
            # if we have enough group 1/2 (high snr) eclipses, those are used
            full = m_full_g12
            half = np.invert(full)
            n_full_ecl = n_full_g12_ecl
        else:
            full = m_full
            half = m_half
        # sort the eclipses first
        ecl_sorter = np.argsort(ecl_mid)
        ecl_mid = ecl_mid[ecl_sorter]
        widths = widths[ecl_sorter]
        depths = depths[ecl_sorter]
        added_snr = added_snr[ecl_sorter]
        m_full = m_full[ecl_sorter]
        # determine where to search for the best period
        t_between = np.diff(ecl_mid[full])
        t_between_2 = t_between[:-1] + t_between[1:]
        min_t_b = max(np.min(t_between), 0.001)  # don't allow it to become too small
        step_t_b = 0.04 * min_t_b
        t_bins = np.arange(min_t_b, np.max(t_between) + step_t_b, step_t_b)
        t_hist, t_bins = np.histogram(t_between, bins=t_bins)
        if (len(t_between_2) > 1):
            t_bins_2 = np.arange(np.min(t_between_2), np.max(t_between_2) + step_t_b, step_t_b)
        else:
            t_bins_2 = np.array([np.min(t_between_2) - step_t_b, np.max(t_between_2) + step_t_b])
        t_hist_2, t_bins_2 = np.histogram(t_between_2, bins=t_bins_2)
        max_1 = np.argmax(t_hist)
        hist_i = np.arange(len(t_hist))
        include = (hist_i > max_1 + 1) | (hist_i < max_1 - 1)  # to include for the second maximum
        if np.any(include):
            max_2 = np.max(t_hist[include])
            max_2 = hist_i[t_hist == max_2][0]
        else:
            max_2 = max_1
        max_3 = np.argmax(t_hist_2)
        # determine the bin widths and centres, then put some points in period space
        b_width_1 = t_bins[max_1 + 1] - t_bins[max_1]
        b_width_2 = t_bins[max_2 + 1] - t_bins[max_2]
        b_width_3 = t_bins_2[max_3 + 1] - t_bins_2[max_3]
        n_cover = 1000
        linspace_0 = np.arange(0, 1, 2 / n_cover)
        linspace_1 = linspace_0 * b_width_1 + t_bins[max_1]
        linspace_1 = np.append(linspace_1, 3 * linspace_0 * b_width_1 + t_bins[max_1] - b_width_1)
        linspace_1 = np.abs(linspace_1)
        linspace_2 = linspace_0 * b_width_2 + t_bins[max_2]
        linspace_2 = np.append(linspace_2, 3 * linspace_0 * b_width_2 + t_bins[max_2] - b_width_2)
        linspace_2 = np.abs(linspace_2)
        linspace_3 = linspace_0 * b_width_3 + t_bins_2[max_3]
        linspace_3 = np.append(linspace_3, 3 * linspace_0 * b_width_3 + t_bins_2[max_3] - b_width_3)
        linspace_3 = np.abs(linspace_3)
        all_periods = np.concatenate([linspace_1, linspace_2, linspace_3])
        # measure the std of the phases and added_snr
        all_phases, phase_dev, snr_dev = measure_grouping(all_periods, ecl_mid[full], added_snr[full])
        total_dev = phase_dev + snr_dev
        # fig, ax = plt.subplots()
        # ax.hist(all_periods, bins=100)
        # ax.hist(t_between_2, bins=t_bins_2)
        # ax.hist(t_between, bins=t_bins)
        # ax.set_xlim([0, None])
        # plt.show()
        # fig, ax = plt.subplots()
        # ax.scatter(all_periods, phase_dev)
        # ax.scatter(all_periods, snr_dev)
        # ax.scatter(all_periods, total_dev)
        # plt.show()
        argmin_1 = np.argmin(total_dev[:n_cover])
        argmin_2 = np.argmin(total_dev[n_cover:2 * n_cover])
        argmin_3 = np.argmin(total_dev[2 * n_cover:3 * n_cover])
        argmin_all = [argmin_1, argmin_2 + n_cover, argmin_3 + 2 * n_cover]
        phases = all_phases[argmin_all]
        # test separation, first in phase, then in depth, then in added_snr and finally width
        separate_p = np.zeros(3, dtype=bool)
        separate_d = np.zeros(3, dtype=bool)
        separate_s = np.zeros(3, dtype=bool)
        separate_w = np.zeros(3, dtype=bool)
        for i, argmin in enumerate(argmin_all):
            if (n_full_ecl > 3):
                separate_p[i] = test_separation(phases[i])
            separate_d[i] = test_separation(phases[i], depths[full])
            separate_s[i] = test_separation(phases[i], added_snr[full])
            separate_w[i] = test_separation(phases[i], widths[full])
        # select the best period from the three best calculated ones
        i_best = np.arange(3)
        if np.any(separate_p):
            i_best = i_best[separate_p][0]
        elif np.any(separate_d):
            i_best = i_best[separate_d][0]
        elif np.any(separate_s):
            i_best = i_best[separate_s][0]
        elif np.any(separate_w):
            i_best = i_best[separate_w][0]
        else:
            # nothing separates out, then take smallest found period
            i_best = 0
        ecl_period = all_periods[argmin_all[i_best]]
    # refine the period (using all full eclipses)
    if ecl_period is not None:
        period_range = np.arange(0.999 * ecl_period, 1.001 * ecl_period, (0.002 * ecl_period) / 200)
        all_phases, phase_dev, snr_dev = measure_grouping(period_range, ecl_mid[m_full], added_snr[m_full])
        ecl_period = period_range[np.argmin(phase_dev)]
    # # now better determine the primaries and secondaries and other/tertiaries
    if ecl_period is not None:
        # regroup the eclipses, including some eclipses and throwing out deviating ones (as 'tertiaries')
        phases = fold_time_series(ecl_mid[full], ecl_period, zero=(ecl_mid[full][0] + ecl_period / 4))
        phases_half = fold_time_series(ecl_mid[half], ecl_period, zero=(ecl_mid[full][0] + ecl_period / 4))
        g_full_1 = (phases <= 0)
        g_full_2 = (phases > 0)
        # regroup the eclipses and throw out deviating ones (as 'tertiaries')
        # select only full eclipses within 0.02 in phase space
        avg_1 = np.average(phases[g_full_1])
        g_full_1 = (np.abs(phases - avg_1) < 0.02)
        if np.any(g_full_2):
            avg_2 = np.average(phases[g_full_2])
            g_full_2 = (np.abs(phases - avg_2) < 0.02)
        # select only those half eclipses within 0.01 in phase space
        if np.any(g_full_1):
            avg_1 = np.average(phases[g_full_1])
            g_half_1 = (np.abs(phases_half - avg_1) < 0.01)
        else:
            g_half_1 = np.zeros_like(phases_half, dtype=bool)
        if np.any(g_full_2):
            avg_2 = np.average(phases[g_full_2])
            g_half_2 = (np.abs(phases_half - avg_2) < 0.01)
        else:
            g_half_2 = np.zeros_like(phases_half, dtype=bool)
        # assemble the final groups
        group_1 = np.zeros_like(ecl_mid, dtype=bool)
        group_2 = np.zeros_like(ecl_mid, dtype=bool)
        group_1[full] = g_full_1
        group_2[full] = g_full_2
        group_1[half] = g_half_1
        group_2[half] = g_half_2
        # check that eclipses are at least separated by the period (remove those that are not)
        ecl_indices = np.arange(len(ecl_mid))
        for i in ecl_indices:
            if group_1[i]:
                selection = group_1
            elif group_2[i]:
                selection = group_2
            else:
                continue
            too_close = np.abs(ecl_mid[i] - ecl_mid[selection]) < 0.8 * ecl_period
            i_selection = ecl_indices[selection][too_close]
            larger_snr = np.argmax(added_snr[i_selection])
            i_selection = i_selection[i_selection != i_selection[larger_snr]]
            group_1[i_selection] = False
            group_2[i_selection] = False
        # determine which group is primary/secondary if possible (from the full eclipses)
        primary_g1 = determine_primary(full, group_1, group_2, g_full_1, g_full_2, phases, depths, widths, added_snr)
        # make the primary/secondary/tertiary flags_pst
        flags_pst = np.zeros_like(ecl_mid, dtype=str)
        flags_pst[group_1] = 'p' * primary_g1 + 's' * (not primary_g1)
        flags_pst[group_2] = 'p' * (not primary_g1) + 's' * primary_g1
        flags_pst[np.invert(group_1 | group_2)] = 't'
        # finally, put the t_zero on the first primary eclipse
        primaries = (flags_pst == 'p')
        if np.any(primaries):
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
        ecl_mask = np.zeros_like(times, dtype=bool)
        ecl_bottom_mask = np.zeros_like(times, dtype=bool)
    if period is not None:
        t_0 = ecl_mid[prim][0]
        period_array = np.arange(t_0, times[0], -period)[::-1]
        period_array = np.append(period_array, np.arange(t_0, times[-1], period))
        phases = fold_time_series(ecl_mid, period, (t_0 + period / 4))
    else:
        phases = np.zeros_like(ecl_mid)
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


def find_ephemeris(times, signal, n_points, diagnostic_plot=False):
    """Finds t0 and the eclipse period from a set of eclipses.
    See also: find_eclipses
    """
    # do the derivatives
    signal_s, r_derivs, s_derivs = prepare_derivatives(times, signal, n_points)
    # find the likely eclipse in/egresses and put them together
    peaks, added_snr, slope_sign = mark_eclipses(times, signal_s, s_derivs, r_derivs, n_points)
    ecl_indices, added_snr, flags = assemble_eclipses(times, signal, signal_s, peaks, added_snr, slope_sign)
    # take some measurements
    ecl_mid, widths, depths, ratios = measure_eclipses(times, signal_s, ecl_indices, flags)
    # if possible, find the period and flags for p/s/t
    t_0, period, flags_pst = estimate_period(ecl_mid, widths, depths, added_snr, flags)
    if diagnostic_plot:
        plot_period_diagnostics(times, signal, signal_s, ecl_indices, ecl_mid, widths, depths, flags, flags_pst, period)
    return t_0, period, flags_pst


def confidence_level(times, period, ecl_mid, added_snr, flags_pst):
    """Determine a number that expresses the confidence level that we have found
    actual eclipses. Higher is better, below ... is probably a false positive.
    """
    # avg snr of primaries plus that of secondaries (minus tertiaries?)
    # times the number of ecl vs number of theoretically visible ones
    primaries = (flags_pst == 'p')
    secondaries = (flags_pst == 's')
    if np.any(primaries):
        sum_p = np.sum(added_snr[primaries])
    else:
        sum_p = 0
    if np.any(secondaries):
        sum_s = np.sum(added_snr[secondaries])
    else:
        sum_s = 0
    if np.any(np.invert(primaries | secondaries)):
        sum_t = np.sum(added_snr[np.invert(primaries | secondaries)])
    else:
        sum_t = 0
    # the number of ecl vs number of theoretically visible ones
    gaps, gap_widths = mark_gaps(times)
    gap_times = np.concatenate([[times[0]], times[gaps], [times[-1]]])
    # primaries
    if np.any(primaries):
        t_0 = ecl_mid[primaries][0]
        possible_prim = np.arange(t_0, times[0], -period)[::-1]
        possible_prim = np.append(possible_prim, np.arange(t_0, times[-1], period))
        # count only possible eclipses where there is data coverage
        mask = np.ones_like(possible_prim, dtype=bool)
        for gt1, gt2 in zip(gap_times[:-1], gap_times[1:]):
            if np.any((times > gt1) & (times < gt2)):
                # there is coverage between these points
                mask |= ((possible_prim > gt1) & (possible_prim < gt2))
        possible_prim = possible_prim[mask]
    else:
        possible_prim = []
    # secondaries
    if np.any(secondaries):
        t_0 = ecl_mid[secondaries][0]
        possible_sec = np.arange(t_0, times[0], -period)[::-1]
        possible_sec = np.append(possible_sec, np.arange(t_0, times[-1], period))
        # count only possible eclipses where there is data coverage
        mask = np.ones_like(possible_sec, dtype=bool)
        for gt1, gt2 in zip(gap_times[:-1], gap_times[1:]):
            if np.any((times > gt1) & (times < gt2)):
                # there is coverage between these points
                mask |= ((possible_sec > gt1) & (possible_sec < gt2))
        possible_sec = possible_sec[mask]
    else:
        possible_sec = []
    # combine into one number
    n_possible = len(possible_prim) + len(possible_sec)
    n_found = np.sum((primaries | secondaries))
    if (n_possible != 0):
        ratio_1 = n_found / n_possible
    else:
        ratio_1 = 1
    ratio_2 = n_found / np.sum(added_snr > 0.5 * np.average(added_snr[primaries | secondaries]))
    confidence = (sum_p + sum_s) * ratio_1 * ratio_2
    if (confidence == 0):
        confidence = -sum_t
    confidence = np.arctan(confidence / 40) * 200 / (np.pi)
    return confidence


def eclipse_stats(flags_pst, widths, depths):
    """Measures the average width and depth for the primary and for the
    secondary eclipses, plus the standard deviations.
    """
    prim = (flags_pst == 'p')
    sec = (flags_pst == 's')
    width_p = [np.average(widths[prim]), np.std(widths[prim])]
    width_s = [np.average(widths[sec]), np.std(widths[sec])]
    depth_p = [np.average(depths[prim]), np.std(depths[prim])]
    depth_s = [np.average(depths[sec]), np.std(depths[sec])]
    width_stats = np.column_stack([width_p, width_s])
    depth_stats = np.column_stack([depth_p, depth_s])
    return width_stats, depth_stats


def find_all(times, signal, mode=1, rescale=True, max_n=40, dev_limit=1.8):
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
    n_points = find_best_n(times, signal, max_n=max_n, dev_limit=dev_limit, diagnostic_plot=dp)
    # do the derivatives
    signal_s, r_derivs, s_derivs = prepare_derivatives(times, signal, n_points)
    # get the likely eclipse indices from the derivatives
    peaks, added_snr, slope_sign = mark_eclipses(times, signal_s, s_derivs, r_derivs, n_points)
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
        t_0, period, flags_pst = None, None, np.array([])
    # check if any primary eclipses were found
    if (len(flags_pst) != 0):
        if (np.sum(flags_pst == 'p') != 0):
            # determine the confidence level
            conf = confidence_level(times, period, ecl_mid, added_snr, flags_pst)
        else:
            conf = None
    else:
        conf = None
    # check if the w/d statistics need to be calculated
    if (conf is not None) & (mode in [4, 5, -1]):
        # determine the collective characteristics
        width_stats, depth_stats = eclipse_stats(flags_pst, widths, depths)
    else:
        width_stats = np.array([[None, None], [None, None]])
        depth_stats = np.array([[None, None], [None, None]])
    # depending on the mode, return (part of) the results
    if (period is None) & (dev_limit == 1.8):
        # if we didn't find anything, try a second time with another n_points
        return find_all(times, signal, mode=mode, rescale=rescale, max_n=max(max_n, 100), dev_limit=2.5)
    elif (period is None) & (dev_limit == 2.5):
        # if we didn't find anything, try a second time with another n_points
        return find_all(times, signal, mode=mode, rescale=rescale, max_n=max(max_n, 150), dev_limit=10**4)
    elif (mode == 2):
        return t_0, period, conf, ecl_mid, widths, depths, ratios, flags_pst
    elif (mode == 3):
        return t_0, period, conf, ecl_mid, widths, depths, ratios, flags_pst, ecl_indices, added_snr
    elif (mode == 4):
        return t_0, period, conf, width_stats, depth_stats
    elif (mode in [5, -1]):
        return t_0, period, conf, width_stats, depth_stats, \
               ecl_mid, widths, depths, ratios, flags_pst, ecl_indices, added_snr
    else:
        # mode == 1 or anything not noted above
        return t_0, period, conf


def extrapolate_eclipses(period, t_0, time_frame):
    """Calculate where eclipses are expected to happen from the period and t_zero.
    Give the time interval of where to extrapolate eclipses (as two time points).
    """
    t_before = time_frame[0] - t_0
    n_start = np.ceil(t_before / period).astype(int)
    n_end = n_start + np.floor((time_frame[1] - time_frame[0]) / period).astype(int)
    eclipses = t_0 + period * np.arange(n_start, n_end)
    return eclipses

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    