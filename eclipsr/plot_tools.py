"""ECLIPSR

This module contains functions to make (diagnostic) plots
for various stages of the eclipse finding.

Code written by: Luc IJspeert
"""

import numpy as np
import matplotlib.pyplot as plt

from . import eclipse_finding as ecf
from . import utility as ut


def rescale_tess_dplot(times, signal, signal_copy, averages, low, high, threshold, mask_sect, jd_sectors):
    """Diagnostic plot for rescale_tess."""
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
    return


def find_best_n_dplot(n_range, deviation, optimize, sine_like, best_n):
    """Diagnostic plot for find_best_n."""
    fig, ax = plt.subplots(figsize=[14, 10])
    ax.plot(n_range[[0, -1]], [2.5, 2.5])
    ax.plot(n_range, deviation, label='deviation')
    ax.plot(n_range, optimize, label='optimize')
    ax.plot(n_range, sine_like, label='sine_like')
    ax.set_xlabel('n_kernel')
    ax.set_ylabel('statistic')
    plt.tight_layout()
    plt.legend(title=f'n={best_n}')
    plt.show()
    return


def plot_marker_diagnostics(times, signal, signal_s, s_derivs, peaks, ecl_indices, flags_lrf, n_kernel):
    """Plot the signal and derivatives with the eclipse points marked."""
    deriv_1s, deriv_2s, deriv_3s, deriv_13s = s_derivs
    peaks_1, peaks_2_neg, peaks_2_pos, peaks_edge, peaks_bot, peaks_3, peaks_13 = peaks
    plot_height = np.max(signal) + 0.02 * (np.max(signal) - np.min(signal))
    
    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=[14, 10])
    ax[0].plot(times, signal, label='raw')
    ax[0].plot(times, signal_s, label=f'smoothed n={n_kernel}')
    ax[0].scatter(times[peaks_1], signal[peaks_1], label='peak marker', c='tab:orange')
    ax[0].scatter(times[peaks_edge], signal[peaks_edge], label='outside', c='tab:red')
    ax[0].scatter(times[peaks_bot], signal[peaks_bot], label='inside', c='tab:green')
    for i, ecl in enumerate(ecl_indices):
        colour = 'tab:red' * (flags_lrf[i] == 0) + 'tab:purple' * (flags_lrf[i] != 0)
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
    ax[4].set_xlabel('time', fontsize=20)
    ax[0].set_ylabel('signal', fontsize=20)
    ax[4].set_ylabel('d1 * d3', fontsize=20)
    # ax[4].tick_params(axis='x', labelsize=14)
    for i in range(5):
        if i in [1, 2, 3]:
            ax[i].set_ylabel(fr'$\frac{{d^{i}}}{{dt^{i}}}$ signal', fontsize=20)
        # ax[i].tick_params(axis='y', labelsize=14)
        ax[i].legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    return


def plot_period_diagnostics(times, signal, signal_s, ecl_indices, ecl_mid, widths, depths, flags_lrf, flags_pst, period):
    """Plot the signal, mark primary and secondary eclipses and plot the period."""
    full = (flags_lrf == 0)
    prim = (flags_pst == 1)
    sec = (flags_pst == 2)
    tert = (flags_pst == 3)
    if (len(ecl_indices) != 0):
        ecl_mask = ecf.mask_eclipses(times, ecl_indices[:, [0, -1]])
        ecl_bottom_mask = ecf.mask_eclipses(times, ecl_indices[:, [1, -2]])
    else:
        ecl_mask = np.zeros([len(times)], dtype=bool)
        ecl_bottom_mask = np.zeros([len(times)], dtype=bool)
    if (period > 0) & np.any(prim):
        t_0 = ecl_mid[prim][0]
        period_array = np.arange(t_0, times[0], -period)[::-1]
        period_array = np.append(period_array, np.arange(t_0, times[-1], period))
        phases = ut.fold_time_series(ecl_mid, period, (t_0 + period / 4))
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
    w_ax.set_xlabel('eclipse width (units of time)', fontsize=14)
    d_ax.set_xlabel('eclipse depth (units of signal)', fontsize=14)
    ax[0].set_xlabel('time', fontsize=20)
    ax[0].set_ylabel('phase', fontsize=20)
    if period is not None:
        ax[0].legend(title=f'period = {period:1.4f}')
    else:
        ax[0].legend()
    ax[1].scatter(times[ecl_mask], signal[ecl_mask])
    ax[1].scatter(times[np.invert(ecl_mask)], signal[np.invert(ecl_mask)], label='eclipses')
    ax[1].scatter(times[np.invert(ecl_bottom_mask)], signal[np.invert(ecl_bottom_mask)], label='eclipse bottoms')
    ax[1].plot(times, signal_s, marker='.', c='grey', alpha=0.6, label='smoothed light curve')
    if (period > 0) & np.any(prim):
        ax[1].plot(period_array, np.full_like(period_array, height_1), c='k', marker='|')
    ax[1].scatter(ecl_mid[prim], np.full_like(ecl_mid[prim], height_2), c='tab:red', marker='^', label='primaries')
    ax[1].scatter(ecl_mid[sec], np.full_like(ecl_mid[sec], height_2), c='tab:purple', marker='s', label='secondaries')
    ax[1].scatter(ecl_mid[tert], np.full_like(ecl_mid[tert], height_2), c='tab:pink', marker='x',
                  label='tertiaries/other')
    ax[1].set_xlabel('time', fontsize=20)
    ax[1].set_ylabel('signal', fontsize=20)
    ax[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    return