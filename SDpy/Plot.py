#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:46:57 2025

@author: Zhang et al.
"""
import os
from obspy import read
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import traceback
from .Fit import compute_spec_ratio,compute_single_spectrum,D_squared,fit_spectrum_model,parallel_spectrum_fit
from .Preprocess import ev_baz,extract_signal_and_noise

import matplotlib.colors as mcolors

def generate_color_list(num_colors):
    """
    Generate a list of color names for plotting.
    
    Parameters:
    -----------
    num_colors : int
        Number of distinct colors required.
        
    Returns:
    --------
    color_list : list of str
        List of color names or codes.
    """
    # Predefined color palette for fewer than 27 items
    preset_colors = [
        'red', 'yellow', 'skyblue', 'rebeccapurple', 'peru', 'sienna',
        'indigo', 'purple', 'pink', 'palevioletred', 'turquoise',
        'coral', 'tomato', 'lightsteelblue', 'teal', 'firebrick',
        'orchid', 'olivedrab', 'bisque', 'thistle', 'orangered',
        'darkcyan', 'wheat', 'azure', 'salmon', 'linen'
    ]
    
    if num_colors <= len(preset_colors):
        return preset_colors[:num_colors]
    
    # If more colors are needed, use CSS4 colors and sort by HSV
    all_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    hsv_sorted = sorted(
        ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
          for name, color in all_colors.items())
    )
    full_color_list = [name for hsv, name in hsv_sorted]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_color_list = []
    for name in full_color_list:
        if name not in seen:
            seen.add(name)
            unique_color_list.append(name)
        if len(unique_color_list) >= num_colors:
            break

    return unique_color_list

def plot_event_waveform(win_length, t_coda, taper_count, overlap_ratio, egf_list, stream, pick_start, noise_start,
                        p_arrival, s_arrival, origin_time, segment_len, event_label, ax, phase_type):
    """
    Plot waveform segment used in spectral ratio analysis.

    Parameters:
    -----------
    win_length     : float, total time length (s) of waveform display
    t_coda         : float, coda normalization factor
    taper_count    : int, number of tapers used in multitaper
    overlap_ratio  : float, overlap ratio between windows (0~1)
    egf_list       : list of EGF event IDs
    stream         : ObsPy Stream object containing waveform
    pick_start     : UTCDateTime, start time of signal window
    noise_start    : UTCDateTime, start time of noise window
    p_arrival      : UTCDateTime, P arrival time
    s_arrival      : UTCDateTime, S arrival time
    origin_time    : UTCDateTime, origin time of event
    segment_len    : float, time window length for plotting (s)
    event_label    : str, event ID or label
    ax             : matplotlib.axes.Axes, target plot axes
    phase_type     : str, 'P', 'S', or 'CODA'

    Returns:
    --------
    None
    """
    n_windows = int((win_length - segment_len * overlap_ratio) / (segment_len * (1 - overlap_ratio)))
    colors = generate_color_list(n_windows)

    if not ax:
        return

    # Choose component
    component = 'T' if stream.select(component='T') else 'Z'
    try:
        trace = stream.select(component=component)[0]
    except IndexError:
        trace = stream.select(component='1')[0]

    trace.detrend('linear')
    trace.detrend('demean')
    trace.taper(max_percentage=0.05)

    rel_noise_start = round(noise_start - trace.stats.starttime, 2)
    time_vals = trace.times(reftime=trace.stats.starttime)
    ax.plot(time_vals, trace.data, "k-", label=trace.stats.channel)
    ax.set_ylabel('Displacement (nm)', fontsize=26, fontweight='bold')

    # Plot phase windows
    time_ref = {
        'P': p_arrival,
        'S': s_arrival,
        'CODA': s_arrival + (t_coda - 1) * (s_arrival - origin_time)
    }.get(phase_type.upper(), pick_start)

    rel_phase_start = time_ref - trace.stats.starttime
    time_vals = [round(t, 2) for t in time_vals]

    try:
        idx_start = time_vals.index(round(rel_phase_start, 2))
        idx_end = time_vals.index(round(rel_phase_start + win_length, 2))
    except ValueError:
        idx_start = np.where(np.array(time_vals) <= round(rel_phase_start, 2))[0][-1]
        idx_end = np.where(np.array(time_vals) <= round(rel_phase_start + win_length, 2))[0][-1]

    y_min, y_max = ax.get_ylim()
    y_segment = trace.data[idx_start:idx_end]
    y_max_seg = max(y_segment)
    # y_min_seg = min(y_segment)

    # Draw noise window
    ax.annotate("", xy=(rel_noise_start, max(1.*y_max_seg,0.6*y_max)),
                xytext=(rel_noise_start + segment_len, max(1.*y_max_seg,0.6*y_max)),
                arrowprops=dict(arrowstyle="<->", facecolor='k'))

    # Draw phase window
    if n_windows <= 1:
        ax.annotate("", xy=(rel_phase_start, max(1.*y_max_seg,0.6*y_max)),
                    xytext=(rel_phase_start + segment_len, max(1.*y_max_seg,0.6*y_max)),
                    arrowprops=dict(arrowstyle="<->", facecolor='k'))
    else:
        for i in range(n_windows):
            start = rel_phase_start + segment_len * (1 - overlap_ratio) * i
            end = start + segment_len
            ax.annotate("", xy=(start, 0.3*y_min + 0.05*i*y_min),
                        xytext=(end, 0.3*y_min + 0.05*i*y_min),
                        arrowprops=dict(arrowstyle="-", edgecolor=colors[i]))

    # Fill regions
    ax.fill_between(time_vals, 1.2*min(y_min, -y_max), 1.2*max(y_max, -y_min),
                    where=(np.array(time_vals) >= rel_noise_start) &
                          (np.array(time_vals) <= rel_noise_start + segment_len),
                    color='gray', alpha=0.3)

    ax.fill_between(time_vals[idx_start:idx_end], 1.2*min(y_min, -y_max), 1.2*max(y_max, -y_min),
                    facecolor='green', alpha=0.1)

    # Add labels
    ax.text(rel_noise_start+segment_len/2, max(1.05*y_max_seg,0.65*y_max), 'Ns',ha='center', va='bottom', 
            fontsize=20, fontweight='bold')#min(-1.1*y_max_seg, -0.7*y_max)
    ax.text(rel_phase_start + win_length/2, max(1.05*y_max_seg,0.65*y_max), phase_type,
            ha='center', va='bottom', fontsize=24, fontweight='bold')

    ax.tick_params(axis='both', labelsize=24)
    ax.get_yaxis().get_major_formatter().set_powerlimits((0, 0))
    ax.set_xlim([0, max(time_vals)])
    ax.set_ylim([1.2*min(y_min, -y_max), 1.2*max(y_max, -y_min)])
    ax.set_xlabel('Time (s)', fontsize=26, fontweight='bold')

    # Set title
    is_egf = event_label == 'EGF' or event_label in egf_list
    title = f"{'EGF' if is_egf else 'Main'} event: {event_label}"
    ax.set_title(title, fontsize=22, fontweight='bold')
    station_name = stream[0].stats.station.strip()
    ax.text(max(time_vals), 1.15*min(y_min, -y_max), station_name,
            ha='right', va='bottom', fontsize=22, fontweight='bold')

    # Tick locator settings
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.get_offset_text().set_fontsize(25)
    
    return None

def plot_station_spectrum(all_station_labels, egf_event_id, freq_main, amp_main, noise_main, ax,
                          station_label, line_color, freq_min, freq_max, wave_type,**egf_data):
    """
    Plot main/EGF/noise spectra for a given station in the spectral ratio panel.

    Parameters:
    -----------
    all_station_labels : list of str, used to track all plotted stations
    egf_event_id        : str or int, EGF event ID
    freq_main           : array-like, frequency array for main spectrum
    amp_main            : array-like, amplitude spectrum of main event
    noise_main          : array-like, noise spectrum of main event
    ax                  : matplotlib.axes.Axes, axes for plotting
    station_label       : str, current station name
    line_color          : color, line color for this station
    freq_min            : float, min frequency (Hz)
    freq_max            : float, max frequency (Hz)
    wave_type           : str, wave type (P/S/Coda)
    **egf_data          : dict with keys:
        - 'x2'         : freq array for EGF
        - 'y2'         : amplitude array for EGF
        - 'y22'        : noise array for EGF
        - 'stations'   : list of already-plotted stations
        - 'egf'        : EGF event ID
        - 'max_y'      : max amplitude value for y-axis scaling

    Returns:
    --------
    updated_stations : list, updated station list
    updated_max_y    : float, updated max y value
    updated_min_y    : float, updated min y value (typically noise floor)
    """

    freq_egf = egf_data.get('freq_egf', [])
    amp_egf = egf_data.get('amp_egf', [])
    noise_egf = egf_data.get('noise_egf', [])
    station_list = egf_data.get('station_list', [])
    egf_label = egf_data.get('egf_label')
    max_y_global = egf_data.get('max_y_global')

    # Plot main event spectrum
    if station_label not in station_list:
        ax.loglog(freq_main, amp_main, linewidth=3, label=station_label, color=line_color)
        station_list.append(station_label)
    else:
        ax.loglog(freq_main, amp_main, linewidth=3, color=line_color)

    # Plot main event noise
    show_label = (f"{station_label}({egf_label})" == all_station_labels[-1]) or (station_label == all_station_labels[-1])
    ax.loglog(freq_main, noise_main, linewidth=2, ls='-', color='darkgray',
              label='noise' if show_label else None)

    # Plot EGF spectra if present
    if not np.array(freq_egf).any() or not np.array(amp_egf).any() or not np.array(noise_egf).any():
        min_y = min(noise_main)
        y_min = min(min(noise_main), max_y_global) * 0.01
        y_max = max(max(amp_main), max_y_global) * 10
    else:
        ax.loglog(freq_egf, amp_egf, linewidth=1.5, ls='--', color=line_color, alpha=1)
        ax.loglog(freq_egf, noise_egf, linewidth=1.5, ls='-.', color='lightgray')
        min_y = min(noise_egf)
        y_min = min(min(noise_egf), max_y_global) * 0.01
        y_max = max(max(amp_main), max_y_global) * 10

    # Set axis limits and appearance
    ax.set_xlim([freq_min, freq_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel('Frequency (Hz)', fontsize=26, fontweight='bold')
    ax.set_ylabel('Amplitude (nm/Hz)', fontsize=26, fontweight='bold')

    # Set log ticks
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.get_xaxis().get_major_formatter().labelOnlyBase = True
    ax.get_xaxis().get_minor_formatter().labelOnlyBase = False
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.tick_params(axis='both', which='both', length=4.0, labelsize=24)

    # Font size for ticks
    for tick in ax.xaxis.get_major_ticks():
        tick.set_label1(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.set_label1(20)

    # Add legend
    ax.legend(loc='best', ncol=3, prop={'size': 18})

    return station_list, max(amp_main), min_y

def plot_individual_spectra_fits(worker_count, model_type, figure, axis, signal_spectra, frequency_bins, noise_spectra, 
                                 travel_times, wave_type, fit_flag, stack_method, quality_factor):
    """
    Description:
    ------------
    Fits and plots individual or stacked seismic spectra, including model curve,
    corner frequency markers, and RMS inset histogram (for weighted stacking).

    Parameters:
    ------------
    worker_count : int
        Number of parallel workers
    model_type : str
        Source model ('sm', etc.)
    figure : matplotlib.figure.Figure
    axis : matplotlib.axes._subplots.AxesSubplot
    signal_spectra : dict
        Amplitude spectra for each station
    frequency_bins : dict
        Frequency bins for each station
    noise_spectra : dict
        Noise spectra for each station
    travel_times : dict
        Travel times per station
    wave_type : str
        Wave type ("P" or "S")
    fit_flag : bool
        Whether to use automatic fitting
    stack_method : str
        "weighted", "average", or "median"
    quality_factor : float
        Quality factor (Q)

    Returns:
    --------
    axis : matplotlib axis
    fc_final : float or dict
        Fitted corner frequency (or stats)
    """
    station_list = list(signal_spectra.keys())
    color_map = generate_color_list(len(station_list))

    if stack_method.lower() == 'weighted':
        fc_list, rms_list = [], []
        for idx, station in enumerate(station_list):
            freqs = frequency_bins[station]
            data = signal_spectra[station]
            noise = noise_spectra[station]
            travel_time = travel_times[station]

            if worker_count > 1 and model_type.lower() == 'sm':
                popt, _, fc_vals, _ = parallel_spectrum_fit(data, freqs, station,
                                                            min(freqs), max(freqs)*1.,
                                                            travel_time, model_type, quality_factor, worker_count)
            else:
                popt, _, _ = fit_spectrum_model(data, freqs, station,
                                                min(freqs), max(freqs)*1.,
                                                travel_time, fit_flag, model_type, quality_factor)

            if popt is not None:
                fc = round(float(popt[1]), 2)
                fit_curve = compute_single_spectrum(freqs, *popt)
                residual = np.square(np.subtract(data, fit_curve))
                rms = round(np.sqrt(np.sum(residual) / np.sum(np.square(data))), 4)
                fc_list.append(fc)
                rms_list.append(rms)

                axis.loglog(freqs, data, lw=1, color=color_map[idx], label=station)
                axis.loglog(freqs, fit_curve, '--', lw=1, color=color_map[idx])
                axis.loglog([fc], compute_single_spectrum(fc, *popt), marker='*', color='orangered', markersize=15)
                axis.loglog(freqs, noise, lw=1, color='gray', alpha=0.6)

        fc_avg, d2_val, fc_list, rms_list = D_squared(fc_list, rms_list)
        axis.text(0.55, 0.1, f'$D^2$ = {d2_val:.4f}', transform=axis.transAxes, fontsize=18, weight='bold')
        axis.text(0.55, 0.15, f'$f_c$ = {fc_avg:.2f}', transform=axis.transAxes, fontsize=18, weight='bold')

        if axis:
            inset_ax = inset_axes(axis, width="30%", height="25%", loc=3, borderpad=6)
            x_range = max(fc_list) - min(fc_list)
            bar_width = 0.05 * x_range if x_range else 0.1
            inset_ax.bar(fc_list, rms_list, width=bar_width, color='#1f77b4', alpha=0.7)
            inset_ax.set_xlabel('Corner Frequency (Hz)', fontsize=18, weight='bold')
            inset_ax.set_ylabel('RMS', fontsize=18, weight='bold')
            inset_ax.tick_params(axis='both', which='both', length=4, labelsize=16)

        fc_final = {'fc_list': fc_list, 'rms_list': rms_list, 'avg_fc': fc_avg, 'D2': d2_val}

    else:
        stacked_data = []
        max_len = max(len(v) for v in signal_spectra.values())
        station_ref = station_list[np.argmax([len(v) for v in signal_spectra.values()])]
        freqs = frequency_bins[station_ref]
        travel_time = travel_times[station_ref]

        for sta in station_list:
            padded = np.pad(signal_spectra[sta], (0, max_len - len(signal_spectra[sta])), mode='edge')
            stacked_data.append(padded)

        if stack_method.lower() == 'average':
            combined = np.mean(stacked_data, axis=0)
        elif stack_method.lower() == 'median':
            combined = np.median(stacked_data, axis=0)

        if worker_count > 1 and model_type.lower() == 'sm':
            popt, _, _, rms = parallel_spectrum_fit(combined, freqs, station_ref,
                                                  min(freqs), max(freqs)*1.,
                                                  travel_time, model_type, quality_factor, worker_count)
        else:
            popt, _, rms = fit_spectrum_model(combined, freqs, station_ref,
                                              min(freqs), max(freqs)*1.,
                                              travel_time, fit_flag, model_type, quality_factor)

        if popt is not None:
            for idx, station in enumerate(station_list):
                axis.loglog(frequency_bins[station], signal_spectra[station], lw=1, color=color_map[idx], label=station)
                axis.loglog(frequency_bins[station], noise_spectra[station], lw=1, color='gray', alpha=0.6)

            fc_final = round(popt[1], 2)
            axis.loglog(freqs, combined, lw=4, color='orangered', label='Stacked')
            axis.loglog(freqs, compute_single_spectrum(freqs, *popt), lw=2, ls='--', color='green')
            y_fit = compute_single_spectrum(freqs, *popt)[np.argmax(freqs >= fc_final)]
            axis.loglog(fc_final, y_fit * 1.12, 'v', color='green')
            axis.text(fc_final * 0.95, y_fit * 1.25, f'f$_c$ = {fc_final}', fontsize=18, weight='bold',ha='right', va='bottom')
            axis.text(0.55, 0.15, f'RMS = {round(rms,2)}', fontsize=18, weight='bold', transform=axis.transAxes)
            axis.tick_params(axis='both', which='both', length=4, labelsize=16)

    axis.tick_params(axis='both', which='both', length=5, labelsize=22)
    axis.set_xlim([min(freqs)*0.9, max(freqs)*1.1])
    ylims = axis.get_ylim()
    axis.set_ylim([ylims[0]*0.5, ylims[1]**1.5/ylims[0]**0.5])
    axis.legend(loc='upper center', ncol=3, fontsize=20)
    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))

    return axis, fc_final

def plot_spectral_ratio_fit(show_fc2, magnitude, target_event, egf_event, freqs, mean_ratio, model_ratio, 
                            fc_candidates, residuals, ax, fit_params, y_max, *additional_info):
    '''
    Plot spectral ratio fit results, including:
    - Spectral ratios for all stations
    - Fitted fc values
    - Optional inset of RMS vs fc
    '''

    summary_type = additional_info[1]

    if fit_params.any():
        if summary_type.lower() != 'weighted':
            try:
                fc1 = fit_params[0]
                ratio_fit = compute_spec_ratio(freqs, *fit_params)
                y_at_fc1 = np.divide(ratio_fit,model_ratio)[np.where(freqs >= fc1)[0][0]]

                ax.text(fc1 * 0.95, y_at_fc1 * 1.25,
                        'f$_c$$_($$_1$$_)$ =  %.2f' % fc1,
                        style='normal', weight='bold', size=18)
                ax.loglog(fc1, y_at_fc1 * 1.12, marker='v', color='green', markersize=10)

                if show_fc2.upper() == 'YES':
                    fc2 = fit_params[1]
                    y_at_fc2 = np.divide(ratio_fit,model_ratio)[np.where(freqs >= fc2)[0][0]]

                    ax.text(fc2 * 0.55, y_at_fc2 * 0.7,
                            'f$_c$$_($$_2$$_)$ =  %.2f' % fc2,
                            style='normal', weight='bold', size=18)
                    ax.loglog(fc2, y_at_fc2 * 0.9, marker='^', color='green', markersize=10)

                # RMS inset
                rms_ax = inset_axes(ax, width="30%", height="25%", loc=3, borderpad=6)
                valid_rms = [(f, r) for f, r in zip(fc_candidates, residuals) if r < 10.6]
                if valid_rms:
                    x_vals, y_vals = zip(*valid_rms)
                    best_idx = np.argmin(y_vals)
                    rms_ax.semilogx(x_vals, y_vals, 'o', ms=3, mfc='blue')
                    rms_ax.semilogx(x_vals[best_idx], y_vals[best_idx],
                                    '*', mfc='blue', ms=8, mec='red')
                    # rms_ax.set_xlim([10 ** np.floor(np.log10(max(min(fc_candidates),1e-10))), max(fc_candidates) * 2])
                    # rms_ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
                    # rms_ax.get_xaxis().get_major_formatter().labelOnlyBase = True
                    # rms_ax.get_xaxis().get_minor_formatter().labelOnlyBase = False
                    rms_ax.set_xticks([1,10,100],[r'$10^0$',r'$10^1$',r'$10^2$'])
                    rms_ax.set_ylim([0, 0.6])
                    rms_ax.set_yticks(np.linspace(0, 0.6, num=4))
                    rms_ax.set_xlabel('Corner Frequency (Hz)', fontsize=18, fontweight='bold')
                    rms_ax.set_ylabel('RMS', fontsize=18, fontweight='bold')
                    rms_ax.tick_params(axis='both', which='both', length=4)
                ax.text(0.55, 0.15, 'rms = %.2f' % y_vals[best_idx],
                        style='normal', weight='bold', size=18, transform=ax.transAxes)

            except:
                traceback.print_exc()

    elif summary_type.lower() == 'weighted':
        freq_by_sta = freqs
        ratio_by_sta = mean_ratio
        fc_weighted = additional_info[2]
        rms_weighted = additional_info[3]
        fc_all = additional_info[4]
        var_all = additional_info[5]

        ax.text(0.55, 0.1, r'$D^2$ = %.4f' % rms_weighted,
                style='normal', weight='bold', size=18, transform=ax.transAxes)
        ax.text(0.55, 0.15, r'$f_c$ = %.2f' % fc_weighted,
                style='normal', weight='bold', size=18, transform=ax.transAxes)

        for station in fc_all:
            nearest_idx = np.argmin(abs(freq_by_sta[station] - fc_all[station]))
            value = ratio_by_sta[station][nearest_idx]
            ax.loglog(fc_all[station], value, marker="*", color='orangered',
                      markersize=15, label='f$_c$' if station == list(fc_all)[-1] else "")

        # Inset barplot
        inset_ax = inset_axes(ax, width="30%", height="25%", loc=3, borderpad=6)
        x_range = max(fc_all.values()) - min(fc_all.values())
        bar_width = 0.05 * x_range
        inset_ax.bar(fc_all.values(), var_all.values(), width=bar_width,
                     color='#1f77b4', alpha=0.7)
        inset_ax.set_xlabel('Corner Frequency (Hz)', fontsize=18, fontweight='bold')
        inset_ax.set_ylabel('RMS', fontsize=18, fontweight='bold')
        inset_ax.set_yticks(np.linspace(0, 0.6, num=4))
        inset_ax.tick_params(axis='both', which='both', length=4)
        ax.legend(loc='upper center', ncol=3, prop={'size': 20})

    ax.set_xlabel('Frequency (Hz)', fontsize=26, fontweight='bold')
    ax.set_ylabel('Spectral Ratio', fontsize=26, fontweight='bold', labelpad=0)
    ax.tick_params(axis='both', which='both', length=5, labelsize=22)
    ax.legend(loc='upper center', ncol=3, prop={'size': 20})
    
    return rms_weighted if summary_type.lower() == 'weighted' else y_vals[best_idx]

def create_spectral_ratio_plots(paths, stations, s_arrivals, p_arrivals, origin_times, min_magnitude_diff, remove_response, apply_fs_correction,
                                 vp_fs, vs_fs, lat, lon, depth, network, time_offset, window_length, coda_length, taper_count,
                                 overlap_ratio, worker_count, model, egf_ids, mode, stack_type, all_residuals, fit_results, quality_factor,
                                 channel, main_event_name, egf_event_name, spectral_ratios, freq_data, wmfc_data, wm_data, wmn_data,
                                 wefc_data, we_data, wen_data, included_stations, time_window, main_event_files,
                                 egf_event_files, wave_type, magnitude_diff_threshold, travel_times):
    """
    Generate and organize spectral ratio plots with subplots arranged for waveform and spectral comparisons.
    
    Returns:
        fig (matplotlib.figure.Figure): Main figure object
        ax_spec (matplotlib.axes.Axes): Spectrum ratio axis
        popt (list): Spectrum fit parameters
    """
    style_list = ['-', '--', '-.', ':', '.']
    station_keys = list(spectral_ratios.keys())
    color_scheme = generate_color_list(len(station_keys))
    
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    fig1 = plt.figure(figsize=(9, 16))
    fig1.subplots_adjust(hspace=0.5)
    gs1 = fig1.add_gridspec(3, 1, left=0.2)
    
    Ptime, Stime, popt= None, None, None
    if included_stations:
        z = 0
        while z <= len(station_keys) - 1:
            try:
                sta_id = station_keys[z].split('(')[0]
    
                if egf_event_files.get(sta_id):
                    st = read(egf_event_files[sta_id])
                    if s_arrivals:
                        Stime = s_arrivals.get(st[0].stats.station.strip(), {}).get(egf_event_name)
                    if p_arrivals:
                        Ptime = p_arrivals.get(st[0].stats.station.strip(), {}).get(egf_event_name)
    
                    orig_time = origin_times[st[0].stats.station.strip()][egf_event_name]
                    baz = ev_baz(paths, lat, lon, st, egf_event_name)
                    _, _, noise_start, stn, _, st_time, _, _ = extract_signal_and_noise(
                        paths, remove_response, apply_fs_correction,
                        vp_fs, vs_fs, lat, lon, depth, time_offset,
                        coda_length, window_length, egf_event_files[sta_id],
                        egf_event_name, orig_time, Ptime, Stime,
                        time_window, True, wave_type, True, baz, chan=channel)
    
                    if egf_event_name and mode == 0:
                        axs = fig.add_subplot(gs[0, 1])
                        plot_event_waveform(window_length, coda_length, taper_count,
                                      overlap_ratio, egf_ids, stn, st_time, noise_start,
                                      Ptime, Stime, orig_time, time_window, egf_event_name,
                                      axs, wave_type)
    
                st = read(main_event_files[sta_id])
                if s_arrivals:
                    Stime = s_arrivals.get(st[0].stats.station.strip(), {}).get(main_event_name)
                if p_arrivals:
                    Ptime = p_arrivals.get(st[0].stats.station.strip(), {}).get(main_event_name)
    
                orig_time = origin_times[st[0].stats.station.strip()][main_event_name]
                baz = ev_baz(paths, lat, lon, st, main_event_name)
                _, _, noise_start, stn, _, st_time, _, _ = extract_signal_and_noise(
                    paths, remove_response, apply_fs_correction, vp_fs, vs_fs, lat, lon, depth,
                    time_offset, coda_length, window_length, main_event_files[sta_id],
                    main_event_name, orig_time, Ptime, Stime, time_window,
                    True, wave_type, True, baz, chan=channel)
    
                if egf_event_name:
                    axs = fig.add_subplot(gs[0, :] if mode == 1 else gs[0, 0])
                    plot_event_waveform(window_length, coda_length, taper_count,
                                  overlap_ratio, egf_ids, stn, None, noise_start,
                                  Ptime, Stime, orig_time, time_window, main_event_name,
                                  axs, wave_type)
                else:
                    axs1 = fig1.add_subplot(gs1[0, 0])
                    plot_event_waveform(window_length, coda_length, taper_count,
                                  overlap_ratio, egf_ids, stn, None, noise_start,
                                  Ptime, Stime, orig_time, time_window, main_event_name,
                                  axs1, wave_type)
    
                z = len(station_keys)
            except:
                # traceback.print_exc()
                z += 1
    
    ax_spec = fig.add_subplot(gs[1:3, 1])
    ax_compare = fig.add_subplot(gs[1:3, 0])
    max_y, min_y = 1, 1
    stations_used, egfs_used, stations1, all_y_max, all_y_min = [], [], [], 1, 1
    
    for idx in range(len(station_keys)):
        station, egf = station_keys[idx].split('(')[0], egf_event_name
        if egf:
            if '(' in station_keys[idx]:
                egf = station_keys[idx].split('(')[1].split(')')[0]
        
            sr = spectral_ratios[station_keys[idx]]
            y_max, y_min = max(sr[:len(sr) // 4]), min(sr[3 * (len(sr) // 4):])
            all_y_max, all_y_min = max(all_y_max, y_max), min(all_y_min, y_min)
        
            station_idx = stations.index(station)
        
            if stack_type.lower() != 'weighted':
                ax_spec.loglog(freq_data[station_keys[idx]], sr, linewidth=1, color=color_scheme[station_idx], label=station, alpha=0.5)
        
            elif stack_type.lower() == 'weighted' and len(egf_ids) <= len(style_list):
                if station_keys[idx] in fit_results:
                    i = egf_ids.index(egf)
                    if station not in stations_used:
                        stations_used.append(station)
                        ax_spec.loglog(freq_data[station_keys[idx]], sr, linestyle='-', linewidth=1, color=color_scheme[station_idx], label=station)
                    elif egf not in egfs_used:
                        egfs_used.append(egf)
                        ax_spec.loglog(freq_data[station_keys[idx]], sr, linestyle=style_list[i], linewidth=1, color=color_scheme[station_idx])
                    else:
                        ax_spec.loglog(freq_data[station_keys[idx]], sr, linestyle=style_list[i], linewidth=1, color=color_scheme[station_idx])
        
            elif stack_type.lower() == 'weighted' and len(egf_ids) > len(style_list):
                if station_keys[idx] in fit_results:
                    if station not in stations_used:
                        stations_used.append(station)
                        ax_spec.loglog(freq_data[station_keys[idx]], sr, linestyle='-', linewidth=1, color=color_scheme[station_idx], label=station)
                    else:
                        ax_spec.loglog(freq_data[station_keys[idx]], sr, linestyle='-', linewidth=1, color=color_scheme[station_idx])
        
            if station in included_stations:
                x_start = freq_data[station_keys[idx]][1] - 0.1 if freq_data[station_keys[idx]][0] == 0 else freq_data[station_keys[idx]][0] - 0.1
                x_end = max(freq_data[station_keys[idx]]) + 5
        
                try:
                    if (len(egf_ids) == 1 and egf_ids[0]) or (len(egf_ids) > 1 and mode == 0):
                        x2, y2, y22 = wefc_data[station_keys[idx]], we_data[station_keys[idx]], wen_data[station_keys[idx]]
                    else:
                        x2, y2, y22 = [], [], []
                except:
                    x2, y2, y22 = [], [], []
        
                stations1, y_max_temp, y_min_temp = plot_station_spectrum(station_keys,egf_ids,wmfc_data[station],wm_data[station],wmn_data[station],ax_compare,station,color_scheme[station_idx],
                    x_start,x_end,wave_type,freq_egf=x2,amp_egf=y2,noise_egf=y22,max_y_global=max_y,
                    station_list=stations1, egf=egf)
                max_y = max(max_y, y_max_temp)
                min_y = min(min_y, y_min_temp)
        
                ax_spec.set_xlim([x_start, x_end])
                ax_spec.set_ylim(10 ** np.floor(np.log10(all_y_min)), 10 ** (1 + np.ceil(np.log10(all_y_max))))
    
        else:
            ax_spec1 = fig1.add_subplot(gs1[1:3, 0])
            ax_spec1, popt = plot_individual_spectra_fits(worker_count, model, fig1, ax_spec1, wm_data, wmfc_data, wmn_data, travel_times, wave_type, 'yes', stack_type, quality_factor)
            ax_spec1.set_xlabel('Frequency (Hz)', fontsize=26, weight='bold')
            ax_spec1.set_ylabel('Amplitude (nm/Hz)', fontsize=26, weight='bold')
            fig = fig1
            ax_spec = ax_spec1
    
    return fig, ax_spec, popt

def export_figure(config, win_len, t_coda, magnitude, egf_id, model, figure, plot_type, station_id, wave_type, event_type, *options):
    """
    Save figure according to type and metadata.

    Parameters:
    ------------
    config       : dict, contains output directory path with key 'out_path'
    win_len      : float/int, waveform time window length
    t_coda       : float/int, coda window length
    magnitude    : float/int, event magnitude
    egf_id       : str, EGF event identifier
    model        : str, source model name
    figure       : matplotlib figure object
    plot_type    : str, type of plot: 'ind', 'spec', or 'stf'
    station_id   : str, station identifier
    wave_type    : str, waveform type, e.g., 'P' or 'S'
    event_type   : str, e.g., target event name
    options      : additional parameters (tuple), used for file naming
    
    Returns:
    --------
    None (figure saved to disk)
    """

    output_dir = config['out_path']
    folder = os.path.join(output_dir, event_type)
    os.makedirs(folder, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

    if plot_type == 'ind':
        figure.subplots_adjust(hspace=0.4, wspace=0.2)
        filename = f"(Single){event_type}_sinspec_{win_len}s_{model}_{wave_type.upper()}.pdf"
        path = os.path.join(folder, filename)
        figure.savefig(path, format='pdf', dpi=600)

    elif plot_type == 'spec':
        figure.subplots_adjust(hspace=0.4, wspace=0.2)
        if options[1] == 0:
            filename = f"(EGF){event_type}_{options[0]}_specratio_{win_len}_{model}_{wave_type.upper()}.pdf"
        elif options[1] == 1:
            filename = f"(EGFs){event_type}_{egf_id}_specratio_{win_len}_{model}_{wave_type.upper()}.pdf"
        path = os.path.join(folder, filename)
        figure.savefig(path, format='pdf', dpi=300)

    elif plot_type == 'stf':
        filename = f"{timestamp}_STF_{event_type}_{model}_{wave_type.upper()}.pdf"
        path = os.path.join(folder, filename)
        figure.savefig(path, format='pdf', dpi=300)

    return None
