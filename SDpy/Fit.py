#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:57:59 2025

@author: Zhang et al.
"""
import numpy as np
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import traceback

def compute_spec_ratio(freq_bins, f_main, f_aux, moment_main, moment_aux, n_param, gamma_param):
    """
    Theoretical spectral ratio model (simplified and vectorized).

    Parameters:
    -----------
    freq_bins     : array-like, frequency bins
    f_main        : float, corner frequency of main event
    f_aux         : float, corner frequency of EGF
    moment_main   : float, seismic moment of main event
    moment_aux    : float, seismic moment of EGF
    n_param       : float, spectral decay exponent (n)
    gamma_param   : float, high-frequency fall-off control (γ)

    Returns:
    --------
    model_out     : array-like, theoretical spectral ratio model
    """
    # Convert inputs to NumPy arrays
    freq_bins = np.asarray(freq_bins, dtype=float)

    # Compute frequency ratios
    r1 = (freq_bins / f_main) ** (gamma_param * n_param)
    r2 = (freq_bins / f_aux) ** (gamma_param * n_param)

    # Compute spectral ratio model
    ratio = ((1 + r2) / (1 + r1)) ** (1 / gamma_param)
    model_out = (moment_main / moment_aux) * ratio

    return model_out

def compute_single_spectrum(freqs, moment, fc, n_exp, gamma_exp, Q, travel_t):
    """
    Single-event source spectrum model.

    Parameters:
    -----------
    freqs      : array-like, frequency bins
    moment     : float, seismic moment (M0)
    fc         : float, corner frequency (Hz)
    n_exp      : float, spectral decay exponent n
    gamma_exp  : float, high-frequency roll-off gamma
    Q          : float, quality factor
    travel_t   : float, wave travel time (s)

    Returns:
    --------
    spectrum_model : array-like, theoretical single-event spectrum
    """
    
    # Attenuation term: exp(-πfT/Q)
    decay_factor = np.exp(-np.pi * freqs * travel_t / Q)

    # Numerator: moment * attenuation
    numerator = moment * decay_factor

    # Denominator: [1 + (f/fc)^(gamma * n)]^(1/gamma)
    denom = (1 + (freqs / fc) ** (gamma_exp * n_exp)) ** (1 / gamma_exp)

    spectrum_model = numerator / denom
    return spectrum_model

def D_squared(gamma,var,freqmain=None,specmain=None,max_var=0.3):
    # print(gamma,var)
    if gamma and var:
        if type(gamma)==dict and type(var)==dict:
            # fc_median=np.median(list(gamma.values()))
            
            for i in list(gamma.keys()):
                # print(freqmain[i],'\n',gamma[i])
                try:
                    a=np.where(freqmain[i]<=gamma[i])[0][-1]
                except:
                    a=np.where(freqmain[i]>=gamma[i])[0][0]
                if var[i]>max_var or (np.mean(specmain[i][:a])<2*np.mean(specmain[i][a:])):#(fc_median*2.8 < gamma[i] or gamma[i] < fc_median*0.8):
                    del gamma[i]
                    del var[i]
            
            fc=list(gamma.values());variance=list(var.values())    
            D_squared=1/sum(1/i**1 for i in variance if i != 0)
            mean_gamma=sum(fc[i]/variance[i]**1 for i in range(len(fc)) if variance[i] != 0)*D_squared
            
        if type(gamma)==list and type(var)==list:
            # fc_median=np.median(gamma)
            gamma=[gamma[i] for i in range(len(gamma)) if var[i]<=max_var]# and (fc_median*2.5 >= fc[i] >= fc_median*0.5)]
            var=[var[i] for i in range(len(var)) if var[i]<=max_var]# and (fc_median*2.5 >= fc[i] >= fc_median*0.5)]
            
            fc=gamma;variance=var
            D_squared=1/sum(1/variance[i]**2 for i in range(len(variance)))
            mean_gamma=sum(fc[i]/variance[i]**2 for i in range(len(fc)))*D_squared
            
    else:
        mean_gamma,D_squared,fc,variance=[],[],[],[]
    return mean_gamma,D_squared,gamma,var

def determine_asymptote_bounds(freq_bins,spectral_data):
    """
    Description:
    ------------
    Determine the bounds for low frequency asymptotes in spectral ratio analysis

    Input:
    -----------------
    spectral_data: Spectral ratio values

    Returns:
    ----------------------
    upper_bound_major    --> Upper bound for larger event asymptote
    lower_bound_major    --> Lower bound for larger event asymptote
    upper_bound_minor    --> Upper bound for smaller event asymptote
    lower_bound_minor    --> Lower bound for smaller event asymptote
    """
    bounded_region = []
    index=np.where(freq_bins >= (0.5*freq_bins[-1]))[0][-1]
    try:
        threshold_index = np.where(spectral_data >= (max(spectral_data[:index]) * 1))[0][-1]
        bounded_region = spectral_data[slice(0, threshold_index)]
    except:    
        try:
            threshold_index = np.where(spectral_data <= (max(spectral_data[:index]) * 1))[0][0]
            bounded_region = spectral_data[slice(0, threshold_index)]
            if len(bounded_region) == 0:
                bounded_region = spectral_data
        except:
            bounded_region = spectral_data
    if len(bounded_region) == 0:
        bounded_region = spectral_data
    # lower_bound_major = max(bounded_region) * 0.9
    # upper_bound_major = max(bounded_region) * 1.3
    # lower_bound_minor = min(spectral_data) * 0.6
    # upper_bound_minor = min(spectral_data) * 1.2
    lower_bound_minor = 0.9
    upper_bound_minor = 1.1
    lower_bound_major = max(bounded_region)*lower_bound_minor
    upper_bound_major = max(bounded_region)*upper_bound_minor
    
    return upper_bound_major, lower_bound_major, upper_bound_minor, lower_bound_minor

def optimize_corner_freq(freq_bins, spec_ratio, scale, freq_range, calc_freq, init_guess, fc_min, fc_max, model, **kwargs):
    """
    Refines and determines the optimal corner frequency through spectral ratio fitting.

    Args:
        freq_bins: Frequency bins for spectral ratio.
        spec_ratio: Spectral ratio values.
        scale: Scaling factor for spectral ratio.
        freq_range: Range of possible corner frequencies.
        calc_freq: Flag for frequency calculation.
        init_guess: Initial guess for corner frequency.
        fc_min: Minimum bound for corner frequency.
        fc_max: Maximum bound for corner frequency.
        model: Type of source model ('B', 'FB', 'SM').
        **kwargs: Additional optional parameters.

    Returns:
        norm_resid: Normalized RMS of model fit.
        opt_params: Optimized model parameters (fc1, fc2, omega, n, gamma).
        param_cov: Covariance matrix of fitted parameters.
        spec_copy: Copy of input spectral ratio.
    """
    spec_copy = spec_ratio.copy()
    spec_ratio = np.multiply(spec_ratio, scale, dtype=float)
    upper_freq = fc_min * 2
    ub_major, lb_major, ub_minor, lb_minor = determine_asymptote_bounds(freq_bins,spec_ratio)
    
    # Set model-specific exponents
    if model.upper() == 'B':
        n1, n2, n3, n4 = 2, 1, 2, 1
    elif model.upper() == 'FB':
        n1, n2, n3, n4 = 2, 2, 2, 2
    elif model.upper() == 'SM':
        n1, n2, n3, n4 = 1.5, 1, 3, 2
    
    # Perform curve fitting
    if calc_freq:
        opt_params, param_cov = curve_fit(
            compute_spec_ratio,
            freq_bins,
            spec_ratio,
            method='trf',
            bounds=((fc_min, upper_freq, lb_major, lb_minor, n1-0.01, n2-0.01),
                    (fc_max, max(freq_bins), ub_major, ub_minor, n3, n4)),
            loss='soft_l1',
            verbose=0,
            tr_solver='lsmr',
            maxfev=10000000)
    else:
        if init_guess:
            opt_params, param_cov = curve_fit(
                compute_spec_ratio,
                freq_bins,
                spec_ratio,
                method='trf',
                bounds=((init_guess*0.95, upper_freq, lb_major, lb_minor, n1-0.01, n2-0.01),
                        (init_guess*1.05, max(freq_bins), ub_major, ub_minor, n3, n4)),
                loss='soft_l1',
                verbose=0,
                tr_solver='lsmr',
                maxfev=10000000)
        else:
            opt_params, param_cov = curve_fit(
                compute_spec_ratio,
                freq_bins,
                spec_ratio,
                method='trf',
                bounds=((fc_min, upper_freq, lb_major, lb_minor, n1-0.01, n2-0.01),
                        (fc_max, max(freq_bins), ub_major, ub_minor, n3, n4)),
                loss='soft_l1',
                verbose=0,
                tr_solver='lsmr',
                maxfev=10000000)
    
    # Calculate residuals
    resid = np.power(np.subtract(spec_ratio, compute_spec_ratio(freq_bins, *opt_params)), 2)
    norm_resid = np.sqrt(np.sum(resid) / np.sum(np.power(spec_ratio, 2)))
    
    return norm_resid, opt_params, param_cov, spec_copy

def fit_spectrum_model(amps, freqs, station, fc_min, fc_max, ttravel, review_mode, model_type, q_factor):
    """
    Fits a source model to a single spectrum using least-squares optimization.

    Args:
        amps: Spectrum amplitude values
        freqs: Corresponding frequency bins
        station: Station identifier
        fc_min: Minimum corner frequency bound
        fc_max: Maximum corner frequency bound
        ttravel: Travel time
        review_mode: Whether to automatically review fits ('yes'/'no')
        model_type: Source model type ('B', 'FB', or 'SM')
        q_factor: Q value or range [Q_min, Q_max]

    Returns:
        params: Optimized model parameters
        cov: Parameter covariance matrix
        resid: Normalized residual of fit
    """
    params = [None]
    cov = [None]
    
    # Set model exponents
    if model_type.upper() == 'B':
        n = [2, 1, 2, 1]  # n1,n2,n3,n4
        n_range1 = np.arange(n[0], n[0]+1, 1)
        n_range2 = np.arange(n[1], n[1]+1, 1)
    elif model_type.upper() == 'FB':
        n = [2, 2, 2, 2]
        n_range1 = np.arange(n[0], n[0]+1, 1)
        n_range2 = np.arange(n[1], n[1]+1, 1)
    elif model_type.upper() == 'SM':
        n = [1.5, 1, 3, 2]
        n_range1 = np.arange(n[0], n[2], 0.2)
        n_range2 = np.arange(n[1], n[3], 0.2)
    else:
        raise ValueError("Model must be 'B', 'FB', or 'SM'")

    # Set review mode
    review_mode = 'AUTO' if review_mode.lower() == 'yes' else 'MANUAL'

    # Set Q bounds
    q_min, q_max = 200., 2000.
    if isinstance(q_factor, (int, float)):
        q_min = q_factor - 1
        q_max = q_factor + 1
    elif isinstance(q_factor, list):
        q_min, q_max = q_factor[0], q_factor[1]

    # Find low-frequency bound region
    data, bound_region = amps, np.array([])
    try:
        if min(freqs) < 1:
            idx = np.where(freqs >= 0.8)[0][0]
        else:
            idx = np.where(freqs >= (min(freqs)+2))[0][0]
        bound_region = data[:idx]
    except:
        pass

    # Set amplitude bounds
    amp_low = np.median(bound_region)*0.8 if bound_region.any() else max(data)*0.95
    amp_high = amp_low*1.5

    # Fit loop
    approved = False
    while not approved:
        for i in n_range1:
            for j in n_range2:
                if not approved:
                    if review_mode == 'AUTO':
                        approved = True
                    else:
                        n[0], n[1] = i, j
                    
                    try:
                        params, cov = curve_fit(
                            compute_single_spectrum,
                            freqs[:len(data)],
                            data,
                            method='trf',
                            absolute_sigma=True,
                            bounds=(
                                (amp_low, fc_min, n[0]-0.01, n[1]-0.01, q_min, ttravel-0.0001),
                                (amp_high, fc_max, n[2], n[3], q_max, ttravel)
                            ),
                            maxfev=10000000
                        )
                        resid = np.sum((data - compute_single_spectrum(freqs[:len(data)], *params))**2)
                        resid = np.sqrt(resid/np.sum(data**2))
                    except:
                        traceback.print_exc()
                        params, cov, resid = None, None, None

                    
    return params, cov, resid

def parallel_spectrum_fit(spectrum, freqs, station, fc_low, fc_high, travel_time, 
                         model_type, Q_value, num_workers):
    '''
    Parallel fitting of spectrum model with parameter optimization.
    
    Uses parallel computation to test different n/gamma combinations and
    selects the best fit based on normalized RMS.

    Args:
        spectrum: Spectral amplitude values
        freqs: Frequency bins
        station: Station identifier
        fc_low: Lower bound for corner frequency
        fc_high: Upper bound for corner frequency
        travel_time: Event travel time
        model_type: Source model type (only 'sm' supported)
        Q_value: Q factor (single value or [min,max])
        num_workers: Number of parallel workers

    Returns:
        best_params: Optimal parameters
        best_cov: Covariance matrix
        fc_values: Tested corner frequencies
        residuals: Fit residuals
    '''
    best_params, best_cov = None, None
    residuals_list, fc_list = [], []

    if model_type.lower() == 'sm':
        # Set Q bounds
        Q_min, Q_max = 200., 2000.
        if isinstance(Q_value, (int, float)):
            Q_min = Q_value - 1
            Q_max = Q_value + 1
        elif isinstance(Q_value, list):
            Q_min, Q_max = Q_value[0], Q_value[1]

        # Set amplitude bounds
        low_freq_data = spectrum[:4]
        if low_freq_data.any():
            amp_low = np.median(low_freq_data) * 0.9
        else:
            amp_low = max(spectrum) * 0.95
        amp_high = amp_low * 1.5

        # Parameter ranges
        n1_range = np.linspace(1.5, 2.5, 11)
        n2_range = np.linspace(1.0, 2.0, 11)
        freqs = freqs[:len(spectrum)]

        # Parallel fitting
        params_list, cov_list, resid_list = [], [], []
        for i in range(len(n1_range)):
            results = Parallel(n_jobs=num_workers)(
                delayed(parallel_fit_helper)(
                    freqs, spectrum, amp_low, amp_high, 
                    fc_low, fc_high, travel_time,
                    Q_min, Q_max, n1_range[i], n2_range[j]
                ) for j in range(len(n2_range))
            )
            
            for result in results:
                if result[0] is not None:
                    params_list.append(result[0])
                    cov_list.append(result[1])
                    resid_list.append(result[2])

        # Select best fit
        if params_list:
            for params in params_list:
                resid = np.sum((spectrum - compute_single_spectrum(freqs, *params))**2)
                norm_resid = np.sqrt(resid / np.sum(spectrum**2))
                residuals_list.append(norm_resid)
                fc_list.append(params[1])
            
            best_idx = np.argmin(np.array(residuals_list))
            best_params = np.asarray(params_list)[best_idx]
            best_cov = cov_list[best_idx]
            residuals = resid_list[best_idx]
    else:
        raise ValueError("Parallel fitting only supported for 'sm' model type")

    return best_params, best_cov, fc_list, residuals

def parallel_fit_helper(freqs, spectrum, amp_min, amp_max, fc_min, fc_max, 
                      travel_time, Q_min, Q_max, n1, n2):
    """
    Helper function for parallel spectrum fitting.
    
    Performs a single curve fit with given parameters and returns results.
    
    Args:
        freqs: Frequency bins
        spectrum: Spectral amplitude values
        amp_min: Minimum amplitude bound
        amp_max: Maximum amplitude bound
        fc_min: Minimum corner frequency
        fc_max: Maximum corner frequency  
        travel_time: Event travel time
        Q_min: Minimum Q value
        Q_max: Maximum Q value
        n1: First spectral index
        n2: Second spectral index
        
    Returns:
        params: Fitted parameters (or None if failed)
        cov: Covariance matrix (or None if failed) 
        resid: Normalized residual (or None if failed)
    """
    try:
        params, cov = curve_fit(
            compute_single_spectrum,
            freqs,
            spectrum,
            method='trf',
            absolute_sigma=True,
            bounds=((amp_min, fc_min+0.01, n1-0.01, n2-0.01, Q_min, travel_time-0.0001),
                    (amp_max, fc_max, 3.0, 2.0, Q_max, travel_time)),
            max_nfev=10000000)
        
        residual = np.sum((spectrum - compute_single_spectrum(freqs, *params))**2)
        norm_residual = np.sqrt(residual / np.sum(spectrum**2))
        
    except:
        params, cov, norm_residual = None, None, None
        
    return params, cov, norm_residual

def find_optimal_fit(model_type, num_workers, source_params, freq_bins, spec_ratio, 
                    fc_min, fc_max, **kwargs):
    """
    Finds the best model fit for spectral ratio data using parallel optimization.
    
    Args:
        model_type: Source model type
        num_workers: Number of parallel workers
        source_params: DataFrame of source parameters
        freq_bins: Frequency bins array
        spec_ratio: Spectral ratio values
        fc_min: Minimum corner frequency
        fc_max: Maximum corner frequency
        **kwargs: Additional parameters including:
            - sumtype: 'weighted' or other fitting type
            - mode: Analysis mode
            - freqmains: Frequency data for weighted fitting
            - specmains: Spectral data for weighted fitting
    
    Returns:
        - perturbed_freqs: Array of tested frequencies
        - best_freq: Optimal corner frequency
        - best_multiplier: Optimal scaling multiplier
        - residuals: Fitting residuals
        - best_params: Optimal model parameters
        - best_cov: Parameter covariance matrix
        - fc1_dict: Dictionary of corner frequencies (weighted mode)
        - fc2_dict: Dictionary of secondary frequencies (weighted mode)
        - var_dict: Dictionary of variances (weighted mode)
    """
    # print()
    fc_min = freq_bins[0];fc_max = freq_bins[-1]
    # Initialize result containers
    residuals = []
    multipliers = np.arange(0.04, 4.0, 0.04)
    fc1_dict = {}
    fc2_dict = {}
    var_dict = {}
    
    # Weighted fitting mode
    if kwargs.get('sumtype', '').lower() == 'weighted':
        freq_data = kwargs['freqmains']
        spec_data = kwargs['specmains']
        station_keys = list(spec_data.keys())
        
        # First pass: Find best multipliers
        fit_results = Parallel(n_jobs=num_workers)(
            delayed(optimize_corner_freq)(
                freq_data[station],
                spec_data[station],
                multiplier,
                freq_range=None,
                calc_freq=False,
                init_guess=None,
                fc_min=fc_min,
                fc_max=fc_max,
                model=model_type,
                mode=kwargs['mode'],
                sumtype=kwargs['sumtype']
            )
            for station in station_keys
            for multiplier in multipliers
        )
        
        # Process results to find best multipliers
        if fit_results:
            best_multipliers = {}
            for i, station in enumerate(station_keys):
                station_residuals = [res[0] for res in fit_results if np.array_equal(res[3], spec_data[station])]
                best_idx = np.argmin(station_residuals)
                best_multipliers[station] = multipliers[best_idx]
    
    # Standard fitting mode
    else:
        fit_results = Parallel(n_jobs=num_workers)(
            delayed(optimize_corner_freq)(
                freq_bins,
                spec_ratio,
                multiplier,
                freq_range=None,
                calc_freq=False,
                init_guess=None,
                fc_min=fc_min,
                fc_max=fc_max,
                model=model_type,
                mode=kwargs.get('mode'),
                sumtype=kwargs.get('sumtype'))
            for multiplier in multipliers)
        
        if fit_results:
            residuals = [res[0] for res in fit_results]
            best_idx = np.argmin(residuals)
            best_multiplier = multipliers[best_idx]
    
    # Frequency perturbation stage
    test_freqs = np.linspace(fc_min, fc_max, num=len(multipliers))
    
    if kwargs.get('sumtype', '').lower() == 'weighted':
        best_freq = None
        best_params = np.array([])
        best_cov = None
        # Weighted frequency perturbation
        perturb_results = Parallel(n_jobs=num_workers)(
            delayed(optimize_corner_freq)(
                freq_data[station],
                spec_data[station],
                best_multipliers[station],
                freq_range=None,
                calc_freq=False,
                init_guess=test_freqs[m],
                fc_min=fc_min,
                fc_max=fc_max,
                model=model_type,
                mode=kwargs['mode'],
                sumtype=kwargs['sumtype'])
            for station in station_keys
            for m in range(len(test_freqs)))
        
        # Process perturbation results
        if perturb_results:
            for i, station in enumerate(station_keys):
                station_residuals = [res[0] for res in perturb_results 
                                  if np.array_equal(res[3], spec_data[station])]
                station_freqs = [res[1][0] for res in perturb_results 
                               if np.array_equal(res[3], spec_data[station])]
                
                best_idx = np.argmin(station_residuals)
                fc1_dict[station] = station_freqs[best_idx]
                var_dict[station] = min(station_residuals)
                
                if kwargs.get('mode') == 0:
                    try:
                        secondary_freqs = [res[1][1] for res in perturb_results[i*len(test_freqs):(i+1)*len(test_freqs)]]
                        fc2_dict[station] = secondary_freqs[best_idx]
                    except:
                        pass
    
    else:
        # Standard frequency perturbation
        perturb_results = Parallel(n_jobs=num_workers)(
            delayed(optimize_corner_freq)(
                freq_bins,
                spec_ratio,
                best_multiplier,
                freq_range=None,
                calc_freq=False,
                init_guess=test_freqs[m],
                fc_min=fc_min,
                fc_max=fc_max,
                model=model_type,
                mode=kwargs.get('mode'),
                sumtype=kwargs.get('sumtype'))
            for m in range(len(test_freqs)))
        
        if perturb_results:
            residuals = [res[0] for res in perturb_results]
            best_idx = np.argmin(residuals)
            best_freq = [res[1][0] for res in perturb_results][best_idx]
            best_params = perturb_results[best_idx][1]
            best_cov = perturb_results[best_idx][2]
    
    # print('')
    return test_freqs,best_freq,best_multiplier if 'best_multiplier' in locals() else None,\
        residuals if residuals else None,best_params if 'best_params' in locals() else np.array([]),\
        best_cov if 'best_cov' in locals() else None,fc1_dict,fc2_dict,var_dict
