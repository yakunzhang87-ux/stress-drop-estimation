#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:57:59 2025

@author: Zhang et al.
"""
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
import traceback
import os
from .Preprocess import define_fcerr
import matplotlib.pyplot as plt
from functools import partial
import warnings

def compute_spec_ratio(freq_bins, f_main, f_aux, moment_r, n_param, gamma_param):
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
    f_main = f_main if f_main>0 else 1e-12
    f_aux = f_aux if f_aux>0 else 1e-12
    # Compute frequency ratios
    r1 = (freq_bins / (f_main)) ** (gamma_param * n_param)
    r2 = (freq_bins / (f_aux)) ** (gamma_param * n_param) 

    # Compute spectral ratio model
    ratio = ((1 + r2) / (1 + r1)) ** (1 / (gamma_param))
    model_out = moment_r * ratio

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
    denom = (1 + (freqs / (fc+1e-12)) ** (gamma_exp * n_exp)) ** (1 / (gamma_exp+1e-12))

    spectrum_model = numerator / denom
    return spectrum_model

def objective_function(params, freq_bins, observed_data, n, gamma, t):
    fc, omega_0, Q = params[0], params[1], params[2]
    params = [omega_0, fc, n, gamma, Q, t]
    model_data = compute_single_spectrum(freq_bins, *params)
    with np.errstate(divide='ignore', invalid='ignore'):
        warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
        resid = np.power(np.subtract(np.log10(observed_data), np.log10(model_data)), 2)
    norm_resid = np.sum(resid) 
    return norm_resid

def D_squared(gamma1,gamma2,uncertainties,var,popts = None,freqmain=None,specmain=None,max_var=0.6):
    try:
        if type(gamma1)==dict and type(var)==dict:
            fc_median=np.median(list(gamma1.values()))
            # fc_median=np.mean(list(gamma1.values()))
            for i in list(gamma1.keys()):
                if uncertainties[i]>1 or (not 0<var[i]<=0.005) or (not 0.7*fc_median<=gamma1[i]<=1.5*fc_median):
                    del gamma1[i]
                    del uncertainties[i]
                    del var[i]
                    if gamma2:
                        del gamma2[i]
            
            fc=list(gamma1.values());fc_errs=list(uncertainties.values());variance=list(var.values())
            if gamma2:
                fc2=list(gamma2.values());
            
        if type(gamma1)==list and type(var)==list:
            # fc_median=np.median(gamma)
            fc=[gamma1[i] for i in range(len(gamma1)) if 0<var[i]<=max_var]
            if gamma2:
                fc2=[gamma2[i] for i in range(len(gamma2)) if 0<var[i]<=max_var]
            fc_errs=[uncertainties[i] for i in range(len(uncertainties)) if 0<var[i]<=max_var]
            variance=[i for i in var if 0<i<=max_var]
        if not 'fc' in locals():
            print('#'*72,'\n','\t'*3,'There is no available measurement.\n','#'*72,sep='')
        # fc_err_squared=1/sum(1/i**2 for i in variance)
        D_squared=1/sum(1/i**1 for i in variance)
        fc_err=sum(fc_errs[i]/variance[i]**1 for i in range(len(fc)))*D_squared
        mean_gamma=sum(fc[i]/variance[i]**1 for i in range(len(fc)))*D_squared
        mean_gamma2=sum(fc2[i]/variance[i]**1 for i in range(len(fc2)))*D_squared if gamma2 else None
        mean_gamma,fc_err = round(mean_gamma,2),round(fc_err,2)
    except:
        traceback.print_exc()
        mean_gamma,mean_gamma2,fc_err,fc,variance=[],[],[],[],[]
    return mean_gamma,mean_gamma2,fc_err,fc,variance#gamma,var

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
    lower_bound_minor = 0.9
    upper_bound_minor = 1.1
    lower_bound_major = max(bounded_region)*lower_bound_minor
    upper_bound_major = max(bounded_region)*upper_bound_minor
    
    return upper_bound_major, lower_bound_major, upper_bound_minor, lower_bound_minor

def optimize_corner_freq(freq_bins, spec_ratio, nyquist_freq, station, model, path, fit_method='L-BFGS-B', fit_amp_ratio=2):
    """
    Refines and determines the optimal corner frequency through spectral ratio fitting.

    Args:
        freq_bins: Frequency bins for spectral ratio.
        spec_ratio: Spectral ratio values.
        scale: Scaling factor for spectral ratio.
        freq_range: Range of possible corner frequencies.
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
    freq_copy = freq_bins.copy()
    spec_copy = spec_ratio.copy()
    indices = np.where(spec_ratio<fit_amp_ratio)
    if len(indices[0])!=0:
        indice = indices[0][0]
    else:
        indice = len(freq_bins)
    freq_copy = freq_bins[:indice]
    spec_copy = spec_ratio[:indice]
    if len(freq_copy)==0:
        freq_copy = freq_bins
        spec_copy = spec_ratio
        indices1 = np.where(spec_ratio>=2)
        if len(indices1[0])!=0:
            indice1 = indices1[0][0]
            indice2 = indices1[0][-1]
            freq_copy = freq_bins[indice1:indice2]
            spec_copy = spec_ratio[indice1:indice2]
        else:
            freq_copy, spec_copy = [], []
    # Set model-specific exponents
    if len(freq_copy)!=0:
        if model.upper() == 'B':
            n1, n2, n3, n4 = 2, 1, 2, 1
        elif model.upper() == 'FB':
            n1, n2, n3, n4 = 2, 2, 2, 2
        elif model.upper() == 'SM':
            n1, n2, n3, n4 = 1.5, 1, 3, 2
        init_fc1, init_fc2, init_omega_0r = 0.9 * max(freq_copy), 0.1 * max(freq_copy), max(spec_copy) #0.05 * max(freq_copy),
        bounds = [(0,max(freq_bins)),(0,np.inf),(init_omega_0r*0.9,init_omega_0r*1.1),(n1-0.01,n3+0.01),(n2-0.01,n4+0.01)]
        # Perform curve fitting
        def model_fit(f, spec, fc1, fc2, omega_0r, n, gamma):
            
            model_func = partial(compute_spec_ratio, n_param=n, gamma_param=gamma)
            p0 = [fc1, fc2, omega_0r]
            bounds = [(0,max(freq_bins)),(0,np.inf),(init_omega_0r*0.9,init_omega_0r*1.1)]
            def residual_3param_nm(params):
                fc1, fc2 ,omega0= params
                model_vals = model_func(f, fc1, fc2,omega0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
                    resid = (np.log10(model_vals) - np.log10(spec))**2
                return np.sum(resid)
            
            if fit_method=='Nelder-Mead':
                result = minimize(residual_3param_nm, p0,method=fit_method,
                                  options=dict(maxiter=10000))
            else:
                result = minimize(residual_3param_nm,p0,method=fit_method,bounds=bounds,
                options=dict(maxiter=10000))
            
            fc1_opt, fc2_opt, omega0_opt = result.x
    
            min_res =result.fun
            min_var = min_res / (len(freq_copy) * omega0_opt+1e-12)
            
            return min_var, fc1_opt, fc2_opt, omega0_opt
        
        if n1==n3 and n2==n4:
            min_var, fc1_opt, fc2_opt, omega0_opt = model_fit(freq_copy, spec_copy, init_fc1, init_fc2, init_omega_0r, n1, n2)
            n, gamma = n1, n2
        
        else:
            min_vars, fc1_opts, fc2_opts, omega0_opts = [], [], [], []
            idxs = []
            for n in np.arange(n1,n3+0.1,0.1):
                for gamma in np.arange(n2,n4+0.1,0.1):
                    min_var, fc1_opt, fc2_opt, omega0_opt = model_fit(freq_copy, spec_copy, init_fc1, init_fc2, init_omega_0r, n, gamma)
                    min_vars.append(min_var)
                    fc1_opts.append(fc1_opt)
                    fc2_opts.append(fc2_opt)
                    omega0_opts.append(omega0_opt)
                    idxs.append([n,gamma])
                    
            min_var = min(min_vars)
            best_idx = min_vars.index(min_var)
            fc1_opt = fc1_opts[best_idx]
            fc2_opt = fc2_opts[best_idx]
            omega0_opt = omega0_opts[best_idx]
            n, gamma = idxs[best_idx]
            print(f'The best  parameter [n, gamma] of {station} is ',[round(n,1), round(gamma,1)])
            
        # fc1_scan = np.linspace(fc1_opt*0.1, fc1_opt*2, 200)
        fc1_scan = np.linspace(freq_copy[0], freq_copy[-1], 200)
        omega0_list = []
        fc1_list = []
        fc2_list = []
        res_list = []
        # res_var_list =[]
        model_func = partial(compute_spec_ratio, n_param=n, gamma_param=gamma)
        for fc1_fixed in fc1_scan:
            def residual_fc1(params):
                fc2, omega0 = params
                model_vals = model_func(freq_copy, fc1_fixed, fc2, omega0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
                    resid = np.sum((np.log10(model_vals+1e-12) - np.log10(spec_copy+1e-12))**2)
                return resid
            # 初值
            p0_fc2 = [0.1*max(freq_copy),max(spec_copy)]
            bounds = [(0,np.inf),(init_omega_0r*0.9,init_omega_0r*1.1)]
            if fit_method=='Nelder-Mead':
                result_fc2 = minimize(residual_fc1, p0_fc2,method=fit_method,options=dict(maxiter=10000))
            else:
                result_fc2 = minimize(residual_fc1,p0_fc2,method=fit_method,bounds=bounds,
                options=dict(maxiter=10000))
            fc2_fit,omega0_fit = result_fc2.x
            resid = result_fc2.fun
            if not np.isnan(resid):
                omega0_list.append(omega0_fit)
                fc1_list.append(fc1_fixed)
                fc2_list.append(fc2_fit)
                # var = resid / (len(freq_bins) * omega0_fit)
                res_list.append(resid)
                # res_var_list.append(var)
    
        res_array = np.array(res_list)
        res_var_array = res_array / (len(freq_copy) * omega0_opt) # 
        best_var = min_var
        best_fc1 = fc1_list[np.argmin(res_var_array)]#fc1_opt
        best_fc2 = fc2_list[np.argmin(res_var_array)]#fc2_opt
        best_omega0 = omega0_list[np.argmin(res_var_array)]#omega0_opt
     
        fc1_uncertainty, fc1_upper, fc1_lower = define_fcerr(fc1_list, res_var_array)
        # fc1_uncertainty = (fc1_upper-fc1_lower)/best_fc1
        # --- plot figure ---
        os.makedirs(path, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
        # left: omega0 (blue) & fc2 (red) vs fc1
        ax1 = axes[0]
        ax1.scatter(fc1_list, omega0_list, marker='o', color='blue', label='Ω0')
        ax1.set_xlabel('Large Eq. Corner Freq. (Hz)')
        ax1.set_ylabel('Ω0', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)
    
        ax2 = ax1.twinx()
        ax2.scatter(fc1_list, fc2_list, marker='o', color='red', label='fc2')
        ax2.set_ylabel('Small Eq. Corner Freq. (Hz)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        # ax1.set_xlim(fc1_lower*0.9, fc1_upper*1.1)
        # right: normalized variance
        ax3 = axes[1]
        ax3.scatter(fc1_list, res_var_array/min(res_var_array), marker='o', color='C0')
        ax3.axhline(1.05, color='black', linestyle='-')
        ax3.plot(best_fc1, 1, 'k*', markersize=12)
        ax3.set_xlabel('Large Eq. Corner Freq. (Hz)')
        ax3.set_ylabel('Normalized Variance')
        ax3.grid(True)
        # ax3.set_xlim(fc1_lower*0.9, fc1_upper*1.1)
        ax3.set_ylim(0.99,1.2)
        plt.tight_layout()
        plt.savefig(f'{path}/{station}_fc1_scan_results.png', dpi=300)
        fit_curve = model_func(freq_copy, best_fc1, best_fc2,best_omega0)
        plt.figure()
        plt.plot(freq_copy, spec_copy, '-', label='Observed')
        plt.plot(freq_copy, fit_curve, '--',linewidth=3, label='Fitted model')
        plt.text(0.02, 0.02, f'fc1={best_fc1:.2f} Hz\nfc2={best_fc2:.2f} Hz\nuncertainty={fc1_uncertainty:.2f}\nVar={best_var:.4f}',
             transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='left')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Frequency (Hz)'); plt.ylabel('Spectral ratio')
        plt.legend(); plt.grid(True);
        plt.savefig(f'{path}/{station}_fitted_spectral_ratio.png', dpi=300)
        plt.close(fig)
        outresult= {station: {"var": res_var_array, "popt": [best_fc1, best_fc2, best_omega0, n, gamma],"fc1_uncertainty":fc1_uncertainty}}
        # print(outresult)
    else:
        outresult={station:{}}
        fc1_list=[]
    return outresult,fc1_list,freq_copy,spec_copy

def fit_spectrum_model(amps, freqs, station, fc_min, fc_max, ttravel, review_mode, model_type, q_factor, fit_method="L-BFGS-B"):
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
    # cov = [None]
    
    # Set model exponents
    if type(model_type)==str:
        if model_type.upper() == 'B':
            n = [2, 1, 2, 1]  # n1,n2,n3,n4
        elif model_type.upper() == 'FB':
            n = [2, 2, 2, 2]
        elif model_type.upper() == 'SM':
            n = [1.5, 1, 3, 2]
        else:
            raise ValueError("Model must be 'B', 'FB', or 'SM'")
    elif type(model_type)==list:
        n = [model_type[0], model_type[1], model_type[0], model_type[1]]  

    # Set review mode
    review_mode = 'AUTO' if review_mode.lower() == 'yes' else 'MANUAL'

    # Set Q bounds
    q_min, q_max = 1, 1000000.
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
    try:
        bounds=[(fc_min,fc_max), (amp_low,amp_high), (q_min,q_max)]
        result = minimize(
            objective_function,
            [0.1*freqs[len(data)-1], max(data), q_min],
            args=(freqs[:len(data)], data, n[0], n[1], ttravel),
            method=fit_method,
            bounds=bounds,
            options=dict(maxiter=10000))
        fc, omega_0, Q = result.x
        params = [omega_0, fc, n[0], n[1], Q, ttravel]
        resid = result.fun
        
    except:
        traceback.print_exc()
        params, resid = None, None

                    
    return params, resid

def parallel_spectrum_fit(spectrum, freqs, station, fc_low, fc_high, travel_time, 
                         model_type, Q_value, fit_method):
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
    best_params = None #; best_cov = None
    residuals_list, fc_list = [], []

    if model_type.lower() == 'sm':
        # Set Q bounds
        Q_min, Q_max = 1, 1000000.
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
        params_list, resid_list = [], []
        for i in range(len(n1_range)):
            for j in range(len(n2_range)):
                result = parallel_fit_helper(
                        freqs, spectrum, amp_low, amp_high, 
                        fc_low, fc_high, travel_time,
                        Q_min, Q_max, n1_range[i], n2_range[j], fit_method)
                if result[0] is not None:
                    params_list.append(result[0])
                    resid_list.append(result[1])
            
        # Select best fit
        if params_list:
            for params in params_list:
                with np.errstate(divide='ignore', invalid='ignore'):
                    warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
                    resid = np.sum((np.log10(spectrum) - np.log10(compute_single_spectrum(freqs, *params)))**2)
                norm_resid = resid / (len(spectrum)*spectrum[0])
                residuals_list.append(norm_resid)
                fc_list.append(params[1])
            
            best_idx = np.argmin(np.array(residuals_list))
            best_params = np.asarray(params_list)[best_idx]
            residuals = resid_list[best_idx]
    else:
        raise ValueError("Parallel fitting only supported for 'sm' model type")

    return best_params, fc_list, residuals

def parallel_fit_helper(freqs, spectrum, amp_min, amp_max, fc_min, fc_max, 
                      travel_time, Q_min, Q_max, n1, n2, fit_method='L-BFGS-B'):
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
        bounds=[(fc_min,fc_max), (amp_min,amp_max), (Q_min,Q_max)]
        result = minimize(
            objective_function,
            [0.1*freqs[len(spectrum)-1], max(spectrum), Q_min],
            args=(freqs[:len(spectrum)], spectrum, n1, n2, travel_time),
            method=fit_method,
            bounds=bounds,
            options=dict(maxiter=10000))
        fc, omega_0, Q = result.x
        params = [omega_0, fc, n1, n2, Q, travel_time]
        norm_residual = result.fun
        
    except:
        params, norm_residual = None, None
        
    return params, norm_residual

def find_optimal_fit(model_type, num_workers, fc_min, fc_max, **kwargs):
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
    # Initialize result containers
    fc1_dict = {}
    fc2_dict = {}
    var_dict = {}
    uncertainty_dict = {}
    popts_dict = {}
    sumtype = kwargs.get('sumtype', '').lower()
    fig_path = kwargs.get('fig_path', '')
    fit_method = kwargs.get('fit_method','')
    freq_data = kwargs['freqmains']
    spec_data = kwargs['specmains']
    fit_range = kwargs['fit_range']
    fit_amp_ratio = kwargs['fit_amp_ratio']
    
    station_keys = list(spec_data.keys())
    aligned_specs, aligned_freqs, freq_bin, spec_ratio, stations = spec_data.copy(), freq_data.copy(), [], [], station_keys.copy()
    # aligned_specs, aligned_freqs, freq_bin, spec_ratio, stations = [], [], [], [], []

    for k in station_keys:
        if len(spec_data[k]) == 0:
            station_keys.remove(k)
            del spec_data[k]
            del freq_data[k]
    station_keys = list(spec_data.keys())
    
    fit_results =Parallel(n_jobs=num_workers)(
                delayed(optimize_corner_freq)(
                freq_data[station],
                spec_data[station],
                fc_max,
                station,
                model_type,
                fig_path,
                fit_method=fit_method,
                fit_amp_ratio=fit_amp_ratio)
            for station in station_keys)
    
    # Weighted fitting mode
    # if sumtype == 'weighted':
    for result, fc1_scan, freq_copy, _ in fit_results:
        if len(fc1_scan)!=0:
            station = list(result.keys())[0]
            data = result[station]
            fc1_popt = data['popt']
            fc1 = fc1_popt[0]; fc2 = fc1_popt[1]
            fc1_var = min(data['var'])
            fc1_uncertainty = data["fc1_uncertainty"]
            try:
                a=np.where(freq_data[station]<=0.5*fc1_popt[0])[0][-1]
            except:
                a=np.where(freq_data[station]>=0.5*fc1_popt[0])[0][0]
            try:
                b=np.where(freq_data[station]<=2*fc1_popt[0])[0][-1]
            except:
                b=np.where(freq_data[station]>=2*fc1_popt[0])[0][0]
            model = compute_spec_ratio(freq_data[station],*fc1_popt)
            low_freq_amp = np.mean(model[:a]); high_freq_amp = np.mean(model[b:])
            
            if low_freq_amp>=fit_amp_ratio*high_freq_amp and fc1<=0.8*fc_max:
                fc1_dict[station] = fc1
                fc2_dict[station] = fc2 
                var_dict[station] = fc1_var
                popts_dict[station] = fc1_popt
                uncertainty_dict[station] = fc1_uncertainty
            if low_freq_amp<fit_amp_ratio*high_freq_amp:
                del aligned_specs[station]
                del aligned_freqs[station]
                stations.remove(station)
                print(f'Remove {station} because low frequency amp_ratio is less than {fit_amp_ratio}x high frequency amp_ratio')
    # Standard fitting mode
    if sumtype != 'weighted':
        try:
            if len(stations)!=0:
                all_f_min = [aligned_freqs[i][0] for i in stations if len(aligned_freqs[i])!=0]; all_f_max = [aligned_freqs[i][-1] for i in stations if len(aligned_freqs[i])!=0]
                all_length = [len(aligned_freqs[i]) for i in stations]
                max_length = max(all_length)
                f_min = min(all_f_min); f_max = max(all_f_max)
                f_range = f_max - f_min
                f_thr_l = f_min + 0.3 * f_range#; f_thr_r = f_max - 0.3 * f_range
                
                for i in stations:
                    if len(aligned_freqs[i])!=0:
                        if (aligned_freqs[i][-1]-aligned_freqs[i][0])<0.5*f_range or aligned_freqs[i][0] > f_thr_l:#not (aligned_freqs[i][0] <= f_thr_l and aligned_freqs[i][-1] >= f_thr_r):
                            del aligned_freqs[i]
                            del aligned_specs[i]
                    else:
                        del aligned_freqs[i]
                        del aligned_specs[i]
                
                stations = list(aligned_freqs.keys())
                if len(stations)!=0:
                    all_f_min = [aligned_freqs[i][0] for i in stations ]; all_f_msax = [aligned_freqs[i][-1] for i in stations]   
                    freq_l = max(all_f_min); freq_r = min(all_f_max)
                    for i,sta in enumerate(stations):
                        n_l = np.where(aligned_freqs[sta]>=freq_l)[0][0]
                        if i==0:
                            n_r = np.where(aligned_freqs[sta]<=freq_r)[0][-1]
                            aligned_freqs[sta] = aligned_freqs[sta][n_l:n_r+1]
                            aligned_specs[sta] = aligned_specs[sta][n_l:n_r+1]
                            max_length = len(aligned_freqs[sta])
                        else:
                            aligned_freqs[sta] = aligned_freqs[sta][n_l:n_l+max_length]
                            aligned_specs[sta] = aligned_specs[sta][n_l:n_l+max_length]
                    aligned_freqs = list(aligned_freqs.values())
                    aligned_specs = list(aligned_specs.values())
                    if sumtype == 'average':
                        spec_ratio = np.average(aligned_specs, axis=0)
                    elif sumtype == 'median':
                        spec_ratio = np.median(aligned_specs, axis=0)
                    freq_bin = aligned_freqs[0]
                    
                    if isinstance(fit_range, list) and len(fit_range) > 0:
                        l = np.where(freq_bin >= fit_range[0])[0][0] if freq_bin[-1]>fit_range[0] else 0
                        u = np.where(freq_bin <= fit_range[-1])[0][-1] if freq_bin[0]<fit_range[-1] else len(freq_bin)
                        freq_bin = freq_bin[l:u]
                        spec_ratio = spec_ratio[l:u]
                
            fit_results = optimize_corner_freq(
                    freq_bin,
                    spec_ratio,
                    fc_max,
                    sumtype,
                    model_type,
                    fig_path,
                    fit_method=fit_method,
                    fit_amp_ratio=fit_amp_ratio)
            result, fc1_scan, _, _ = fit_results
            popt = result[sumtype]['popt']
            var = result[sumtype]['var']
            uncertainty = result[sumtype]["fc1_uncertainty"]
            
        except:
            if len(aligned_freqs)==0:
                print('No amplitude spectrum ratio curve satisfies the requirement that the ratio\nof low-frequency amplitude spectrum to high-frequency amplitude spectrum is greater than 2.\n','#'*72,sep='')
            else:
                traceback.print_exc()
            popt,var,uncertainty,freq_bin,spec_ratio=[],[],[],[],[]# traceback.print_exc()
    
    return popt if 'popt' in locals() else popts_dict,fc1_scan,fc1_dict,fc2_dict,\
        var if 'var' in locals() else var_dict,uncertainty if 'uncertainty' in locals() else uncertainty_dict,freq_bin,spec_ratio,stations
        
