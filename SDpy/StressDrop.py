#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:09:26 2025

@author: Zhang et al.
"""
from obspy import read
import os
import sys
import pandas as pd
import numpy as np
import ast
import json
import traceback
from .Preprocess import compute_spectra,cal_mag_diff,effective_sta_EGF,Read_metadata
from .Fit import fit_spectrum_model,find_optimal_fit,compute_spec_ratio,determine_asymptote_bounds,D_squared
from .Plot import create_spectral_ratio_plots,plot_spectral_ratio_fit,export_figure

def fit_and_plot(output_path, sta_start_times, sta_end_times, origin_time, magnitude, showfc2, min_magnitude_gap, combine_type,
                         model_type, plot_stations, num_tapers, use_fixed_window, n_workers,
                         result_df, overlap_ratio, t_coda, egf_flag, remove_instrument_resp,
                         free_surface_corr, vp, vs, late_picks, lon, dep, network, time_padding,
                         fit_range, mode, wavelen, quality_factor, channel,
                         main_event, egf_event, spec_ratios, freq_bins, egf_file, main_file,
                         wm_spec, wm_freq, wm_noise, main_tt, we_spec, we_freq, we_noise,
                         egf_tt, time_window, wave_type_main, wave_type_egf,
                         stf_amp, stf_time, mag_diff):

    if spec_ratios and freq_bins:
        station_keys = list(spec_ratios.keys())
        fc_dict = {}
        freq_keys = list(freq_bins.keys())
        max_len_key = max(freq_keys, key=lambda k: len(freq_bins[k]))
        stats = []

        for k in station_keys:
            sta = k.split('(')[0]
            if k not in freq_bins:
                freq_bins[k] = freq_bins[max_len_key][:len(spec_ratios[k])]
            else:
                diff = len(spec_ratios[k]) - len(freq_bins[k])
                if diff > 0:
                    freq_bins[k] = freq_bins[max_len_key][:len(spec_ratios[k])]
                elif diff < 0:
                    spec_ratios[k] = spec_ratios[k][:len(freq_bins[k])]

            try:
                if sta not in stats:
                    popt, pcov, _ = fit_spectrum_model(wm_spec[sta], wm_freq[sta], k,
                                                       min(wm_freq[sta]), max(wm_freq[sta]) * 0.5,
                                                       main_tt[sta], 'yes', model_type, quality_factor)
                    if popt is not None:
                        stats.append(sta)
                        fc = round(popt[1], 1)
                        fc_dict[k] = fc
                    else:
                        station_keys.remove(k)
                        print(f'Misfit too big at station {k}, removed.')
            except:
                traceback.print_exc()

        aligned_specs, aligned_freqs, spec_ratio= [], [], []
        valid_keys = station_keys.copy()

        for k in valid_keys:
            if egf_event and len(spec_ratios[k]) != 0:
                aligned_specs.append(spec_ratios[k])
                aligned_freqs.append(freq_bins[k])
            elif egf_event and len(spec_ratios[k]) == 0:
                station_keys.remove(k)
                del spec_ratios[k]
                del freq_bins[k]

        try:
            max_len = max(map(len, aligned_freqs))
            max_spec_len = max(map(len, aligned_specs))
            ref_idx = [i for i, f in enumerate(aligned_freqs) if len(f) == max_len][0]
            for i in range(len(aligned_freqs)):
                f_diff = max_len - len(aligned_freqs[i])
                s_diff = max_spec_len - len(aligned_specs[i])
                if s_diff > 0:
                    aligned_specs[i] = np.pad(aligned_specs[i], (0, s_diff), 'edge')
                    aligned_freqs[i] = np.pad(aligned_freqs[i], (0, f_diff), 'edge')

            if combine_type.lower() == 'average' and egf_event:
                spec_ratio = np.average(aligned_specs, axis=0)
            elif combine_type.lower() == 'median' and egf_event:
                spec_ratio = np.median(aligned_specs, axis=0)
            freq_bin = aligned_freqs[ref_idx]
            
            fc_max = max(fc_dict.values()) + 50
            fc_min = np.median(list(fc_dict.values())) * 0.5

        except:
            # traceback.print_exc()
            pass

        if plot_stations:
            indexx = list(set(stats) & set(plot_stations))
        else:
            indexx = station_keys[0]

        if egf_event:
            print('#' * 72, f"\nAmplitude Spectral Ratio fitted by {wave_type_main} wave\n", '#' * 72, sep='')
            result_df['wavetype'] = [wave_type_main]

            if isinstance(fit_range, list) and len(fit_range) > 0:
                l = np.where(freq_bin >= fit_range[0])[0][0];u = np.where(freq_bin <= fit_range[-1])[0][-1]
                freq_bin = freq_bin[l:u]
                spec_ratio = spec_ratio[l:u]

            frqprtub, freq_final, tplier, residuals, popt, pcov, fc1, fc2, var = find_optimal_fit(
                model_type, n_workers, result_df, freq_bin, spec_ratio, fc_min, fc_max,
                path=output_path, mode=mode, sumtype=combine_type, freqmains=freq_bins, specmains=spec_ratios)

            if combine_type.lower() != 'weighted':
                fc_main = float(round(popt[0], 2))
                gamma_a = fc_main
                all_fc, all_var = fc1, var
                print('#' * 72, f"\nfc(main) = {fc_main}\nfc(egf) = {round(popt[1], 2)}\n", '#' * 72, sep='')
            else:
                gamma_a, D_s, all_fc, all_var = D_squared(fc1, var, freq_bins, spec_ratios)
                gamma_a2, _, _, _ = D_squared(fc2, var, freq_bins, spec_ratios)
                fc_main = round(gamma_a, 2)
                print('#' * 72, f"\nfc(main) = {fc_main}\n", '#' * 72, sep='')

            fig, ax, _ = create_spectral_ratio_plots(output_path, stats, sta_start_times, sta_end_times, origin_time, min_magnitude_gap,
                                                      remove_instrument_resp, free_surface_corr, vp, vs, late_picks, lon, dep,
                                                      network, time_padding, wavelen, t_coda, num_tapers, overlap_ratio, n_workers,
                                                      model_type, egf_flag, mode, combine_type, residuals, var, quality_factor,
                                                      channel, main_event, egf_event, spec_ratios, freq_bins, wm_freq, wm_spec,
                                                      wm_noise, we_freq, we_spec, we_noise, indexx, time_window, main_file,
                                                      egf_file, wave_type_main, mag_diff, main_tt)

            result_df['fc(main)'] = [fc_main]
            if (len(egf_flag) == 1 and egf_flag[0]) or (len(egf_flag) > 1 and mode == 0):
                result_df['fc(egf)'] = [gamma_a2 if fc2 else round(popt[1], 2)]

            if combine_type.lower()!='weighted':
                ax.loglog(freq_bin, spec_ratio, 'orangered', label='data', linewidth=5)
                ax.loglog(freq_bin, np.divide(compute_spec_ratio(freq_bin, *popt), tplier), 'g--', linewidth=2)

            if residuals:
                ubm, lbm, ube, lbe = determine_asymptote_bounds(freq_bin,spec_ratio)
                if ube > lbe and min(residuals) <= 10.6:
                    D_s = min(residuals)
                    maxy = max(max(spec_ratios.values(), key=len))
                    if (mode == 1 and egf_event == egf_flag[-1]) or mode == 0:
                        rms = plot_spectral_ratio_fit(showfc2, magnitude, main_event, egf_event, freq_bin,
                                                      spec_ratio, tplier, frqprtub, residuals, ax, popt,
                                                      maxy, mag_diff, combine_type, gamma_a, D_s, all_fc, all_var)
                        result_df['RMS'] = [rms]
                        export_figure(output_path, wavelen, t_coda, magnitude, [egf_event], model_type,
                                      fig, 'spec', indexx, wave_type_main, main_event, egf_event, mode)
                else:
                    print('The smallest RMS value exceeds 0.6!')
            else:
                if combine_type.lower() == 'weighted' or mode == 1:
                    freq_bin, spec_ratio = freq_bins, spec_ratios
                    maxy = max(max(spec_ratios.values(), key=len))
                    D_s = plot_spectral_ratio_fit(showfc2, magnitude, main_event, egf_event, freq_bin,
                                                  spec_ratio, tplier, frqprtub, residuals, ax, popt,
                                                  maxy, mag_diff, combine_type, gamma_a, D_s, all_fc, all_var)
                    result_df['D_s'] = [D_s]
                    export_figure(output_path, wavelen, t_coda, magnitude, egf_flag, model_type,
                                  fig, 'spec', indexx, wave_type_main, main_event, egf_event, mode)
        else:
            print('#' * 72, f"\nIndividual Amplitude Spectra fitted using {wave_type_egf} wave\n", '#' * 72, sep='')
            fig, ax, popt = create_spectral_ratio_plots(output_path, station_keys, sta_start_times, sta_end_times, origin_time, min_magnitude_gap,
                                                        remove_instrument_resp, free_surface_corr, vp, vs, late_picks, lon, dep,
                                                        network, time_padding, wavelen, t_coda, num_tapers, overlap_ratio, n_workers,
                                                        model_type, egf_flag, mode, combine_type, [], [], quality_factor, channel,
                                                        main_event, egf_event, spec_ratios, freq_bins, wm_freq, wm_spec, wm_noise,
                                                        we_freq, we_spec, we_noise, indexx, time_window, main_file, egf_file,
                                                        wave_type_egf, mag_diff, main_tt)
            fc_main = popt
            if combine_type == 'weighted':
                gamma_a, D_s, _, _ = D_squared(popt['fc_list'], popt['rms_list'])
                fc_main = round(gamma_a, 2)
                result_df['VAR'] = [D_s]
            result_df['fc'] = fc_main
            print('#' * 72, f"\nfc = {fc_main}\n", '#' * 72, sep='')
            export_figure(output_path, wavelen, t_coda, magnitude, egf_flag, model_type, fig,
                          'ind', indexx, wave_type_egf, main_event, egf_event, mode)
    return fc_main

def sd_esti(paths,ssta,EGF_event,min_mag_diff,fixed_window,snrthres,num_tapers,assume_drop,sumtype,source_model,overlap,plot_station,Parri,\
              Sarri,Orig,showfc2,mag,numworkers,T_coda,wave_len,time_add,remove_resp,Late,Lone,Dep,freesurface_cor,freesurface_vp,freesurface_vs,K,\
              Beta,fit_freq_range,sourcepara_df,Q,all_sta_spec,fname,wv,mode,freqmains,specmains,stfys,all_stations,*paras):
    
    data_path=paths['data_path']
    # out_path=paths['out_path']
    j=fname[0];k=None
    wv1=wv;wv2=wv
    specmain, stfy, freqmain, wefc, we, wen, trte, egffile, wmfc, wm, wmn, mainfile, trtm, stfx, chan = paras
    mag_diffs=min_mag_diff
    mainev = j;egfev = None;fc1main = None
        
    if chan:
        evfold1 = os.path.join(data_path, ssta, f'{j}_*{chan}*.SAC')   
    else:
        evfold1 = os.path.join(data_path, ssta, f'{j}_*.SAC')

    if fname[1]:
        k=fname[1]
        egfev = k
        if chan:
            evfold2 = os.path.join(data_path, ssta, f'*{k}_*{chan}*.SAC')
        else:
            evfold2 = os.path.join(data_path, ssta, f'*{k}_*.SAC')
        
        print('%s with %s' % (j,k))
    else:
        print('*******Using direct amplitude spectrum fitting method*******')

    
    mt=read(evfold1)
    net=mt[0].stats.network.strip()
    station = mt[0].stats.station.strip()
   
    file1 = evfold1;file2 = None
    mag_diff=1000
    if fname[1]:
        file2 = evfold2
        st1_calmag=read(evfold1)
        st2_calmag = read(evfold2)
        mag_diff=cal_mag_diff( st1_calmag, st2_calmag )
    
        print('==========================MAGTITUDE DIFFERENCE==========================\n','\t'*4,round(mag_diff,2),\
              '\n========================================================================')
  
    if mag_diff>=min_mag_diff:
        mag_diffs=mag_diff
        
    specratio,freqbin,rawefc,rawe,rawen,rawmfc,rawm,rawmn,trav_time_main,_,sta_freq,sta_spec,R,\
        = compute_spectra(paths,T_coda,time_add,wave_len,num_tapers,fixed_window,remove_resp,snrthres,Sarri,Parri,Orig,\
          Late,Lone,Dep,freesurface_cor,freesurface_vp,freesurface_vs,file1,file2,mainev,egfev,wv1,overlap,chan)
    # save_out(out_path,freqbin, specratio, rawm, rawe, wv1,station,mainev )
    if fit_freq_range:
        if fname[1]:
            a=(np.array(np.where(freqbin>=fit_freq_range[0]))).min();b=(np.array(np.where(freqbin<=fit_freq_range[1]))).max()+1
            freqbin=freqbin[a:b];specratio=specratio[a:b]
        else:
            a=(np.array(np.where(rawmfc>=fit_freq_range[0]))).min();b=(np.array(np.where(rawmfc<=fit_freq_range[1]))).max()+1
            rawmfc=rawmfc[a:b];rawm=rawm[a:b];rawmn=rawmn[a:b]
            freqbin=freqbin[(np.array(np.where(freqbin>=fit_freq_range[0]))).min():(np.array(np.where(freqbin<=fit_freq_range[1]))).max()+1]
    else:
        fit_freq_range=[0,int(1/mt[0].stats.delta)]
    if (np.isnan(specratio)).any() != True:
        specmain[station] = specratio; freqmain[station] = freqbin
        mainfile[station] = file1; trtm[station] = trav_time_main
        wmfc[station] = rawmfc; wm[station] = rawm; wmn[station] = rawmn
        all_sta_spec[station]=[sta_freq, sta_spec, R]
        egffile[station] = file2; trtm[station] = trav_time_main;
        wefc[station] = rawefc; we[station] = rawe; wen[station] = rawen;
    
    if ssta==all_stations[-1]:
        if specmain and k==EGF_event[-1] and mode==1:
            freqmains[k]=freqmain
            specmains[k]=specmain
            freqmain={};specmain={}
            for item in list(freqmains.keys()):
                for sta in list(freqmains[item].keys()):
                    sta_n=sta+'('+item+')'
                    freqmain[sta_n]=freqmains[item][sta]
            for item in list(specmains.keys()):
                for sta in list(specmains[item].keys()):
                    sta_n=sta+'('+item+')'
                    specmain[sta_n]=specmains[item][sta]     
        
        if k==EGF_event[-1] or k==None:
            
            fc1main=fit_and_plot(paths,Sarri,Parri,Orig,mag,showfc2,min_mag_diff,sumtype,source_model,plot_station,num_tapers,fixed_window,numworkers,\
                                sourcepara_df,overlap,T_coda,EGF_event,remove_resp,freesurface_cor,freesurface_vp,freesurface_vs,\
                                Late,Lone,Dep,net,time_add,fit_freq_range,mode,wave_len,Q,chan,\
                                j,k,specmain,freqmain,egffile,mainfile,wm,wmfc,wmn,trtm,we,wefc,wen,trte,fixed_window,wv1,wv2,stfy,stfx,mag_diffs)
    
                    
    return specmain,stfy,freqmain,wefc,we,wen,trte,egffile,wmfc,wm,wmn,mainfile,trtm,stfx,all_sta_spec,fc1main

def calc_M0(fc,all_sta_spec,main_tt,wave,model,Q,fsf,rho=2700,c=None,U=None):
    """
    The seismic moment of the target earthquake is estimated 
    based on the calculation formula in Brune (1970).

    Parameters
    ----------
    fc : float
        conner freqenc.
    freqbin : array
        The frequency range of the spectrum.
    all_sta_spec : dict
        The three-component frequencies and spectra of each station and the distance between the station and the seismic source.
    wave : str
        Wave type used for calculate spectrum.
    c : float
        The propagation velocity of seismic waves (Unit: m/s).
    wave : str
        The wave type used for estimating stress drop.
    model : str
        The source model used for fitting the spectrum.
    Q : float
        The quality factor.
    fsf : float
        The free surface factor (Without free surface correction).
    rho : float, optional
        Source region rock density. The default is 2600 kg/m^3.
    U : float, optional
        The average value of the radiation pattern term. The default is None.

    Returns
    -------
    M0 : float
        The moment of target event (Unit: Nm).

    """
    all_sta_M0=[]
    
    if wave.upper()=='P':
        U=U if U else 0.52
    elif wave.upper()!='P':
        U=U if U else 0.63
    for sta in list(all_sta_spec.keys()):
        try:
            omega=[]
            freqbin=all_sta_spec[sta][0]
            sta_spec=all_sta_spec[sta][1]
            for chan in list(sta_spec.keys()):
                popt, _, _ = fit_spectrum_model(sta_spec[chan], freqbin, sta, 
                            min(freqbin), max(freqbin), main_tt[sta], 'yes', model, Q)
                omega_c=popt[0]
                # index=np.where(freqbin>=fc)[0][0]
                # omega_c=np.mean(sta_spec[chan][:index])
                omega.append(omega_c**2)
            R=all_sta_spec[sta][2] #The distance between epicenter and station (Unit: m).
            omega0=np.sqrt(sum(omega))
            M0=4*np.pi*rho*(c**3)*R*omega0*(10**-6)/(U*fsf)
            all_sta_M0.append(M0)
        except:
            traceback.print_exc()
            pass        
    M0=np.mean(all_sta_M0)
    M0 = float("%.2e" % M0)
    return M0

def get_stressdrop(mainev,sourcepara_df,fc,all_sta_spec,main_tt,wave,model,Q,fsf,rho,c,U,moment,k,b,dep,assume_drop):
    # M0 in Nm;b in m/s;stress_drop in pa
    if isinstance(b, (int, float)):
        b=b
    if os.path.exists(b):
        v_m=pd.read_csv(b)
        Dep=list(v_m['Dep(km)'].values)
        V_s=list(v_m['V_s(m/s)'].values)
        for i in range(len(Dep)):
            if i==0:
                if dep<Dep[i]:
                    b=V_s[i]
            else:
                if Dep[i-1]<=dep<Dep[i]:
                    b=V_s[i]
    rad = (k * b) / fc
    if not pd.isna(moment):
        M0=moment
    else:
        if not c and wave.lower()=='P':
            c=1.73*b
        if not c and wave.lower()!='P':
            c=b
        M0=calc_M0(fc,all_sta_spec,main_tt,wave,model,Q,fsf,rho=rho,c=c,U=U)

    stress_drop=(7*M0)/(16*rad**3)
    drop_Mpa=stress_drop*(1E-6)
    
    
    sourcepara_df['Moment(Nm)'] = [M0]
    print('Seismic moment of {0} is {1} Nm'.format(mainev, M0))
    sourcepara_df['Stressdrop(Mpa)'] = [round(drop_Mpa,2)]
    sourcepara_df['Radius(m)'] = [round(rad,2)]
    print('Stress drop of {0} is {1} Mpa'.format(mainev,round(drop_Mpa,2)))
    print('Radius of {0} is {1} m'.format(mainev,round(rad,2)))
    
    try:
        assu_r=assume_strdrop(assume_drop,M0)
    except Exception as e:
        print(e)
    print('Assume the stress drop is {} Mpa,the radius of main event is {} m'.format(assume_drop,round(assu_r,2)),'\n','#'*72,sep='')
    return sourcepara_df

def assume_strdrop(str_drop,M0):

    a=7*M0/(16*str_drop*1e6 )
    r=np.power(a, 1/3)
    return r

def save_out(out_path,freqbin,specratio,wm,we,wv,station,evid):
    
    if not os.path.exists(os.path.join(out_path, station)):
        os.makedirs(os.path.join(out_path, station))
    spec_data= pd.DataFrame()
    spec_data['freq'] = freqbin
    if  len(specratio)!=0:
        spec_data['specratio'] = specratio
    else:
        pass
    spec_data['mainspec'] = wm[0:len(freqbin)]
    try:
        if len(we)!=0:
            spec_data['egfspec'] = we[0:len(specratio)]
            spec_data.to_csv(os.path.join(out_path, station, f"{wv}_{evid}specratio.csv"),index=False)
    except:
        spec_data.to_csv(os.path.join(out_path, station, f"{wv}_{evid}spectra.csv"),index=False)
    
    return None

def read_control_file(control_filepath,p_dict):
    '''
    Reads the user's control file.
    If the control file does not contain a variable, the default value in p_dict is used.
    If the control file contains an unexpected variable, it is ignored after printing a warning message.
    This file should follow the format:
        $var_name
        var_value
        ...
    Where "$var_name" is the name of the variable with a "$" as the first non-whitespace character,
    and "var_value" is the value of the variable on the following line.
    Velocity models ($vmodel_paths) can be specified on multiple lines if desired, e.g.:
        $vmodel_paths
        path_to/vmodel1.txt
        path_to/vmodel2.txt

    User text on all other lines will be ignored, provided the first non-whitespace character on the line is not a "$".
    User text will also be ignored following a space (" ") after "$var_name"
    '''
    with open(control_filepath) as f:
        lines=f.read().splitlines()
        lines=[line.strip() for line in lines]

    var_name=[]
    var_line=[]
    value_line=[]
    for line_x,line in enumerate(lines):
            if line[0]=='$':
                tmp_var_name=line.split()[0][1:]
                tmp_var_name=tmp_var_name.split('#')[0] 
                if tmp_var_name in p_dict.keys():
                    var_name.append(tmp_var_name)
                    var_line.append(line_x)
                else:
                    raise ValueError('Unknown variable name ({}) in the control file on line {}:\n\t{}'.format(tmp_var_name,line_x,line))
            elif line[0]!='#':
                value_line.append(line_x)

    var_set=set()
    duplicate_var=[x for x in var_name if x in var_set or var_set.add(x)]
    if duplicate_var:
        raise ValueError('Duplicate variable name in control file: \"{}\"'.format(duplicate_var[0]))


    corr_bool=np.isin( (np.asarray(value_line)-1),var_line)
    if not(np.all(corr_bool)):
        prob_ind=np.where(corr_bool==False)[0][0]
        raise ValueError('Unable to associate a value (\"{}\") on line {} with a variable in the control file. If this is a variable, start the line with a \'$\'. If this is intended to be ignored, start the line with a \'#\'.'.format(lines[value_line[prob_ind]],value_line[prob_ind]))

    var_value=[]
    for line_x in var_line:
        tmp_var_value=lines[line_x+1].strip()

        if tmp_var_value[0]=='#':
            raise ValueError('Value (\"{}\") on line {} in the control file starts with a \'#\'.'.format(tmp_var_value,line_x+1) )
        if tmp_var_value.isdigit(): 
            tmp_var_value=int(tmp_var_value)
        elif tmp_var_value.replace('.','',1).isdigit(): 
            tmp_var_value=float(tmp_var_value)
        elif (tmp_var_value=='True'):
            tmp_var_value=True
        elif (tmp_var_value=='False'): 
            tmp_var_value=False
        elif ('[' in tmp_var_value):
            tmp_var_value=ast.literal_eval(tmp_var_value)
        elif (' ' in tmp_var_value): 
            tmp_var_value=tmp_var_value.split()
        var_value.append(tmp_var_value)

    for name,value in zip(var_name,var_value):
        p_dict[name]=value

    return p_dict

def Para_init():
    p_dict={'controlfile':'',
    'method':1,
    'wv':'p',
    'Main_events':'',
    'EGF_events':'',
    'All_stations':'',
    'wave_align':'cc',
    'fixed_window':1,
    'num_windows':1,
    'overlap':0.0,
    'remove_resp':'no',
    'snrthres':3,
    'Q':None,
    'T_coda':3,
    'fs_cor':'no',
    'fs_vs':3500,
    'fs_vp':6000,
    'fs_factor':2,
    'num_workers':6,
    'num_tapers':5,
    'source_model':'sm' ,  
    'sumtype' : 'weighted', 
    'showfc2':'yes',
    'pre_filt' :'' ,
    'time_add':1,
    'min_mag_diff':0.3,
    'assume_drop':3,
    'k':None,
    'beta':3000,
    'fit_freq_range':None,
    'data_center':'IRIS',
    'data_path':'',
    'resp_path':'',
    'out_path':'',
    'rho':2700,
    'c':None,
    'U':None,
    'chan':'*',
    'mode':1}
    return p_dict

def Read_para(controlfile):
    p_dict_init=Para_init()
    p_dict_init['controlfile']=controlfile
    if p_dict_init['controlfile']:
        file_type=os.path.splitext(p_dict_init['controlfile'])[1]
        if 'json' in file_type.lower():
            with open(p_dict_init['controlfile'], "r", encoding="utf-8") as file:
                p_dict = json.load(file)
                p_dict_update = p_dict_init.copy()
                for p_dict_var in p_dict.keys():
                    if p_dict_var in list(p_dict_init.keys()) and p_dict[p_dict_var]:
                        p_dict_update[p_dict_var]=p_dict[p_dict_var]
                print('para_dict:\n',p_dict_update)
        elif 'txt' in file_type.lower():
            p_dict=read_control_file(p_dict_init['controlfile'],p_dict_init)
            p_dict_update = p_dict_init.copy()
            for p_dict_var in p_dict.keys():
                if p_dict_var in list(p_dict_init.keys()) and p_dict[p_dict_var]:
                    p_dict_update[p_dict_var]=p_dict[p_dict_var]
            print('para_dict:\n',p_dict_update)
    return p_dict_update

def stressdrop(controlfile): 
    p_dict=Read_para(controlfile)
    for key, value in p_dict.items():
        exec(f"{key} = value")
    method=p_dict['method']
    wv=p_dict['wv']
    d1=pd.read_csv(p_dict['Main_events'])
    Main_events=list(d1['Event ID'].values)
    for i in range(len(Main_events)):
        Main_events[i]=str(Main_events[i])
    if method==2:
        EGF_info_path=p_dict['EGF_events']
        if EGF_info_path:
            d2=pd.read_csv(EGF_info_path)
            EGF_events=list(d2['Event ID'].values)
            EGF_events=[x for x in EGF_events if not pd.isna(x)]
            all_finals=[]
            for i in range(len(EGF_events)):
                all_finals.append({})
                if 'EGF' not in str(EGF_events[i]).upper():
                    EGF_events[i]='EGF'+str(EGF_events[i])
                # if str(EGF_events[i]).isdigit():
                #     EGF_events[i]='EGF'+str(EGF_events[i])
                # else:
                #     EGF_events[i]=str(EGF_events[i])
            d2['Event ID']=EGF_events
        else:
            EGF_events=['']
        if not EGF_events:
            print('EGF_events_infomation.csv is empty!!!')
        d3=pd.concat([d1,d2], ignore_index=True)
    elif method==1:
        EGF_events=['']
        d3=d1
        
    all_stations_path=p_dict['All_stations']
    d=pd.read_csv(all_stations_path)
    all_stas=[x for x in list(d['Stations'].values) if not pd.isna(x)]
    all_stas=sorted(all_stas)
    plot_station=list(d['Plot stations'].values)
    plot_station=[x for x in plot_station if not pd.isna(x)]
    sorted(plot_station)
    wave_align=p_dict['wave_align']
    fixed_window=p_dict['fixed_window']   
    num_window=p_dict['num_windows']
    overlap=p_dict['overlap']
    remove_resp=p_dict['remove_resp']
    snrthres=p_dict['snrthres']
    Q=p_dict['Q']
    T_coda=p_dict['T_coda']
    freesurface_cor=p_dict['fs_cor']
    freesurface_vs=p_dict['fs_vs']
    freesurface_vp=p_dict['fs_vp']
    fsf=p_dict['fs_factor'] 
    numworkers=p_dict['num_workers']
    num_tapers=p_dict['num_tapers']
    wave_len=(num_window+overlap-num_window*overlap)*fixed_window
    source_model=p_dict['source_model']  
    sumtype = p_dict['sumtype']
    showfc2=p_dict['showfc2']
    time_add=p_dict['time_add']
    min_mag_diff=p_dict['min_mag_diff']
    assume_drop=p_dict['assume_drop']
    k=p_dict['k']
    mode=p_dict['mode']
    if len(EGF_events)<=1 or method==1:
        mode=0
    if not k:
        if wv.upper()=='P':
            k=0.32
        elif wv.upper()=='S' or wv.upper()=='CODA':
            k=0.21
    beta=p_dict['beta']
    fit_freq_range=p_dict['fit_freq_range']
    data_path=p_dict['data_path']
    resp_path=p_dict['resp_path']
    out_path=p_dict['out_path']
    if out_path:
        out_path=out_path
    else:
        out_path=os.path.dirname(data_path)+'/Output/'
    dataframe_final ={}
    rho=p_dict['rho']
    c=p_dict['c']
    U=p_dict['U']
    chan=p_dict['chan']
    if len(EGF_events)>1 and mode==1:
        sumtype='weighted'
    paths={}
    paths['data_path']=data_path
    paths['resp_path']=resp_path
    paths['out_path']=out_path

    all_stations=all_stas.copy()
    for eid in range(len(Main_events)):
        freqmains,specmains,stfys={},{},{}
        
        j=0
        while j==0:
            wmfc = {};wm = {};wmn = {};trtm = {};mainfile = {};egffile = {};all_sta_spec={}
            for n,EGF_event in enumerate(EGF_events):
                try:
                    if method==2:
                        Method='EGFs'
                    if method==1:
                        Method='Single'
            
                    fname=[Main_events[eid],EGF_event]
                    all_stas=effective_sta_EGF(method,data_path,all_stations,fname)
                    all_sta=all_stas
                    if plot_station:
                        plot_station=plot_station
                    else:
                        plot_station=all_stations   
                    Orig,Sarri,Parri,Late,Lone,Dep,mag,sourcepara_df=Read_metadata(all_stas,d3,fname,data_path,wave_align=wave_align)
                    moment=(d3[d3['Event ID']==Main_events[eid]]['Moment']).values[0]
                    
                    if method==2:
                        sourcepara_df['EGF']=[fname[1]]
                        sourcepara_df['Origtime_egf'] = list(d3[d3['Event ID'] == fname[1]]['Origin time'])[0]
                    elif method==1:
                        pass 
                    s=all_sta.copy()
                    specmain,stfy={},{}
                    freqmain = {};wefc = {};we = {};wen = {};trte = {};stfx = {};
                    for station in s:
                        print('#'*72,'\n','\t'*3,f'Working on {station}\n','#'*72,sep='')
                        specmain,stfy,freqmain,wefc,we,wen,trte,egffile,wmfc,wm,wmn,mainfile,trtm,stfx,all_sta_spec,fc1main = sd_esti(paths,station,EGF_events,min_mag_diff,fixed_window,\
                            snrthres,num_tapers,assume_drop,sumtype,source_model,overlap,plot_station,Parri,Sarri,Orig,showfc2,mag,numworkers,T_coda,wave_len,\
                            time_add,remove_resp,Late,Lone,Dep,freesurface_cor,freesurface_vp,freesurface_vs,k,beta,fit_freq_range,sourcepara_df,Q,all_sta_spec,\
                            fname,wv,mode,freqmains,specmains,stfys,all_sta,specmain,stfy,freqmain,wefc,we,wen,trte,egffile,wmfc,wm,wmn,mainfile,trtm,stfx,chan)   
                        
                        j=1
                        
                    if mode==0 or (mode==1 and EGF_event==EGF_events[-1]):
                        get_stressdrop(Main_events[eid], sourcepara_df, fc1main, all_sta_spec, trtm, wv, source_model, Q,
                                       fsf, rho, c, U, moment, k, beta, Dep[station][Main_events[eid]], assume_drop)
                    
                    excel_path=os.path.join(out_path,Main_events[eid])
                    if not os.path.exists(excel_path):
                        os.makedirs(excel_path)
                    sourcepara_df.to_excel(os.path.join(excel_path,f'{wv}_{fixed_window}s_{Main_events[eid]}_{Method}_{source_model}.xlsx'))   
                    if EGF_event!=EGF_events[-1]:     
                        freqmains[EGF_event]=freqmain;specmains[EGF_event]=specmain;stfys[EGF_event]=stfy
                    dataframe_final[eid] =sourcepara_df
                    if method==2:
                        all_finals[n][eid]=sourcepara_df
                
                except:
                    traceback.print_exc() 
                    sys.exit()
            j=1
            try:
                final = pd.concat(list(dataframe_final.values()), ignore_index=True)
            except:
                final=pd.DataFrame(dataframe_final)
    # for i in np.arange(len(all_stations)):
    #     excel_path=os.path.join(out_path,all_stations[i])
    excel_path=out_path
    i=len(all_stations)-1
    if not os.path.exists(excel_path):
        os.makedirs(excel_path)
    # EGFs(num>3) or Spectral fiiting
    if (len(EGF_events)>3 and mode==1) or method==1:
        if i==len(all_stations)-1:
            print('Results of Target events:\n',final.iloc[:, [0] + list(range(-5, 0))])
        final.to_excel(os.path.join(excel_path,f'{wv}_{fixed_window}s_AllTargetEvents_{Method}_{source_model}.xlsx'))
    # EGFs(num<=3)
    if 1<len(EGF_events)<=3 and mode==1:
        if i==len(all_stations)-1:
            print('Results of Target events with All EGFs:\n',final.iloc[:, [0] + list(range(-5, 0))])
        final.to_excel (os.path.join(excel_path,f'{wv}_{fixed_window}s_AllTargetEvents_{EGF_events}_{source_model}.xlsx'))
    # Singe EGF
    if (len(EGF_events)>1 and mode==0) or (len(EGF_events)==1 and EGF_events[0]):
        for e,item in enumerate(all_finals):
            final = pd.concat(list(item.values()), ignore_index=True)
            if i==len(all_stations)-1:
                print(f'Results of Target events with {EGF_events[e]}:\n',final.iloc[:, [0] + list(range(-6, 0))])
            final.to_excel(os.path.join(excel_path,f'{wv}_{fixed_window}s_AllTargetEvents_{EGF_events[e]}_{source_model}.xlsx'))         
    print('#'*72,'\n','\t'*4,'Complete !\n','#'*72,sep='')  
    return final

