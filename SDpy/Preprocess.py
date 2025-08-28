  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 15:48:53 2024

@author: Zhang et al.
"""

import os
import glob
from obspy import read,UTCDateTime
import pandas as pd
import numpy as np
import multitaper
import traceback
from obspy.core import Stream
from obspy import read_inventory
from pyproj import Geod
from scipy.signal import hilbert
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import locations2degrees as loc2deg
import warnings

def Read_metadata(all_stas,ev_info,fname,data_path,wave_align='c'):
    
    ev_info['Event ID'] = ev_info['Event ID'].astype(str)
    Orig,Sarri,Parri,Late,Lone,Dep={},{},{},{},{},{}
    sourcepara_df = pd.DataFrame()
    ev_paths=[]
    # print(fname)
    for ev in fname:
        for root,_,_ in os.walk(data_path):
            ev_path=glob.glob(os.path.join(root, f'{str(ev)}_*'))
            ev_paths=set(ev_paths)|set(ev_path)
    for sta in all_stas:
        if ev_paths:
            main_path=[path for path in ev_paths if read(path)[0].stats.station==sta and os.path.basename(path).split('_')[0]==fname[0]]
            egf_path=[path for path in ev_paths if read(path)[0].stats.station==sta and os.path.basename(path).split('_')[0]==fname[1]]
            if len(main_path)==0 or len(egf_path)==0:
                try:
                    main_path=[path for path in ev_paths if read(path)[0].stats.station==sta and os.path.basename(path).rsplit('_',1)[0]==fname[0]]
                    egf_path=[path for path in ev_paths if read(path)[0].stats.station==sta and os.path.basename(path).rsplit('_',1)[0]==fname[1]]
                    if len(main_path)==0 or (fname[-1] and len(egf_path))==0:
                        print('Too much "_" in the "Event ID",please rename the "Event ID" without "_"!')
                except:
                    pass
            main_path.sort()
            egf_path.sort()
            st_main=read(main_path[0])
            channel=st_main[0].stats.channel
            Start = st_main[0].stats['starttime']
            starttime = UTCDateTime(Start)
            starttime=st_main[0].stats.starttime
            p_arrival=UTCDateTime(starttime+st_main[0].stats.sac.t1)
            s_arrival=UTCDateTime(starttime+st_main[0].stats.sac.t2)
            result = ev_info[ev_info['Event ID']==fname[0]]
            origin_time=UTCDateTime(result['Origin time'].values[0])
            MAG = result['Mag'].values[0]
            LAT = result['Lat'].values[0]
            LON = result['Lon'].values[0]
            DEP = result['Dep'].values[0]
            
            if egf_path:
                for e in egf_path:
                    st=read(e)
                    if st[0].stats.channel==channel:
                        st_egf=st
                    else:
                        try:
                            st_egf=read(egf_path[0])
                        except:
                            traceback.print_exc()
                Start1 = st_egf[0].stats['starttime']
                starttime1 = UTCDateTime(Start1)
                starttime1=st_egf[0].stats.starttime
                p_arrival1=UTCDateTime(starttime1+st_egf[0].stats.sac.t1)
                s_arrival1=UTCDateTime(starttime1+st_egf[0].stats.sac.t2)
                result1 = ev_info[ev_info['Event ID']==fname[1]]
                origin_time1=UTCDateTime(result1['Origin time'].values[0])
                MAG1 = result['Mag'].values[0]
                LAT1 = result['Lat'].values[0]
                LON1 = result['Lon'].values[0]
                DEP1 = result['Dep'].values[0]
                Orig[sta]={fname[0]:origin_time,fname[1]:origin_time1}
                if wave_align.lower()=='cc':  
                    template1=st_main.copy()
                    template2=st_main.copy()
                    template1.trim(starttime=p_arrival,endtime=min(p_arrival+30,st_main[0].stats.endtime-2))
                    template2.trim(starttime=s_arrival,endtime=min(s_arrival+30,st_main[0].stats.endtime-2))
                    p_arrival1,cor1,_=Corrss1(template1,st_egf)
                    s_arrival1,cor2,_=Corrss1(template2,st_egf)
                    
                Sarri[sta]={fname[0]:s_arrival,fname[1]:s_arrival1}
                Parri[sta]={fname[0]:p_arrival,fname[1]:p_arrival1}
                Late[sta]={fname[0]:LAT,fname[1]:LAT1}
                Lone[sta]={fname[0]:LON,fname[1]:LON1}
                Dep[sta]={fname[0]:DEP,fname[1]:DEP1}
                mag={fname[0]:MAG,fname[1]:MAG1}
            else:
                Orig[sta]={fname[0]:origin_time}
                Sarri[sta]={fname[0]:s_arrival}
                Parri[sta]={fname[0]:p_arrival}
                Late[sta]={fname[0]:LAT}
                Lone[sta]={fname[0]:LON}
                Dep[sta]={fname[0]:DEP}
                mag={fname[0]:MAG}
                
    sourcepara_df['Tartget Events']=[fname[0]]
    sourcepara_df['Origtime'] =list((ev_info[ev_info['Event ID']==fname[0]]['Origin time']).values)[0]
    sourcepara_df['LON'] = list(ev_info[ev_info['Event ID'] == fname[0]]['Lon'])[0]
    sourcepara_df['LAT'] = list(ev_info[ev_info['Event ID'] == fname[0]]['Lat'])[0]
    sourcepara_df['DEP'] = list(ev_info[ev_info['Event ID'] == fname[0]]['Dep'])[0]
    
    return Orig,Sarri,Parri,Late,Lone,Dep,mag,sourcepara_df

def effective_sta_EGF(method,data_path,all_stas,events):
    stations=all_stas.copy()
    Main_event=events[0]
    EGF_event=events[1]
    all_stations=[]
    for station in stations:
        path=os.path.join(data_path,f'{station}')
        if EGF_event:
            try:
                file1=glob.glob(os.path.join(path,f'{Main_event}_*'))
                file2=glob.glob(os.path.join(path,f'{EGF_event}_*'))
                if len(file1)!=0 and len(file2)!=0:
                    all_stations.append(station)
            except:
                traceback.print_exc()
        elif method==1 or not EGF_event:
            try:
                file=glob.glob(os.path.join(path,f'{Main_event}_*'))
                if len(file)!=0:
                    all_stations.append(station)
            except:
                traceback.print_exc()
    return all_stations

def cal_mag_diff(st1,st2):
    a,b=len(st1),len(st2)
    paz_wa = {'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],'zeros': [0 + 0j], 'gain': 1.0, 'sensitivity': 2080}
    st1.detrend(type='linear')
    st2.detrend(type='linear')
    st1_start=st1[0].stats.starttime+st1[0].stats.sac.t1
    stime1=st1[0].stats.starttime+st1[0].stats.sac.t2
    st2_start= st2[0].stats.starttime+st2[0].stats.sac.t1
    stime2= st2[0].stats.starttime+st2[0].stats.sac.t2
    try:
        if a>1:
            for tr in st1.select(component="Z"):
                st1.remove(tr)
        if b>1:
            for tr in st2.select(component="Z"):
                st2.remove(tr)
    except:
        pass
    st1.trim(starttime = stime1 - 0.5 ,endtime=stime1 + (stime1  -st1_start)+0.5)
    st2.trim(starttime = stime2 - 0.5 ,endtime=stime2 + (stime2  -st2_start)+0.5)
    st1.filter('bandpass', freqmin=1, freqmax=20, corners=4, zerophase=True)
    st2.filter('bandpass', freqmin=1, freqmax=20, corners=4, zerophase=True)
    st1.simulate(paz_remove=None, paz_simulate=paz_wa, taper=True, taper_fraction=0.02)
    st2.simulate(paz_remove=None, paz_simulate=paz_wa, taper=True, taper_fraction=0.02)
    datatre1 = st1[0].data
    datatre2 = st2[0].data
    try:
        if a>1 and b>1:
            datatrn1 = st1[1].data
            datatrn2 = st2[1].data
            amp1 = (np.max(datatre1) + np.abs(np.min(datatre1)) + np.abs(np.min(datatrn1)) + np.max(datatrn1)) / 4  
            amp2 = (np.max(datatre2) + np.abs(np.min(datatre2)) + np.abs(np.min(datatrn2)) + np.max(datatrn2)) / 4  
        else:
            amp1 = (np.max(datatre1) + np.abs(np.min(datatre1)))/2
            amp2 = (np.max(datatre2) + np.abs(np.min(datatre2)))/2
    except:
        amp1 = (np.max(datatre1) + np.abs(np.min(datatre1)))/2
        amp2 = (np.max(datatre2) + np.abs(np.min(datatre2)))/2
        traceback.print_exc()
    
    mag_diff =  np.log10(np.mean(amp1) /np.mean(amp2) ) #Bei wang et al,2105
    
    return mag_diff

def ev_baz(paths,Late,Lone,st,evtype,*channel):
    #print('-------------------------ev_bz start--------------------------')
    resp_path=paths['resp_path']
    stationlist = {}
    if resp_path:
        try:
            respf = read_inventory(resp_path + '/*{0}*{1}*'.format(st[0].stats.station,channel[0]))
        except:
            try:
                respf = read_inventory(resp_path + '/*{0}*'.format(st[0].stats.station))
            except:
                respf = read_inventory(resp_path + '/*{0}*'.format(st[0].stats.network))
        for j in respf._networks:
            for i in j:
                stationlist[i._code] = {}
                stationlist[i._code]['lat'] = i._latitude
                stationlist[i._code]['lon'] = i._longitude
                stationlist[i._code]['elev'] = i._elevation/1000.
                stationlist[i._code]['pre_filt'] = []
        stationlist = stationlist
    
    else:
        for tr in st:
            station_code = tr.stats.station
            if station_code not in stationlist:
                stationlist[station_code] = {
                    'lat': tr.stats.coordinates.latitude,
                    'lon': tr.stats.coordinates.longitude,
                    'elev': tr.stats.coordinates.elevation / 1000.0,
                    'pre_filt': []
                }
            response = tr.stats.response
            stationlist[station_code]['response'] = response
    
    late = Late[st[0].stats.station][evtype]
    lone = Lone[st[0].stats.station][evtype]

    lats = stationlist[st[0].stats.station.strip()]['lat']
    lons = stationlist[st[0].stats.station.strip()]['lon']
    
    try:
        baz=st[0].stats.sac.baz
    except:
        gref = Geod(ellps='WGS84')
        _,baz,dist = gref.inv(lone,late,lons,lats,radians=False)
    if baz < 0:
        baz = baz + 360
    #print('-------------------------ev_bz end--------------------------')
    return baz

def apply_free_surface_correction(evla, evlo, evdp, vp_custom, vs_custom, waveform_stream, tr_list, inventory, event_id, apply_surface_corr):
    """
    Applies free-surface correction to 3-component waveform data based on 
    methods from Aki & Richards (1980), House & Boatwright (1980), 
    Kennett (1991), and Kim et al. (1997). Uses the iasp91 model by default.

    Parameters:
    -----------
    evla : dict
        Dictionary of event latitudes keyed by station and event ID.
    evlo : dict
        Dictionary of event longitudes keyed by station and event ID.
    evdp : dict
        Dictionary of event depths keyed by station and event ID.
    vp_custom : float or None
        Custom Vp value in km/s. If None, defaults to 6 km/s.
    vs_custom : float or None
        Custom Vs value in km/s. If None, defaults to Vp/1.73.
    waveform_stream : obspy.Stream
        3-component (ZRT) waveform stream.
    tr_list : list of obspy.Trace
        List of traces (must include at least one to extract station metadata).
    inventory : obspy Inventory or Client response
        Inventory or response object to extract station coordinates.
    event_id : int or str
        ID of the earthquake event.

    Returns:
    --------
    obspy.Stream
        Corrected 3-component stream.
    """

    # Extract components
    t_comp = waveform_stream.select(component='T')
    r_comp = waveform_stream.select(component='R')
    z_comp = waveform_stream.select(component='Z')
    corrected_stream = t_comp + r_comp + z_comp

    evid_str = str(event_id)

    # Get station coordinates
    net_sta_cha = tr_list[0].stats.network.strip() + '.' + \
                  tr_list[0].stats.station.strip() + '..' + \
                  tr_list[0].stats.channel.strip()

    sta_coords = inventory.get_coordinates(net_sta_cha, tr_list[0].stats.starttime)
    sta_lat = sta_coords['latitude']
    sta_lon = sta_coords['longitude']

    eq_lat = evla[z_comp[0].stats.station][evid_str]
    eq_lon = evlo[z_comp[0].stats.station][evid_str]
    eq_dep = evdp[z_comp[0].stats.station][evid_str]

    # Velocity values
    vp = vp_custom/1000 if vp_custom is not None else 6.0
    vs = vs_custom/1000 if vs_custom is not None else vp / 1.73

    # Hypocentral distance
    d_surface=gps2dist_azimuth(eq_lat, eq_lon, sta_lat, sta_lon)[0]/1000
    r=np.sqrt(d_surface**2 + eq_dep**2)
    
    if apply_surface_corr.lower() == 'yes':
        epi_dist = loc2deg(eq_lat, eq_lon, sta_lat, sta_lon)
        # TauP model setup
        tau_model = TauPyModel(model="iasp91")
        arrivals = tau_model.get_travel_times(source_depth_in_km=eq_dep,
                                              distance_in_degree=epi_dist,
                                              phase_list=['PP'])
        Z = z_comp[0].data
        R = r_comp[0].data
        T = t_comp[0].data
        if arrivals:
            rayp_sec_per_rad = arrivals[0].ray_param
            rayp_sec_per_km = rayp_sec_per_rad / 6371  # approx conversion
    
            A_coef = (1 - 2 * (vs ** 2) * (rayp_sec_per_km ** 2)) / (2 * np.sqrt(1 - (vs ** 2) * (rayp_sec_per_km ** 2)))
            B_coef = vs * rayp_sec_per_km
            C_coef = (1 - 2 * (vs ** 2) * (rayp_sec_per_km ** 2)) / (2 * np.sqrt(1 - (vp ** 2) * (rayp_sec_per_km ** 2)))
            D_coef = (vs ** 2) * rayp_sec_per_km / vp
            if rayp_sec_per_km < 1 / vp:
                corrected_Z = C_coef * Z + D_coef * R
                corrected_R = A_coef * R - B_coef * Z
                corrected_T = 0.5 * T
            else:
                Z_hilb = np.imag(hilbert(Z))
                corrected_Z = C_coef * Z_hilb + D_coef * R
                corrected_R = A_coef * R - B_coef * Z_hilb
                corrected_T = 0.5 * T
    
        elif not arrivals:
            # print('Apply free surface correction when the rayp_sec_per_km is approximately zero.')
            A_coef=0.5;B_coef=0;C_coef=0.5;D_coef=0
            corrected_Z = C_coef * Z + D_coef * R
            corrected_R = A_coef * R - B_coef * Z
            corrected_T = 0.5 * T
            # warnings.warn(f'Could not determine ray parameter for event {event_id} at station {waveform_stream[0].stats.station}',
            #               UserWarning)
            
        # Replace data in stream
        corrected_stream.select(component='Z')[0].data = corrected_Z
        corrected_stream.select(component='R')[0].data = corrected_R
        corrected_stream.select(component='T')[0].data = corrected_T
    
    return corrected_stream, r

def rm_instru_resp(apply_resp, apply_surface_corr, vp_value, vs_value, eq_lat, eq_lon, eq_depth, trace_list, backazimuth, response_inventory, eq_tag, output_type):
    '''
    Applies instrument response removal and free-surface correction to waveform data.

    Parameters:
    -----------
    apply_resp : str
        Whether to remove instrument response ('yes'/'no')
    apply_surface_corr : str
        Whether to apply free-surface correction ('yes'/'no')
    vp_value : float
        P-wave velocity for surface correction
    vs_value : float
        S-wave velocity for surface correction
    eq_lat, eq_lon, eq_depth : dict
        Event information dictionaries keyed by station and event id
    trace_list : obspy Stream
        Original waveform stream
    backazimuth : float
        Back-azimuth in degrees
    response_inventory : obspy Inventory or Client
        Station metadata for response correction
    eq_tag : int or str
        Event identifier
    output_type : str
        Output waveform type ('DISP', 'VEL', or 'ACC')

    Returns:
    --------
    obspy Stream
        Processed waveform stream
    '''
    
    stream_copy = trace_list.copy()
    
    # Retrieve pre-filter from metadata dict (custom implementation assumed)
    try:
        pre_filter = trace_list[0].stats.get('pre_filt', None)
    except:
        pre_filter = None

    # === Instrument Response Removal ===
    if apply_resp.lower() == 'yes':
        for trace in trace_list:
            try:
                if response_inventory:
                    trace.remove_response(inventory=response_inventory,
                                          pre_filt=pre_filter,
                                          output=output_type)
                elif hasattr(trace.stats, 'response') and trace.stats.response:
                    trace.remove_response(pre_filt=pre_filter,
                                          output=output_type)
                else:
                    print(f"No response metadata for {trace.id}. Skipped.")
            except Exception as e:
                warnings.warn(f"Error removing response for {trace.id}: {e}", UserWarning)

    # === ZNE Rotation ===
    if hasattr(trace_list[0].stats, 'response'):
        trace_list.rotate(method="->ZNE", inventory=trace_list[0].stats.response)
    elif response_inventory:
        trace_list.rotate(method="->ZNE", inventory=response_inventory)
    else:
        print(f"No response metadata to rotate components for {eq_tag}.")
    
    # === NE->RT Rotation ===
    try:
        # north = trace_list.select(component='N')[0]
        # east = trace_list.select(component='E')[0]
        # rt_stream = Stream([north, east])
        # rt_stream.trim(starttime=max(north.stats.starttime, east.stats.starttime),
        #                endtime=min(north.stats.endtime, east.stats.endtime))
        # rt_stream.rotate('NE->RT', back_azimuth=backazimuth)

        # for i, trace in enumerate(trace_list):
        #     if trace.stats.channel.strip()[-1] == 'N':
        #         trace.data = rt_stream[0].data
        #     elif trace.stats.channel.strip()[-1] == 'E':
        #         trace.data = rt_stream[1].data
                
        starttimes=[chan.stats.starttime for chan in trace_list]
        endtimes=[chan.stats.endtime for chan in trace_list]
        trace_list.trim(starttime=max(starttimes),endtime=min(endtimes))
        trace_list.rotate('NE->RT', back_azimuth=backazimuth)
    except:
        trace_list.rotate(method="NE->RT", back_azimuth=backazimuth)

    # === Free-Surface Correction ===
    # if apply_surface_corr.lower() == 'yes':
    trace_list, R = apply_free_surface_correction(eq_lat, eq_lon, eq_depth,
                                               vp_value, vs_value,
                                               trace_list, stream_copy,
                                               response_inventory, eq_tag, apply_surface_corr)
    # trace_list.rotate(method="RT->NE", back_azimuth=backazimuth)
    return trace_list, R

def Corrss1(tem1,tra1):
    tem1.filter('bandpass',freqmin=1,freqmax=20,corners=4,zerophase=True)
    tra1.filter('bandpass',freqmin=1,freqmax=20,corners=4,zerophase=True)
    starttime=tra1[0].stats.starttime
    delta=tem1[0].stats.delta
    tem1 = np.array(tem1[0].data)
    tra1 = np.array(tra1[0].data)
    tra_leng  = len(tra1)
    tem_leng  = len(tem1)
    time_lags = tra_leng - tem_leng + 1
    
    # demean for the templates
    b1 = tem1 - np.mean(tem1)

    Corr1_val = []
    
    for i in range(time_lags):
        
        a1 = tra1[i:(tem_leng+i)]-np.mean(tra1[i:(tem_leng+i)]) 
        stdev1 = (np.sum(a1 ** 2)) ** 0.5 * (np.sum(b1 ** 2)) ** 0.5
        
        if stdev1!= 0:
            corr1 = np.sum(a1*b1)/stdev1
        else:
            corr1 = 0           

        Corr1_val.append(corr1)
    Corr1_val=np.array(Corr1_val)
    index=np.argmax(Corr1_val)
    time_max_coeff = starttime+index*delta
    return time_max_coeff,Corr1_val[index],np.array(Corr1_val)  

def extract_signal_and_noise(config_paths, do_resp_correction, do_surface_correction,
                             vp_value, vs_value, eq_lat_dict, eq_lon_dict, eq_depth_dict,
                             add_time, coda_ratio, sig_win,
                             waveform_file, eq_id, origin_time, p_arrival, s_arrival,
                             noise_win, for_plot, target_phase, correct_instr, back_az, **kwargs):
    '''
    Extracts windowed waveform segments for signal and noise.

    Parameters:
    -----------
    config_paths : dict
        Contains 'resp_path' key with path to instrument response files.
    do_resp_correction : str
        'yes' or 'no', whether to remove instrument response.
    do_surface_correction : str
        'yes' or 'no', whether to apply free-surface correction.
    vp_value, vs_value : float
        Velocity values for free-surface correction.
    eq_lat_dict, eq_lon_dict, eq_depth_dict : dict
        Dictionaries keyed by station and event ID.
    trace_stream : obspy Stream
        Input waveform stream.
    back_az : float
        Event-station back-azimuth.
    add_time : float
        Time buffer before/after noise/signal windows.
    coda_ratio : float
        Ratio to compute coda window start from S arrival.
    sig_win : float
        Signal window length (sec).
    waveform_file : str
        Path to waveform file.
    eq_id : str or int
        Event identifier.
    origin_time : UTCDateTime
        Event origin time.
    p_arrival, s_arrival : UTCDateTime
        Phase arrival times.
    noise_win : float
        Noise window length (sec).
    for_plot : bool
        Whether for plotting only.
    target_phase : str
        Phase type ('P', 'S', or 'CODA').
    correct_instr : bool
        Whether to return corrected versions.

    Returns:
    --------
    raw_sig : Stream
        Raw signal waveform
    raw_noise : Stream
        Raw noise waveform
    noise_start : UTCDateTime
        Start time of noise window
    corr_sig : Stream
        Corrected signal waveform
    corr_noise : Stream
        Corrected noise waveform
    signal_start : UTCDateTime
        Start time of signal window
    '''
    
    # --- Initialization ---
    resp_dir = config_paths.get('resp_path', None)
    raw_sig, raw_noise = None, None
    corr_sig, corr_noise = None, None
    noise_start, signal_start = None, None
    stream = read(waveform_file)

    stream.detrend(type='demean')
    stream.detrend(type='linear')
    stream.taper(max_percentage=0.05, max_length=5.0)

    # --- Load response file ---
    resp_inv = None
    if resp_dir:
        try:
            resp_inv = read_inventory(resp_dir + f'/*{stream[0].stats.station}*{kwargs["chan"]}*')
        except:
            try:
                resp_inv = read_inventory(resp_dir + f'/*{stream[0].stats.station}*')
            except:
                resp_inv = read_inventory(resp_dir + f'/*{stream[0].stats.network}*')
    
    # --- Adjust timing windows ---
    sig_win += stream[0].stats.delta
    noise_win += stream[0].stats.delta

    # --- Determine start time of signal window ---
    if s_arrival:
        if target_phase.upper() == 'S':
            signal_start = s_arrival
        elif target_phase.upper() == 'CODA':
            signal_start = s_arrival + (coda_ratio - 1) * (s_arrival - origin_time)
    if p_arrival:
        if target_phase.upper() == 'P':
            signal_start = p_arrival

    # --- Full processing ---
    if not for_plot:
        raw_sig = stream.copy()
        if correct_instr:
            raw_sig, R = rm_instru_resp(do_resp_correction, do_surface_correction,
                                                vp_value, vs_value,
                                                eq_lat_dict, eq_lon_dict, eq_depth_dict,
                                                raw_sig.copy(), back_az, resp_inv,
                                                eq_id, 'DISP')

        corr_sig, R = rm_instru_resp('yes', 'yes',vp_value, vs_value,
                                             eq_lat_dict, eq_lon_dict, eq_depth_dict,
                                             stream.copy(), back_az, resp_inv,
                                             eq_id, 'DISP')

        corr_noise = corr_sig.copy()
        corr_for_moment = corr_sig.copy()
        raw_noise = raw_sig.copy()

        raw_sig.trim(starttime=signal_start, endtime=signal_start + sig_win)
        corr_sig.trim(starttime=signal_start, endtime=signal_start + sig_win)
        corr_for_moment.trim(starttime=s_arrival, endtime=s_arrival + sig_win)
        raw_noise.trim(starttime=p_arrival - (noise_win + add_time), endtime=p_arrival - add_time)
        corr_noise.trim(starttime=p_arrival - (noise_win + add_time), endtime=p_arrival - add_time)
        noise_start = p_arrival - (noise_win + add_time)

    # --- Plotting mode ---
    elif for_plot:
        corr_sig, R = rm_instru_resp('no', do_surface_correction,
                                             vp_value, vs_value,
                                             eq_lat_dict, eq_lon_dict, eq_depth_dict,
                                             stream.copy(), back_az, resp_inv,
                                             eq_id, 'DISP')
        raw_sig = corr_sig.copy()
        corr_noise = corr_sig.copy()
        corr_for_moment = None
        raw_noise = corr_sig.copy()

        if s_arrival and not p_arrival:
            fig_duration = coda_ratio * (s_arrival - origin_time) + sig_win
            corr_sig.trim(starttime=s_arrival - (noise_win + add_time),
                          endtime=origin_time + fig_duration + 1)
            noise_start = origin_time - 1

        elif p_arrival and s_arrival:
            fig_duration = coda_ratio * (s_arrival - origin_time) + sig_win
            corr_sig.trim(starttime=p_arrival - (noise_win + add_time) - 0.5,
                          endtime=origin_time + fig_duration + 1)
            noise_start = p_arrival - (noise_win + add_time)
            corr_noise.trim(starttime=noise_start, endtime=p_arrival - add_time)
            raw_noise.trim(starttime=noise_start, endtime=p_arrival - add_time)

        elif p_arrival and not s_arrival:
            fig_duration = sig_win + 4
            corr_sig.trim(starttime=p_arrival - (noise_win + add_time) - 0.5,
                          endtime=p_arrival + fig_duration + 1)
            noise_start = p_arrival - (noise_win + add_time)

    return raw_sig, raw_noise, noise_start, corr_sig, corr_noise, corr_for_moment, signal_start, R

def compute_mt_spectrum(taper_count, window_length, apply_resp_removal,
                        signal_evt1, noise_evt1, signal_evt2, noise_evt2,
                        corr_for_moment, phase_type, window_overlap):

    '''
    Computes multitaper spectral estimates for signal and noise waveforms of two events.

    Parameters:
    -----------
    signal_evt1 : obspy Stream
        Signal waveform for event 1
    noise_evt1 : obspy Stream
        Noise waveform for event 1
    signal_evt2 : obspy Stream
        Signal waveform for event 2
    noise_evt2 : obspy Stream
        Noise waveform for event 2
    taper_count : int
        Number of tapers for multitaper method
    window_length : float
        Length of the time window for spectral estimation (sec)
    apply_resp_removal : str
        'yes' or 'no' to apply instrument response correction
    phase_type : str
        Type of seismic phase ('P', 'S', or 'CODA')
    window_overlap : float
        Overlap between windows (as fraction, e.g., 0.5)

    Returns:
    --------
    snr_full : np.ndarray
        SNR without response correction
    freq_full : np.ndarray
        Frequency array for SNR without response correction
    signal_spec : np.ndarray
        Signal spectrum
    noise_spec : np.ndarray
        Noise spectrum
    snr_resp : np.ndarray
        SNR with instrument response removed
    freq_resp : np.ndarray
        Frequency array for response-corrected spectra
    signal_spec_resp : np.ndarray
        Signal spectrum with response correction
    noise_spec_resp : np.ndarray
        Noise spectrum with response correction
    '''

    # Internal mapping
    fact_resp = 1.0e9 if apply_resp_removal.lower() == 'yes' else 1.0
    factor_no_resp = 1.0e9
    snr_full = freq_full = signal_spec = noise_spec = None
    snr_resp = freq_resp = signal_spec_resp = noise_spec_resp = None

    def detrend_all(stream):
        stream.detrend('demean')
        stream.detrend('linear')

    def process_spectrum(stream, noise, factor, tapers):
        power_sig = {}
        freq_sig = {}
        power_noise = {}
        sta_spec={}
        for idx, tr in enumerate(stream):
            tbw = (tapers + 1) / 2
            sig_mult = np.multiply(tr.data, factor, dtype=float)
            noise_mult = np.multiply(noise[idx].data, factor, dtype=float)
            
            _, freq_sig[idx], power, _ = multitaper.mtspec.spectrogram(
                data=sig_mult, dt=tr.stats.delta, twin=window_length, olap=window_overlap,
                nw=tbw, kspec=tapers, fmin=0, iadapt=0)
            _, _, p_noise, _ = multitaper.mtspec.spectrogram(
                data=noise_mult, dt=noise[idx].stats.delta, twin=window_length, olap=0,
                nw=tbw, kspec=tapers, fmin=0, iadapt=0)
            
            power_sig[idx] = np.median(np.sqrt(power), axis=1)
            power_noise[idx] = np.sqrt(p_noise)
            sta_spec[tr.stats.channel]=np.median(np.sqrt(power), axis=1)
            
            
        power_sig = np.array(list(power_sig.values()))
        power_noise = np.array(list(power_noise.values()))
        power_sig = power_sig.reshape(power_sig.shape[0], power_sig.shape[1])
        power_noise = power_noise.reshape(power_noise.shape[0], power_noise.shape[1])
        signal_avg = np.median(power_sig, axis=0)
        noise_avg = np.median(power_noise, axis=0)
        freq_vals = np.squeeze(freq_sig[0])
        sta_freq = freq_vals
        snr = np.divide(signal_avg, noise_avg, dtype='float64')
        return snr, freq_vals, signal_avg, noise_avg, sta_freq, sta_spec

    def match_length(signal_spec,noise_spec):   
        # Match length by padding if needed
        len_sig = len(signal_spec)
        len_noise = len(noise_spec)
        if len_sig < len_noise:
            pad_len = len_noise - len_sig
            signal_spec = np.pad(signal_spec, (0, pad_len), 'edge')
        elif len_noise < len_sig:
            pad_len = len_sig - len_noise
            noise_spec = np.pad(noise_spec, (0, pad_len), 'edge')
        return signal_spec,noise_spec
    
    # --- Process raw or response-corrected spectra ---
    if signal_evt1:
        detrend_all(signal_evt1)
        detrend_all(noise_evt1)
        snr_resp, freq_resp, signal_spec_resp, noise_spec_resp, _, _ = process_spectrum(
            signal_evt1, noise_evt1, fact_resp, taper_count)
        signal_spec_resp,noise_spec_resp=match_length(signal_spec_resp,noise_spec_resp)
   
    # --- Process instrument-corrected spectra ---
    if signal_evt2:
        detrend_all(signal_evt2)
        detrend_all(noise_evt2)
        snr_full, freq_full, signal_spec, noise_spec, sta_freq, sta_spec = process_spectrum(
            signal_evt2, noise_evt2, factor_no_resp, taper_count)
        signal_spec,noise_spec=match_length(signal_spec,noise_spec)
    
    if signal_evt2 and corr_for_moment:
        detrend_all(corr_for_moment)
        detrend_all(noise_evt2)
        _, _, _, _, sta_freq, sta_spec = process_spectrum(
            corr_for_moment, noise_evt2, factor_no_resp, taper_count)
        
    # Trim frequency range
    if freq_full is not None:
        high_idx = np.where(freq_full >= max(freq_full))[0][0]
        if signal_evt1:
            freq_resp = freq_resp[:high_idx]
            snr_resp = snr_resp[:high_idx]
            signal_spec_resp = signal_spec_resp[:high_idx]
            noise_spec_resp = noise_spec_resp[:high_idx]
        if signal_evt2:
            freq_full = freq_full[:high_idx]
            snr_full = snr_full[:high_idx]
            signal_spec = signal_spec[:high_idx]
            noise_spec = noise_spec[:high_idx]

    return snr_full, freq_full, signal_spec, noise_spec, snr_resp, freq_resp, signal_spec_resp, noise_spec_resp, sta_freq, sta_spec

def extract_spectral_data(paths, T_coda, time_buffer, wave_len, taper_count, window_length, apply_response_correction, s_arrivals, p_arrivals, origin_times,\
                          event_lat, event_lon, event_depth, fs_apply, fs_vp, fs_vs, waveform_path, event_type, window_span, rm_instr_response, wave_type, window_overlap, channels):

    """
    Extracts signal & noise waveforms and computes their spectra.

    Parameters:
    -----------
    waveform_path : str
        Path to waveform file
    event_type : str
        'small' or 'large' event identifier
    window_span : float
        Time window length in seconds
    wave_type : str
        'P', 'S', or 'CODA'
    rm_instr_response : bool
        Whether to remove instrument response
    Other parameters relate to station/event metadata and waveform processing settings.

    Returns:
    --------
    Tuple containing:
    - SNR with response
    - Frequency array with response
    - Signal spectrum with response
    - Noise spectrum with response
    - SNR without response
    - Frequency array without response
    - Signal spectrum without response
    - Noise spectrum without response
    - Phase travel time
    - Signal stream (obspy Stream)
    """

    snr_with_resp = freq_with_resp = spec_signal = spec_noise = None
    snr_no_resp = freq_no_resp = spec_signal_nr = spec_noise_nr = None
    phase_arrival = None
    signal_stream = []

    st = read(waveform_path)

    orig_time = None
    p_arr = None
    s_arr = None
    corr_for_moment = None

    station_id = st[0].stats.station.strip()

    if s_arrivals and station_id in s_arrivals:
        s_arr = s_arrivals[station_id][event_type]
        orig_time = origin_times[station_id][event_type]

    if p_arrivals and station_id in p_arrivals:
        p_arr = p_arrivals[station_id][event_type]
        orig_time = origin_times[station_id][event_type]

    back_azimuth = ev_baz(paths, event_lat, event_lon, st, event_type)

    if wave_type.upper() == 'S' and s_arr:
        phase_arrival = s_arr - orig_time
        signal_stream, noise_stream, _, sig_stream_nr, noise_stream_nr, _, _, R = extract_signal_and_noise(
            paths, apply_response_correction,
            fs_apply, fs_vp, fs_vs,
            event_lat, event_lon, event_depth,
            time_buffer, T_coda, wave_len,
            waveform_path, event_type, orig_time, p_arr, s_arr,
            window_span, False, 'S', rm_instr_response, back_azimuth, chan=channels)

    elif wave_type.upper() == 'P' and p_arr:
        phase_arrival = p_arr - orig_time
        signal_stream, noise_stream, _, sig_stream_nr, noise_stream_nr, _, _, R = extract_signal_and_noise(
            paths, apply_response_correction,
            fs_apply, fs_vp, fs_vs,
            event_lat, event_lon, event_depth,
            time_buffer, T_coda, wave_len,
            waveform_path, event_type, orig_time, p_arr, s_arr,
            window_span, False, 'P', rm_instr_response, back_azimuth, chan=channels)

    elif wave_type.upper() == 'CODA':
        phase_arrival = (s_arr - orig_time) * T_coda
        signal_stream, noise_stream, _, sig_stream_nr, noise_stream_nr, corr_for_moment, _, R = extract_signal_and_noise(
            paths, apply_response_correction,
            fs_apply, fs_vp, fs_vs,
            event_lat, event_lon, event_depth,
            time_buffer, T_coda, wave_len,
            waveform_path, event_type, orig_time, p_arr, s_arr,
            window_span, False, 'Coda', rm_instr_response, back_azimuth, chan=channels)

    if len(signal_stream) != 0 or sig_stream_nr:
        snr_no_resp, freq_no_resp, spec_signal_nr, spec_noise_nr, \
        snr_with_resp, freq_with_resp, spec_signal, spec_noise, sta_freq, sta_spec = compute_mt_spectrum(
            taper_count, window_length, apply_response_correction,
            signal_stream, noise_stream, sig_stream_nr, noise_stream_nr,
            corr_for_moment, wave_type, window_overlap)

    return snr_no_resp, freq_no_resp, spec_signal_nr, spec_noise_nr, snr_with_resp, freq_with_resp, spec_signal, spec_noise, phase_arrival, signal_stream, sta_freq, sta_spec, R

def extract_snr_qualified_range(threshold, spec_a, spec_b, snr_a, snr_b, freq_a, freq_b, noise_a, noise_b):
    """
    Identify frequency ranges where signal-to-noise ratio (SNR) exceeds the defined threshold.

    Parameters:
    -------------
    threshold : float
        Minimum acceptable SNR value
    spec_a : ndarray
        Spectrum of primary signal
    spec_b : ndarray or None
        Spectrum of secondary signal (for spectral ratio), or None
    snr_a : ndarray
        SNR values of primary signal
    snr_b : ndarray or None
        SNR values of secondary signal
    freq_a : ndarray
        Frequency array corresponding to spec_a
    freq_b : ndarray
        Frequency array corresponding to spec_b (same as freq_a if not None)
    noise_a : ndarray
        Noise spectrum of primary signal
    noise_b : ndarray
        Noise spectrum of secondary signal (if applicable)

    Returns:
    -------------
    win_a : ndarray
        spec_a clipped within qualified frequency band
    win_b : ndarray or None
        spec_b clipped within qualified frequency band (if available)
    f_win_a : ndarray
        frequency band for spec_a passing the SNR threshold
    f_win_b : ndarray or None
        frequency band for spec_b passing the SNR threshold
    n_win_a : ndarray
        noise_a clipped within qualified frequency band
    n_win_b : ndarray or None
        noise_b clipped within qualified frequency band
    """
    
    win_a = win_b = f_win_a = f_win_b = n_win_a = n_win_b = None
    skip_flag = False

    if snr_a[0] < threshold:
        try:
            low_a = np.where(snr_a >= threshold)[0][0]
        except:
            low_a = 0
        try:
            upper_a = snr_a[low_a:]
            high_a = np.where(upper_a >= threshold)[0][-1] + low_a
        except:
            high_a = len(snr_a) - 1
    else:
        low_a = np.where(snr_a >= threshold)[0][0]
        high_a = len(snr_a) - 1

    if spec_b is not None:
        if snr_b[0] < threshold:
            try:
                low_b = np.where(snr_b >= threshold)[0][0]
            except:
                low_b = 0
            try:
                upper_b = snr_b[low_b:]
                high_b = np.where(upper_b >= threshold)[0][-1] + low_b
            except:
                high_b = len(snr_b) - 1
        else:
            low_b = np.where(snr_b >= threshold)[0][0]
            high_b = len(snr_b) - 1

        f_low = max(low_a, low_b)
        f_high = min(high_a, high_b)

        f_win_a = freq_a[f_low:f_high]
        f_win_b = freq_b[f_low:f_high]
        win_a = spec_a[f_low:f_high]
        win_b = spec_b[f_low:f_high]
        n_win_a = noise_a[f_low:f_high]
        n_win_b = noise_b[f_low:f_high]
    else:
        if not skip_flag:
            f_win_a = freq_a[low_a:high_a]
            win_a = spec_a[low_a:high_a]
            n_win_a = noise_a[low_a:high_a]

    return win_a, win_b, f_win_a, f_win_b, n_win_a, n_win_b

def compute_spectra(paths, T_coda, t_offset, coda_duration, num_mtapers, win_length, apply_resp_corr, snr_threshold, s_arr, p_arr, origin_time, latitude, 
			longitude, depth, apply_fs_correction,  fs_vp, fs_vs, eventfile_main, eventfile_egf, eventid_main, eventid_egf, wavetype, overlap, channel):
    '''
    Compute signal spectra, spectral ratios (if two events), and perform SNR-based frequency trimming

    Parameters:
    -------------
    eventfile_main: waveform file of main event
    eventfile_egf: waveform file of EGF event (optional)
    wavetype: 'P' or 'S'

    Returns:
    ----------
    spec_ratio: spectral ratio between main and EGF event
    freq_ratio: frequency array for spectral ratio
    egf_freq_uncorrected: frequency bins of EGF signal (uncorrected)
    egf_signal_uncorrected: signal spectrum of EGF (uncorrected)
    egf_noise_uncorrected: noise spectrum of EGF (uncorrected)
    main_freq_uncorrected: frequency bins of main signal (uncorrected)
    main_signal_uncorrected: signal spectrum of main event (uncorrected)
    main_noise_uncorrected: noise spectrum of main event (uncorrected)
    travel_time_main: travel time for main event
    travel_time_egf: travel time for EGF event
    win_used: time window used (seconds)
    main_signal_corrected: signal spectrum of main event (corrected)
    egf_signal_corrected: signal spectrum of EGF event (corrected)
    '''
    win_used = win_length
    spec_ratio = []
    egf_freq_uncorrected = []
    egf_signal_uncorrected = []
    egf_noise_uncorrected = []
    main_freq_uncorrected = []
    main_signal_uncorrected = []
    main_noise_uncorrected = []
    travel_time_main = None
    travel_time_egf = None
    

    print(f'Analyzing main event: {eventid_main}\n' + '='*72)

    snr_main, freq_main, sig_main, noise_main, snr_raw_main, freq_raw_main, sig_raw_main, noise_raw_main, travel_time_main, _, sta_freq, sta_spec, R = extract_spectral_data(
        paths, T_coda, t_offset, coda_duration, num_mtapers, win_length, apply_resp_corr,
        s_arr, p_arr, origin_time, latitude, longitude, depth, apply_fs_correction, fs_vp, fs_vs,
        eventfile_main, eventid_main, win_used, True, wavetype, overlap, channel)

    main_signal_uncorrected = sig_raw_main
    main_freq_uncorrected = freq_raw_main
    main_noise_uncorrected = noise_raw_main

    if eventfile_egf:
        print('='*72 + f'\nAnalyzing EGF event: {eventid_egf}\n' + '='*72)
        snr_egf, freq_egf, sig_egf, noise_egf, snr_raw_egf, freq_raw_egf, sig_raw_egf, noise_raw_egf, travel_time_egf, _, _, _, _ = extract_spectral_data(
            paths, T_coda, t_offset, coda_duration, num_mtapers, win_length, apply_resp_corr,
            s_arr, p_arr, origin_time, latitude, longitude, depth, apply_fs_correction, fs_vp, fs_vs,
            eventfile_egf, eventid_egf, win_used, True, wavetype, overlap, channel)

        egf_signal_uncorrected = sig_raw_egf
        egf_freq_uncorrected = freq_raw_egf
        egf_noise_uncorrected = noise_raw_egf
    else:
        sig_raw_egf = None
        snr_raw_egf = None
        freq_raw_egf = None
        noise_raw_egf = None

    trimmed_main, trimmed_egf, trimmed_freq_main, trimmed_freq_egf, _, _ = extract_snr_qualified_range(
        snr_threshold, sig_raw_main, sig_raw_egf, snr_raw_main, snr_raw_egf, freq_raw_main, freq_raw_egf, noise_raw_main, noise_raw_egf)

    try:
        spec_ratio = np.divide(trimmed_main, trimmed_egf, dtype=float)
    except:
        pass

    if len(spec_ratio) == 0:
        print('No valid EGF spectra available; switching to direct spectral fitting.')

    return spec_ratio, trimmed_freq_main, egf_freq_uncorrected, egf_signal_uncorrected, egf_noise_uncorrected, main_freq_uncorrected, main_signal_uncorrected,\
    		main_noise_uncorrected, travel_time_main, travel_time_egf, sta_freq, sta_spec, R
