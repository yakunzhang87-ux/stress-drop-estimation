#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:08:21 2025

@author: Zhang et al.
"""

import os
import pandas as pd
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees, gps2dist_azimuth
import time
import math
import traceback
from math import log10

def config(chan,velocity_model,pre_event_time,past_event_time):
    
    CONFIG = {
        "processing": {
            "pre_event_time": pre_event_time,    # minutes
            "post_event_time": past_event_time,   # minute
            "channel": chan,
            "filter_params": {
                "enable": False,
                "type": "bandpass",
                "freqmin": 1,
                "freqmax": 20,
                "corners": 4,
                "zerophase": True
            },
            "velocity_model": velocity_model
        }
    }
    return CONFIG
def main(eqtype,events_path,out_path,station_path,center):
    
    start_time = time.time()
    
    setup_environment(eqtype,out_path)

    events_df = load_events(events_path)
    station_df =pd.read_csv(station_path)
    
    velocity_model =  TauPyModel(model=CONFIG["processing"]["velocity_model"])
    
    metadatas={}
    # results = []
    for i, event in events_df.iterrows():
        try:
            _,metadata= process_single_event(eqtype,event,station_df,velocity_model,out_path,center)
            # results.append(result)
            if i==0:
                metadatas=metadata#pd.DataFrame([metadata])
            else:
                metadatas=pd.concat([metadatas,metadata],ignore_index=True)
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing event {event['Event ID']}: {str(e)}")
    
    save_final_results(eqtype,metadatas,out_path)
    
    print(f"Total processing time: {(time.time()-start_time)/60:.1f} minutes")

def setup_environment(eqtype,out_path):
    os.makedirs(out_path, exist_ok=True)

def load_events(file_path):
    df = pd.read_csv(file_path)
    df["Origin time"] = df["Origin time"].apply(UTCDateTime)
    df["Event ID"] = df["Event ID"].astype(str)
    return df 


def process_single_event(eqtype,event, station_df,velocity_model,out_path,center):
    print(f"\nProcessing {eqtype} {event['Event ID']} ({event['Origin time']})")   
    metadatas=pd.DataFrame()
    mag_data = []
    for _, row in station_df.iterrows():
        network = row['Networks']
        station = row['Stations']
        try:
            day_start = UTCDateTime(UTCDateTime(event["Origin time"]))         
            endtime = day_start + 86400
            client=Client(center)
            inventory = client.get_stations(network=network, station=station,
                                  location="*", channel=CONFIG["processing"]["channel"],
                                  starttime= day_start,
                                  endtime=endtime,
                                  level="response")
            station_info = get_stations_information(event, inventory,velocity_model)
            waveform = download_waveform(eqtype,event,station_info,center)
            if waveform is not None:
                processed,st= process_waveform(eqtype,waveform, inventory)
                magnitude = calculate_magnitude(eqtype,event,processed,inventory,station_info)
                if magnitude is not None:
                    mag_data.append(magnitude)
                    metadata=save_results(eqtype,st, event, station_info, magnitude,out_path)
                    metadatas=pd.concat([metadatas,pd.DataFrame([metadata])],ignore_index=True)
                    
        except:
            traceback.print_exc()
            print(f'Error processing event {event["Event ID"]} on station:{station}')
    
    
    avg_mag = round(np.mean(mag_data),2) if mag_data else np.nan
    print(f"{eqtype} {event['Event ID']} average magnitude: {avg_mag:.2f}")
    
    return {
        "eqid": event["Event ID"],
        "time": event["Origin time"],
        "magnitude": avg_mag,
        "stations_processed": len(mag_data)
    },metadatas

def get_stations_information(event, inventory,model):
    for net in inventory:
        for sta in net:
            distance = calculate_distance(
                event["Lat"], event["Lon"],
                sta.latitude, sta.longitude
            )
            arrivals = calculate_arrival_times(
                model, event["Dep"], distance
            )
            station_info={
                "network": net.code,
                "station": sta.code,
                "latitude": sta.latitude,
                "longitude": sta.longitude,
                "distance": distance,
                "P_arrival": float("{:.2f}".format(arrivals["P"])),
                "S_arrival": float("{:.2f}".format(arrivals["S"]))
            }
    return station_info

def calculate_distance(lat1, lon1, lat2, lon2):
    return gps2dist_azimuth(lat1, lon1, lat2, lon2)[0] / 1000

def calculate_arrival_times(model, depth, distance):
    degrees = kilometer2degrees(distance)
    
    try:
        P = model.get_travel_times(depth, degrees, phase_list=["P","p"])[0]
        S = model.get_travel_times(depth, degrees, phase_list=["S","s"])[0]
    except IndexError:
        raise ValueError("No phase arrival found")
    
    return {"P": P.time, "S": S.time}
def obspy_to_sac_header(stream, inventory):
    stream.attach_response(inventory)
    for tr in stream:
        # Add stats for SAC format
        tr.stats.sac = dict()
        # Add station and channel information
        metadata = inventory.get_channel_metadata(tr.id)
        tr.stats.sac.stla = metadata["latitude"]
        tr.stats.sac.stlo = metadata["longitude"]
        tr.stats.sac.stel = metadata["elevation"]
        tr.stats.sac.stdp = metadata["local_depth"]
        tr.stats.sac.cmpaz = metadata["azimuth"]
        tr.stats.sac.cmpinc = metadata["dip"] + 90 # different definitions
def download_waveform(eqtype,event, station,center):
    # CONFIG=config()
    
    start = UTCDateTime(event["Origin time"]) + station["P_arrival"] - 60*float(CONFIG["processing"]["pre_event_time"])
    end = UTCDateTime(event["Origin time"]) +station["S_arrival"] + 60*float(CONFIG["processing"]["post_event_time"])
    try:
        client=Client(center)
        stream = client.get_waveforms(
            network=station["network"],
            station=station["station"],
            location="*",
            channel=CONFIG["processing"]["channel"],
            starttime=start,
            endtime=end,
            attach_response=True
        )
        
        print(f'Success on station: {station["station"]}')
        return stream
    except Exception as e:
        print(f'Failed on station: {station["station"]}')
        print(e)
        return None

def process_waveform(eqtype,stream, inventory):
    # CONFIG=config()
    
    obspy_to_sac_header(stream, inventory)
    stream.merge()
    
    st=stream.copy()
    stream.remove_response(inventory=inventory)
    
    
    if CONFIG["processing"]["filter_params"]["enable"]:
        params = CONFIG["processing"]["filter_params"]
        # stream.detrend("demean")
        stream.filter(
            params["type"],
            freqmin=params["freqmin"],
            freqmax=params["freqmax"],
            corners=params["corners"],
            zerophase=params["zerophase"]
        )   
    
    stream.rotate("->ZNE", inventory=inventory)
    return stream,st

def calculate_magnitude(eqtype,event,stream0, inventory,station_info):
    
    try:
        ptime = UTCDateTime(event["Origin time"]) + station_info["P_arrival"]
        stime = UTCDateTime(event["Origin time"]) + station_info["S_arrival"]
        stream = stream0.copy()
        stream.trim(starttime = stime - 0.5 ,endtime=stime + (stime  -ptime)+0.5)
        # stream.remove_response(inventory=inventory, output="VEL")
        east = stream.select(component="E")[0]
        north = stream.select(component="N")[0]
    except IndexError:
        raise ValueError("Missing components for magnitude calculation")
    paz_wa = {'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],
                'zeros': [0 + 0j], 'gain': 1, 'sensitivity': 2080}
    params = CONFIG["processing"]["filter_params"]
    #if you didn't remove the reponse, please use below two lines.
    east.filter(
            params["type"],
            freqmin=params["freqmin"],
            freqmax=params["freqmax"],
            corners=params["corners"],
            zerophase=params["zerophase"]
        )   
    north.filter(
            params["type"],
            freqmin=params["freqmin"],
            freqmax=params["freqmax"],
            corners=params["corners"],
            zerophase=params["zerophase"]
        )   
    east.simulate(paz_remove = None, paz_simulate = paz_wa, taper=True,taper_fraction=0.02)
    north.simulate(paz_remove = None, paz_simulate = paz_wa, taper=True,taper_fraction=0.02)
    
    amp_e = (abs(np.max(east.data)) + abs(np.min(east.data))) / 2
    amp_n = (abs(np.max(north.data)) + abs(np.min(north.data))) / 2
    amp = (amp_e + amp_n) / 2 * (10 ** (3))
    distance = math.sqrt(station_info["distance"]**2+float(event["Dep"])**2)
    # Hutton-Boore formula
    if amp <= 0:
        return None
    local_mag =log10(amp) + 1.11*log10(distance/100) + 0.00189*(distance-100) + 3.0
    local_mag = round(local_mag, 1)
    return local_mag

def save_results(eqtype,stream, event, station, magnitude,out_path):
    
    output_dir = os.path.join(out_path,station["station"])
    os.makedirs(output_dir, exist_ok=True)
 
    for tr in stream:
        if eqtype == "EGF":
            if 'EGF' not in str(event['Event ID']).upper():
                filename = f"EGF{event['Event ID']}_{tr.id}.SAC"
            else:
                filename = f"{event['Event ID']}_{tr.id}.SAC"
        else:
            filename = f"{event['Event ID']}_{tr.id}.SAC"
        tr.stats.sac.b = 0
        tr.stats.sac.t1 = UTCDateTime(event["Origin time"]) + station["P_arrival"] - tr.stats.starttime
        tr.stats.sac.t2 = UTCDateTime(event["Origin time"]) + station["S_arrival"] - tr.stats.starttime
        
        tr.write(os.path.join(output_dir, filename), format="SAC")

    
    metadata = {
        "event_id": event["Event ID"],
        "station": f"{station['network']}.{station['station']}",
        "distance": station["distance"],
        "magnitude": magnitude,
        "p_arrival": station["P_arrival"],
        "s_arrival": station["S_arrival"]
    }

    return metadata

def save_final_results(eqtype,results,out_path):
    df = pd.DataFrame(results)
    output_path = os.path.join(out_path, f'{eqtype}_final_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nFinal results saved to {output_path}")
    
def download_repsonse(events_path,resp_path,station_path,center):
    events_df = load_events(events_path)
    station_df =pd.read_csv(station_path)
    nets=station_df['Networks'].values
    stas=station_df['Stations'].values
    t1=min(events_df['Origin time'].values)-1000
    t2=max(events_df['Origin time'].values)+1000
    os.makedirs(resp_path,exist_ok=True)
    for n,s in zip(nets,stas):
        client=Client(center)
        response=client.get_stations(network=n, station=s, location="*", channel='*',
                                   starttime=UTCDateTime(t1),endtime=UTCDateTime(t2),level="response")
        response.write(os.path.join(resp_path,f'{n}..{s}.xml'), format="STATIONXML")

def download_data(Target_events,data_path,all_stations,resp_path,chan,data_center='IRIS',EGF_events=None,**paras):
    if 'vel_model' in paras:
        velocity_model=paras['vel_model']
    else:
        velocity_model="ak135"
    if 'pre_event_time' in paras:
        pre_event_time=paras['pre_event_time']
    else:
        pre_event_time=0.5
    if 'past_event_time' in paras:
        past_event_time=paras['past_event_time']
    else:
        past_event_time=1.5
    
    global CONFIG
    CONFIG=config(chan,velocity_model,pre_event_time,past_event_time)
    
    try:
        print('Downloading Main events...')
        main('Target',Target_events,data_path,all_stations,data_center)
        download_repsonse(Target_events, resp_path, all_stations, data_center)
    except:
        traceback.print_exc()
   
    if EGF_events:
        try:
            print('Downloading EGF events...')
            main('EGF',EGF_events,data_path,all_stations,data_center)
            download_repsonse(EGF_events, resp_path, all_stations, data_center)
        except:
            traceback.print_exc()
    
    return None
