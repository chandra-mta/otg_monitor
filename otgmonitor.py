import matplotlib.pyplot as plt
import numpy as np
import warnings
import Ska.engarchive.fetch as fetch
from astropy.table import Table
import itertools
from Chandra.Time import DateTime

OTG_MSIDS = ['4MP28AV', '4MP28BV', # MCE A/B: +28 VOLT MONITOR
             '4MP5AV', '4MP5BV', # MCE A/B: +5 VOLT MONITOR
             '4HENLAX', '4HENLBX', # MCE A/B: HETG ENABLE LOGIC STATUS MONITOR
             '4LENLAX', '4LENLBX', # MCE A/B: LETG ENABLE LOGIC STATUS MONITOR
             '4HEXRAX', '4HEXRBX', # MCE A/B: HETG EXECUTE RELAY STATUS MONITOR
             '4LEXRAX', '4LEXRBX', # MCE A/B: LETG EXECUTE RELAY STATUS MONITOR
             '4HILSA','4HRLSA','4HILSB', '4HRLSB',  # MCE A/B: HETG LIMIT SWITCH 2A MONITOR (INSERTED/RETRACTED)
             '4LILSA','4LRLSA','4LILSBD', '4LRLSBD',  # MCE A/B: LETG LIMIT SWITCH 2A MONITOR (INSERTED/RETRACTED)
             '4HPOSARO', '4HPOSBRO', '4LPOSARO', '4LPOSBRO'] # HETG/LETG ROTATION ANGLE POSITION MONITOR A/B

ANGLE_MON = {'LETG': '4LPOSBRO', 'HETG': '4HPOSBRO'}

# THRESH = {'NONE':3.9, 'LETG':4.5, 'HETG':4.5}

THRESH_V = 4.5 # V


def otgmon(time_start, time_stop):
    """
    """
    # Fetch data (from ska)
    # One can implement also fetchind data from the dump files
    # or other sources
    dat = fetch_msidset_from_ska(OTG_MSIDS, start=time_start, stop=time_stop)
    
    # Identifiy main moves
    main_grating_states = get_main_grating_states(dat)

    # Analyze only completed grating transitions
    bad_times = []
    i0, i1 = 0, main_grating_states
    
    if main_grating_states[0] != 'NONE':
        i0 = list(main_grating_states).index('NONE')
        bad_times.append(f'{time_start} {DateTime(dat[OTG_MSIDS[0]].times[i0]).date}')
        
    if main_grating_states[-1] != 'NONE':
        i1 = max(idx for idx, val in enumerate(main_grating_states) if val == 'NONE')
        bad_times.append(f'{DateTime(dat[OTG_MSIDS[0]].times[i1]).date} {time_stop}')

    if len(bad_times) > 0:
        dat.filter_bad_times(table=bad_times)
        main_grating_states = main_grating_states[i0: i1]

    out = {'tstart': [],
           'tstop': [],
           'grating': [],
           'angle_start': [],
           'angle_stop': [],
           'direction': [],
           'move_tstart': [],
           'move_tstop': [],
           'duration': []}
    
    # Execute if there were main grating moves
    if len(set(main_grating_states)) > 1:
    
        angles = get_angles(dat, main_grating_states)
        angles_start, angles_stop = find_start_stop_vals(angles, main_grating_states, plus1=True)
        
        times = dat[OTG_MSIDS[0]].times
        tstarts, tstops = find_start_stop_vals(times, main_grating_states, plus1=False)
        
        # Find all moves (long and short)
        all_grating_states = get_all_grating_states(dat, main_grating_states)
        
        t0s, t1s = find_start_stop_vals(times, all_grating_states, plus1=False)

        grat0s, grat1s = find_start_stop_vals(all_grating_states, all_grating_states, plus1=False)
        
        for tstart, tstop, angle_start, angle_stop in zip(tstarts,
                                                          tstops,
                                                          angles_start,
                                                          angles_stop):
            for t0, t1, grat1 in zip(t0s, t1s, grat1s):
                if t0 >= tstart and t1 <= tstop:
                    out['tstart'].append(DateTime(tstart).date)
                    out['tstop'].append(DateTime(tstop).date)
                    out['grating'].append(grat1)
                    out['angle_start'].append(angle_start)
                    out['angle_stop'].append(angle_stop)
                    if angle_start > angle_stop:
                        out['direction'].append('INSR')
                    else:
                        out['direction'].append('RETR')
                    out['move_tstart'].append(DateTime(t0).date)
                    out['move_tstop'].append(DateTime(t1).date)
                    out['duration'].append(t1 - t0)

    return out

    
def get_main_grating_states(dat):
    """
    Identify main grating states based on 4HENLBX, 4LENLBX
    """
    times = dat[OTG_MSIDS[0]].times
    nn = len(times)
    out = {}
    
    # Initialize grating state vector
    grating_states = np.full((nn), 'NONE')

    # Identify main grating moves
    hetg = dat['4HENLBX'].vals == 'ENAB'
    letg = dat['4LENLBX'].vals == 'ENAB'
    grating_states[hetg] = 'HETG'
    grating_states[letg] = 'LETG'
    
    return grating_states


def get_all_grating_states(dat, grating_states=None):
    
    if grating_states is None:
        grating_states = get_main_grating_states(dat)

    # Find elements with volt monitor vals above the thresh
    idx = np.array([dat[msid].vals > THRESH_V for msid in OTG_MSIDS[:4]], dtype=bool)
    idx = np.prod(idx, axis=0, dtype=bool)
    
    # If voltage below the threshold -> NONE
    grating_states[~idx] = 'NONE'
    
    return grating_states


def get_angles(dat, grating_states):
    """
    Do this using the main grating states
    """
    angles = np.zeros(len(grating_states))
    for grat in ('HETG', 'LETG'):
        ok = grating_states == grat
        angles[ok] = dat[ANGLE_MON[grat]].vals[ok]
    
    return angles


def find_start_stop_vals(vector, grating_states, plus1=False):
    """
    Find vector vals at the times the grating state changed
    """
    idx = np.where(grating_states[1:] != grating_states[:-1])[0]
    if plus1:
        idx[::2] = idx[::2] + 1
    change_vals = vector[idx]
    
    if len(change_vals) % 2 > 0:
        change_vals = change_vals[:-1]
    
    change_vals = change_vals.reshape((len(change_vals) // 2, 2)).T
    
    start_vals = change_vals[0]
    stop_vals = change_vals[1]
    
    if not isinstance(start_vals, list):
        start_vals = list(start_vals)
        stop_vals = list(stop_vals)
    
    return start_vals, stop_vals


def fetch_msidset_from_ska(msidset, start, stop):
    """
    """
    dat = fetch.MSIDset(msidset, start=start, stop=stop)
    return dat
