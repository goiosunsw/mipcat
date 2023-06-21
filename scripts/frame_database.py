import os
import re
import logging
import argparse
import yaml
import tgt
import pandas as pd
import numpy as np
import librosa as lr
from mipcat.signal.timeseries_generator import ts_from_pickle
from mipcat.signal.build_note_database import read_mouthpiece_ts, read_pose_ts

def get_note_excerpt_tiers(tg):
    initials = ['cl','no']
    tier_names = tg.get_tier_names()
    tiers=[]
    for initial in initials:
        try:
            tier_name = [tn for tn in tier_names if tn[:2] == initial][0]
            tier = tg.get_tier_by_name(tier_name)
        except IndexError:
            tier = None
            
        tiers.append(tier)
    return tiers


# Preferred channels for base timeseries (see nex function)
ref_ch_pref = ['barrel','external','farfield']
other_chans = ['mpcover','cl_angle','mouth_dc', 'reed_dc'] + [f"mouth_h{x:02d}" for x in range(1,6)]
desc_list = ['ampl'] + [f"h{x:02d}" for x in range(1,6)]

def load_raw_timeseries(basename):

    ts = []
    tspath = basename+'_ts.pickle'
    mpts_path = basename+'_mouthpiece_ts.pickle'
    try:
        tsmp = read_mouthpiece_ts(mpts_path)
        ts.append(tsmp)
    except FileNotFoundError:
        logging.warning(f"Not found {mpts_path}")

    pose_path = basename+'_pose_ts.pickle'
    try:
        tsp = read_pose_ts(pose_path)
        ts.extend(tsp)
    except FileNotFoundError:
        logging.warning(f"Not found {pose_path}")
    tsl = ts_from_pickle(tspath)
    ts.extend(tsl)
    return ts

def get_timeseries(tspath):
    """
    returns frequency and amplitude time series from a reference signal

    Arguments:
    - tspath : path to the timeseries file
    """
    tsl = load_raw_timeseries(tspath)
    for ch in ref_ch_pref:
        ftsl = [ts for ts in tsl if ts.label == f'{ch}_f0']
        if len(ftsl)>0:
            fts = ftsl[0]
            break
    ts = {'time': fts.t,
          'f0': fts.v}
    time = fts.t
    # ref_ch = ch
    
    for ch_base in ref_ch_pref:
        for desc in desc_list:
            ch = ch_base+'_'+desc
            try:
                chts =  [ts for ts in tsl if ts.label == ch][0]
            except IndexError:
                continue
            v = np.interp(time, chts.t, chts.v)
            ts[ch] = v
    
    for ch in other_chans:
        try:
            chts =  [ts for ts in tsl if ts.label == ch][0]
        except IndexError:
            continue
        v = np.interp(time, chts.t, chts.v)
        ts[ch] = v

    return pd.DataFrame(ts)

def get_timeseries_with_notes(basepath, tgpath, melody):
    err = 0 
    df = get_timeseries(basepath)
    tg = tgt.read_textgrid(tgpath)
    clip_tier, note_tier = get_note_excerpt_tiers(tg) 
    df['midi'] = pd.Series([-1 for x in df.time],dtype='int8')
    df['note_id'] = pd.Series([-1 for x in df.time],dtype='int8')
    df['excpt_id'] = pd.Series(['' for x in df.time],dtype='string')

    for n in note_tier:
        try:
            nmel_id = int(re.findall('^\d+',n.text)[0])
        except IndexError:
            #print('Missing note id')
            err+=1
            nmel_id = -1
        try:
            mel_note = melody['notes'][nmel_id]
        except IndexError:
            #print(f'Worng note id {nmel_id}')
            err+=2
            mel_note = None
        except TypeError:
            mel_note = None
            
        try:
            pitch = mel_note['pitch']
            midi = lr.note_to_midi(pitch)
            exp_f0 = lr.note_to_hz(pitch)/2**(2/12)
            beats = mel_note['duration']
        except TypeError:
            err+=4
            midi = -1
            exp_f0=None
            beats = None


        idx = (df.time >= n.start_time) & (df.time <= n.end_time)
        df.loc[idx,'midi'] = midi
        df.loc[idx,'note_id'] = nmel_id
    
    try:
        for c in clip_tier.intervals:
            idx = (df.time >= c.start_time) & (df.time <= c.end_time)
            df.loc[idx,'excpt_id'] = c.text
    except AttributeError:
        pass

    df['excpt_id'] = df.excpt_id.astype('category')    
    return df


def read_melody_list():
    
    print(__file__.split(os.path.sep))
    mel_path_list = (__file__.split(os.path.sep)[:-2]+['mipcat','resources','melodies.yaml'])
    melody_path = ('/'.join(mel_path_list))
    if os.path.sep == '/':
        melody_path = '/'+melody_path

    with open(melody_path) as f:
        melodies = yaml.safe_load(f)
    
    return melodies
    

def parse_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("filename", help="csv file with recording list")
    ap.add_argument("root", help="root folder for search")
    ap.add_argument("-o", "--output", default="note_list.csv",
                        help="output file")
    return ap.parse_args()


def build_frame_table(wmldf, tgroot, melodies):
    wmldf['ts_path'] = ''

    tables = []


    for irow, row in wmldf.iterrows():
        print(row['filename'])
        #wpath = os.path.join(wavroot,row['filename'].strip('/'))
        tgpath = os.path.join(tgroot,row['filename'].strip('/').replace('.wav','_notes.TextGrid'))
        
            
        tspath = os.path.join(tgroot,row['filename'].strip('/').replace('.wav','_ts.pickle'))
        basepath = os.path.join(tgroot,row['filename'].strip('/').replace('.wav',''))
        if not os.path.isfile(tspath):
            print(f'Missing {tspath}')
            
        if not os.path.isfile(tgpath):
            print(f'Missing {tgpath}')
            continue
        else:
            wmldf.loc[irow,'tg_path'] = tgpath
            tg = tgt.read_textgrid(tgpath)
            
            
        try:
            melody = melodies[row['tune']]
        except KeyError:
            melody = None
        df = get_timeseries_with_notes(basepath, tgpath, melody)
        df['filename'] = row['filename']
        df['subject_id'] = row['subject_id']
        df['tune'] = row['tune']
        tables.append(df) 

    df = pd.concat(tables)
    for column in ['filename','subject_id','tune','excpt_id','midi','note_id']: 
        df[column] = df[column].astype('category')    
    return pd.DataFrame(df)
    

if __name__ == '__main__':
    args = parse_args()

    melodies = read_melody_list()

    # Read recording list with melody information
    wmldf = pd.read_csv(args.filename, index_col=0)
    notedf = build_frame_table(wmldf, args.root, melodies)

    notedf.to_pickle(args.output)
    

