import os
import re
import wave
import argparse
import yaml
import tgt
import pandas as pd
import numpy as np
import librosa as lr
from mipcat.signal.timeseries_generator import ts_from_pickle

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
ch_pref = ['barrel','external','farfield']

def get_base_timeseries(tspath):
    """
    returns frequency and amplitude time series from a reference signal

    Arguments:
    - tspath : path to the timeseries file
    """
    tsl = ts_from_pickle(tspath)
    for ch in ch_pref:
        ftsl = [ts for ts in tsl if ts.label == f'{ch}_f0']
        if len(ftsl)>0:
            fts = ftsl[0]
            ats = [ts for ts in tsl if ts.label == f'{ch}_ampl'][0]
            break
    return fts, ats

def get_notes(note_tier, clip_tier, ts_path, melody, nseg=3):
    """
    returns a note list for a timeseries extracted from a recording
    """
    fts, ats = get_base_timeseries(ts_path)
    notes = []
    
    for n in note_tier:
        err = 0
        try:
            f0q = fts.percentile([25,50,75],from_time=n.start_time, to_time=n.end_time)
            f0 = f0q[1]
            f0v = f0q[2]-f0q[0]
        except IndexError:
            err+=16
            f0 = np.nan
            f0v = np.nan
            
        try:
            a0q = ats.percentile([25,50,75],from_time=n.start_time, to_time=n.end_time)
            amax = (ats.max_time(from_time=n.start_time, to_time=n.end_time)-n.start_time)/n.duration()
            a0 = a0q[1]
            a0v = a0q[2]-a0q[0]
        except IndexError:
            err+=32
            a0 = np.nan
            a0v = np.nan
        try:
            aseg = []
            for ii in range(nseg):
                st = n.start_time + (n.end_time-n.start_time)/nseg*ii
                et = n.start_time + (n.end_time-n.start_time)/nseg*(ii+1)
                aseg.append(ats.percentile(50,from_time=st, to_time=et))
                
                
        except IndexError:
            err+=64
            aseg = [np.nan]*3

        try:
            nmel_id = int(re.findall('^\d+',n.text)[0])
        except IndexError:
            #print('Missing note id')
            err+=1
            nmel_id = 10000
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
            exp_f0 = lr.note_to_hz(pitch)/2**(2/12)
            beats = mel_note['duration']
        except TypeError:
            err+=4
            pitch = None
            exp_f0=None
            beats = None
        mid_time = (n.start_time+n.end_time)/2
        try:
            nexcpt = [ii for ii,c in enumerate(clip_tier) if c.start_time<mid_time and c.end_time>mid_time][0]
        except (TypeError, IndexError):
            err+=8
            nexcpt=None
        
        note= {'start':n.start_time,
               'end':n.end_time,
               'duration':n.duration(),
               'f0':f0,
               'f0_std':f0v,
               'ampl':a0,
               'ampl_var':a0v,
               'ampl_max':amax,
               'exp_f0':exp_f0,
               'excpt_nbr':nexcpt,
               'mel_note_id':nmel_id,
               'pitch':pitch,
               'beats':beats,
               'err':err}
        for ii,seg in enumerate(aseg):
            note[f"aseg{ii}"]=seg
        notes.append(note)
    return notes


def read_melody_list():
    
    mel_path_list = (__file__.split(os.path.sep)[:-2]+['mipcat','resources','melodies.yaml'])
    melody_path = ('/'+os.path.join(*mel_path_list))

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


def build_note_table(wmldf, tgroot):
    wmldf['ts_path'] = ''

    notes = []


    for irow, row in wmldf.iterrows():
        print(row['filename'])
        #wpath = os.path.join(wavroot,row['filename'].strip('/'))
        tgpath = os.path.join(tgroot,row['filename'].strip('/').replace('.wav','_notes.TextGrid'))
        
            
        tspath = os.path.join(tgroot,row['filename'].strip('/').replace('.wav','_ts.pickle'))
        if not os.path.isfile(tspath):
            print(f'Missing {tspath}')
            
        if not os.path.isfile(tgpath):
            print(f'Missing {tgpath}')
            continue
        else:
            wmldf.loc[irow,'tg_path'] = tgpath
            tg = tgt.read_textgrid(tgpath)
            
            
        clip_tier, note_tier = get_note_excerpt_tiers(tg) 
        try:
            melody = melodies[row['tune']]
        except KeyError:
            melody = None
        n = get_notes(note_tier, clip_tier, tspath, melody)
        for nn in n:
            nn['wav_id']=irow
        notes.extend(n)   

    return pd.DataFrame(notes)
    

if __name__ == '__main__':
    args = parse_args()

    melodies = read_melody_list()

    # Read recording list with melody information
    wmldf = pd.read_csv(args.filename, index_col=0)
    notedf = build_note_table(wmldf, args.root)

    notedf.to_csv(args.output)
    

