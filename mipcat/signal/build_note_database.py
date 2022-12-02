import os
import re
import logging
import argparse
import pickle
import tgt
import traceback
import yaml
import numpy as np
import pandas
import librosa as lr
from scipy.io import wavfile

from timeseries import SampledTimeSeries, TimeSeries 
#import timeseries_generator as tsg
from . import timeseries_generator as tsg

def _angle_percentile(x, pct):
    return np.angle(np.percentile(np.real(x), pct) + 1j * np.percentile(np.imag(x), pct))

def angle_mean(x):
    return np.angle(np.mean(x))

def angle_std(x):
    return min(np.std(np.angle(x)),np.std(np.angle(-x)))
    
def iqr(ts, **kwarg):
    return ts.percentile(75, **kwarg) - ts.percentile(25, **kwarg)

def angle_iqr(ts, half_circle = np.pi, **kwarg):
    return min(iqr(ts, **kwarg), iqr(ts+half_circle, **kwarg))
    
def angle_percentile(ts, pct, from_time, to_time):
    t, v = ts.times_values_in_range(from_time=from_time, to_time=to_time)
    return _angle_percentile(v, pct)

def _base_stats_real(ts, tst, tend):
    label = ts.label
    rdict = {}
    rdict[f"{label}_avg"] = ts.percentile(50, from_time=tst, to_time=tend)
    rdict[f"{label}_var"] = iqr(ts, from_time=tst, to_time=tend)
    t, v = ts.times_values_in_range(from_time=tst, to_time=tend)
    try:
        p = np.polyfit(t-np.min(t), v, 1)
        rdict[f"{label}_trend"] = p[0]
    except (np.linalg.LinAlgError, SystemError):
        pass
    rdict[f"{label}_max_t"] = ts.max_time(from_time=tst, to_time=tend)
    rdict[f"{label}_max_v"] = ts.max(from_time=tst, to_time=tend)
    rdict[f"{label}_min_t"] = ts.min_time(from_time=tst, to_time=tend)
    rdict[f"{label}_min_v"] = ts.min(from_time=tst, to_time=tend)

    # divide the note in 3 parts and calculate averages
    tlims = np.linspace(tst,tend,4)
    for ti ,(tts, tte) in enumerate(zip(tlims[:-1],tlims[1:]),1):
        rdict[f"{label}_t{ti}"] = ts.percentile(50, from_time=tts, to_time=tte)

    return rdict

def _base_stats_angle(ts, tst, tend):
    ts_arg = ts.apply(np.angle)
    ts_arg.label = ts.label+'_arg'
    label = ts_arg.label
    rdict = {}
    avg = angle_percentile(ts, 50, from_time=tst, to_time=tend)
    rdict[f"{label}_avg"] = avg
    # centered time series
    ts_cent = ts_arg.apply(lambda x: np.mod(x-avg+np.pi, 2*np.pi)-np.pi)
    rdict[f"{label}_var"] = iqr(ts_cent, from_time=tst, to_time=tend) 
    t, v = ts_cent.times_values_in_range(from_time=tst, to_time=tend)
    try:
        p = np.polyfit(t-np.min(t), v, 1)
        rdict[f"{label}_trend"] = p[0]
    except np.linalg.LinAlgError:
        pass
    rdict[f"{label}_max_t"] = ts_cent.max_time(from_time=tst, to_time=tend)
    rdict[f"{label}_max_v"] = ts_cent.max(from_time=tst, to_time=tend) + avg
    rdict[f"{label}_min_t"] = ts_cent.min_time(from_time=tst, to_time=tend)
    rdict[f"{label}_min_v"] = ts_cent.min(from_time=tst, to_time=tend) + avg

    # divide the note in 3 parts and calculate averages
    tlims = np.linspace(tst,tend,4)
    for ti ,(tts, tte) in enumerate(zip(tlims[:-1],tlims[1:]),1):
        rdict[f"{label}_t{ti}"] = ts_cent.percentile(50, from_time=tts, to_time=tte) + avg

    return rdict

def apply_base_stats(ts, tst, tend):
    if np.iscomplexobj(ts.v):
        ts_abs = ts.apply(np.abs)
        ts_abs.label = ts.label+'_abs'
        rdict = _base_stats_real(ts_abs, tst, tend)
        rd_angle = _base_stats_angle(ts, tst, tend)
        rdict.update(rd_angle)
        
    else:
        rdict = _base_stats_real(ts, tst, tend)
    return rdict


def get_sounding_boundaries(allts, tst, tend, context_sec=0.2):

    try:
        ii = [x.label == 'barrel_ampl' for x in allts].index(True)
    except ValueError:
        ii = [x.label == 'farfield_ampl' for x in allts].index(True)
    tsi = allts[ii]
    t, v = tsi.times_values_in_range(tst-context_sec, tend+context_sec)
    ti, vi = tsi.times_values_in_range(tst, tend)
    
    # median value in boundaries
    #medv = np.median(v)
    medv = np.percentile(vi,99)

    # left-hand
    minv = tsi.min(from_time=tst-context_sec, to_time=(tend+tst)/2)
    thv = (minv*medv)**.5
    cup, cdown = tsi.crossing_times(thv, from_time=tst-context_sec,
                                     to_time=(tend+tst)/2, interp='nearest')

    sst = cup[-1]
    # right-hand
    minv = tsi.min(from_time=sst,to_time=tend+context_sec)
    thv = (minv*medv)**.5
    _, cdown=tsi.crossing_times(thv,from_time=sst,
                                   to_time=tend+context_sec,interp='nearest')
    return sst, cdown[0]

    
def read_mouthpiece_ts(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return TimeSeries(t=data['t'], v=data['v'], label='mpcover')

def process_notes_in_file(filedict, eg_tier='clip', note_tier='note', 
                          sounding_context_sec=0.2):
    allnotes = []
    allvers = []
    tune = filedict['tune']
    root_dir = filedict['root_dir']
    wfile = os.path.join(root_dir, filedict['filename'].strip('/'))

    #channel_desc = filedict['channels']

    basename = os.path.splitext(wfile)[0]
    ts = tsg.ts_from_pickle(basename+'_ts.pickle')
    mpts_path = basename+'_mouthpiece_ts.pickle'
    try:
        tsmp = read_mouthpiece_ts(mpts_path)
        ts.append(tsmp)
    except FileNotFoundError:
        print(f"Not found {mpts_path}")
    try:
        instrument = filedict['instrument']
    except KeyError:
        logging.warn('Instrument not defined in {basename}')
        instrument = ''

    try:
        tgf = filedict['notes_textgrid']
    except KeyError:
        logging.warn(f'No notes TG in {basename}')
        return [], []
    tg = tgt.read_textgrid(os.path.join(root_dir, tgf.strip('/')))
    egt = tg.get_tier_by_name(eg_tier)
    for ver_nbr, ver in enumerate(egt.annotations):
        verdict = {'wavfile': wfile,
                   'tune': tune,
                   'instrument': instrument,
                   'ver_name': ver.text,
                   'ver_nbr': ver_nbr,
                   'subject_id': filedict['subject_id'],
                   'vers_start': ver.start_time,
                   'vers_end': ver.end_time}
        allvers.append(verdict)
        for this_ts in ts:
            this_dict = apply_base_stats(this_ts, ver.start_time, ver.end_time)
            verdict.update(this_dict)
    
    tier = tg.get_tier_by_name(note_tier)
    prev_start = 0.0
    
    for inote, note in enumerate(tier.annotations):
        text = note.text
        try:
            note_id = int(re.findall(r'^(\d+)', text)[0])
        except IndexError:
            note_id = -1
        eg_name = egt.get_nearest_annotation(note.start_time)[0].text
        mid_time = (note.start_time + note.end_time)/2
        try:
            eg_nbr_in_rec = egt.annotations.index(egt.get_annotations_by_time(mid_time)[0])
        except IndexError:
            logging.warn(f'Clip name not found at {wfile}:{note.start_time}s')
            continue
        
        if inote>0: 
            prev_dur = note.start_time - tier.annotations[inote-1].start_time
        else:
            prev_dur = note.start_time
        
        if inote < len(tier.annotations)-1: 
            next_dur = tier.annotations[inote+1].end_time - note.end_time
        else:
            next_dur = tier.end_time - note.end_time
        
        neighb_dur = min(prev_dur,next_dur)
        
        try:
            if neighb_dur / 2 < sounding_context_sec:
                context_sec = neighb_dur / 2
            else:
                context_sec = sounding_context_sec
            sst, send = get_sounding_boundaries(ts, note.start_time, note.end_time, context_sec=context_sec)
        except (IndexError, ValueError):
            logging.warn(f'Sounding boundaries not found at {wfile}:{note.start_time}s')
            sst = note.start_time
            send = note.end_time

        notedict = {'start': note.start_time,
                    'end': note.end_time,
                    'sound_start': sst,
                    'sound_end': send,
                    'note_id': note_id,
                    'note_label': text,
                    'tune': tune,
                    'subject_id': filedict['subject_id'],
                    'ver_name': eg_name,
                    'ver_nbr': eg_nbr_in_rec,
                    'wavfile': wfile}

        f0chan = ts[[tsi.label.find('f0') > -1 for tsi in ts].index(True)].label
        # for ts_label in [f0chan, 'mouth_dc', 'barrel_ampl', 
        #                  'reed_dc', 'mouth_h01', 'external_ampl',
        #                  'external_cent']:
        #     try:
        #         this_ts = ts[[tsi.label for tsi in ts].index(ts_label)]
        #     except ValueError:
        #         logging.warning(f'Channel {ts_label} not found in {basename}')
        #         this_dict = {}
        #     try:
        #         this_dict = apply_base_stats(this_ts, sst, send)
        #     except IndexError:
        #         logging.warning(f'Note at {note.start_time:.2f} in {ts_label} in {basename} throwed an error.')
        #         this_dict = {}
        #     notedict.update(this_dict)
        for this_ts in ts:
            try:
                this_dict = apply_base_stats(this_ts, sst, send)
                notedict.update(this_dict)
            except IndexError:
                logging.warn(f'Base stats error for {this_ts.label} at {wfile}:{note.start_time}s')
                
        
        allnotes.append(notedict)
    return allnotes, allvers


def build_melodies_table(melodies):
    allmel = []

    for mel in melodies:
        melname = mel
        notes = melodies[melname]['notes']
        nid = 0
        for note in notes:
            # skip silences
            if note['pitch'] == 0:
                allmel[-1]['strict_duration'] = False
                continue
            strict_dur = True
            strong = False
            try:
                susp = note['has_suspension']
                if susp:
                    strict_dur=False
            except KeyError:
                pass
            try:
                strong_flag = note['strong_beat']
                if strong_flag:
                    strong=True
            except KeyError:
                pass
            notedict = {'pitch': note['pitch'],
                        'beats': note['duration'],
                        'tune': mel,
                        'note_id': nid,
                        'strict_duration':strict_dur,
                        'strong_beat':strong}
            allmel.append(notedict)
            nid += 1

    meldf = pandas.DataFrame(allmel)
    idx = meldf['pitch'] != 0 
    meldf.loc[idx,'exp_freq'] = meldf.loc[idx].pitch.apply(lr.note_to_hz)
    meldf["beat_start"] = meldf.groupby("tune").beats.transform(np.cumsum) - meldf.beats
    for mname, mdf in meldf.groupby('tune'):
        strong_beat = mdf[mdf.strong_beat].beat_start.iloc[0]
        bar_nbeats = int(melodies[mname]['bar_duration'])
        idx = mdf[((mdf.beat_start-strong_beat)%bar_nbeats==0)].index
        meldf.loc[idx, 'strong_beat'] = True
    meldf.index.name = 'unid'
    return meldf

def process_runsheets(allruns):
    ri = tsg.run_iterator_from_yaml('allruns.yaml')
    all_notes_list = []
    all_versions_list = []
    for f in ri.iter_files():
        notes_list, versions_list = process_notes_in_file(f)
        all_notes_list.extend(notes_list)
        all_versions_list.extend(versions_list)
        
    return pandas.DataFrame(all_notes_list), pandas.DataFrame(all_versions_list)

    
def process_csv(csvfile,root_dir="."):
    df = pandas.read_csv(csvfile, index_col=0)
    all_notes_list = []
    all_versions_list = []
    for irw, row in df.iterrows():
        filedict = row.to_dict()
        basename, ext = os.path.splitext(filedict['filename'])
        filedict['root_dir'] = root_dir
        filedict['notes_textgrid'] = basename+'_notes.TextGrid'

        try:
            notes_list, versions_list = process_notes_in_file(filedict)
        except Exception:
            logging.warning(f"Failed on file {row['filename']}")
            traceback.print_exc()
            continue
        all_notes_list.extend(notes_list)
        all_versions_list.extend(versions_list)
    return pandas.DataFrame(all_notes_list), pandas.DataFrame(all_versions_list)


def parse_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument('file', help='runsheet (YAML or CSV)')
    parser.add_argument('-m','--melody-file',  help='YAML file containing melodies')
    parser.add_argument('-r','--root-dir',  help='root dir for data', default=".")
    parser.add_argument('-o','--output',  help='output file')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.melody_file) as f:
        melodies = yaml.full_load(f)

    melody_df = build_melodies_table(melodies)
    
    basename, ext = os.path.splitext(args.file)
    if ext.lower() == '.yaml' or ext.lower() == '.yml':
        notes_df, versions_df = process_runsheets(args.file)
    elif ext.lower() == '.csv':
        notes_df, versions_df = process_csv(args.file, root_dir=args.root_dir)
    
    melody_df.to_csv("melody_scores.csv")
    notes_df.to_csv("played_notes.csv")
    versions_df.to_csv("versions_played.csv")



