#!/usr/bin/env python

"""
Segment notes from timeseries and (optionally) align with a score
"""
import os
import logging
import argparse
from pathlib import Path
import pickle
import yaml
import numpy as np
import pandas as pd
import scipy.signal as sig
import librosa as lr
import tgt
from timeseries import SampledTimeSeries

def midi_to_note(x):
    return lr.midi_to_note(x).replace('\u266f','#')

def estimate_silence_voice_meanvals(ats, min_pct=1, max_pct=99):
    sil = ats.percentile(min_pct)
    loud = ats.percentile(max_pct)
    loud_med = np.percentile(ats.v[ats.v>(np.sqrt(sil*loud))],50)
    sil_med = np.percentile(ats.v[ats.v<(np.sqrt(sil*loud))],50)
    return sil_med, loud_med


def bf_matcher(x,y,dist=lambda x,y: np.abs(x-y),maxd=None):
    """Brute force matcher: matches values of x to nearest values of y 
                            with maximum distance

    Args:
        x (float ndarray): First array for matching
        y (float ndarray): Second array for matching
        dist (function): distance function. Defaults to absolute difference.
        maxd ([type], optional): Maximum distance. Defaults to infinity.

    Returns:
        array of tuples with matching indices in X and Y
    """
    dists = dist(x[:,np.newaxis],y[np.newaxis,:])
    if maxd is None:
        maxd = np.max(dists)
    nmatches = min(len(x), len(y))
    pairs = []
    while True:
        ix, iy = np.unravel_index(np.argmin(dists),dists.shape)
        if dists[ix,iy]>=maxd:
            break
        pairs.append((ix,iy))
        dists[ix,:] = maxd
        dists[:,iy] = maxd
    return pairs


def nw_matcher(x, y, match = 1, mismatch = 1, gap = 1):
    """Needleman-Wunsch sequence matcher
       (adapted from https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5)

        Matches add to a general score whereas gaps and mismatches subtract from it.
        The algorithm tries to maximise the score.
        Typically if mismtches are preferred to gaps, increase the gap score
         
    Args:
        x (numpy 1D array): first sequence
        y (numpy 1D array): second sequence 
        match (int, optional): Score for a match. Defaults to 1.
        mismatch (int, optional): Score for a mismatch. Defaults to 1.
        gap (int, optional): Score for a gap. Defaults to 1.

    Returns:
        aligned_x, aligned_y, x_indices, y_indices
        (The aligned sequences with NaNs in gaps,
         and the corresponding indices in the original sequences)
    """
    
    nx = len(x)
    ny = len(y)
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx, nx + 1)
    F[0,:] = np.linspace(0, -ny, ny + 1)
    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4
    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    ix = []
    iy = []
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            ix.append(i-1)
            iy.append(j-1)
            rx.append(x[i-1])
            ry.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            ix.append(i-1)
            iy.append(None)
            rx.append(x[i-1])
            ry.append(np.nan)
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            ix.append(None)
            iy.append(j-1)
            rx.append(np.nan)
            ry.append(y[j-1])
            j -= 1
    # Reverse the strings.
    ix = ix[::-1]
    iy = iy[::-1]
    rx = rx[::-1]
    ry = ry[::-1]
    return rx,ry,ix,iy


def segment_notes_in_timeseries(fts, ats, median_filter_wind_sec=0.1,
                                ampl_db_pk_prominence=[3,None],
                                ampl_db_pk_height=None):
    """Segment notes in an amplitude and frequency timeseries

    Args:
        fts (frequency time series): timeseries with frequency values and times
        ats (amplitude time series): timeseries with amplitude values and times
        median_filter_wind_sec (float): window length in seconds for median filtering
        ampl_db_pk_prominence (list of 2 floats): min and max prominence 
                                                 (default [3,None])
        ampl_db_pk_height (list of 2 floats): max and min height of peaks 
                                                 (default: auto-estimeate)
        """
    amp = ats.v
    f0 = fts.v
    try:
        assert(len(fts.v)==len(ats.v))
    except AssertionError:
        if len(fts.v)<len(ats.v):
            ats = SampledTimeSeries(t=fts.t, v=ats[fts.t])
        else:
            fts = SampledTimeSeries(t=ats.t, v=fts[ats.t])
            

    adb = 20*np.log10(amp)
    
    sil_val, voice_val = estimate_silence_voice_meanvals(ats)
    athr = 10*np.log10(sil_val * voice_val)
    logging.debug(f'Global amplitude threshold for voicing: {athr} dB')

    if ampl_db_pk_height is None:
        ampl_db_pk_height = 20*np.log10(np.asarray([sil_val*.01, voice_val*100]))
        
    logging.debug(f'Dip min/max vals in dB: {min(ampl_db_pk_height):.2f} -- {max(ampl_db_pk_height):.2f}')
    # amplitude dips
    pks,_ = sig.find_peaks(-adb,prominence=ampl_db_pk_prominence,
                              height=-np.asarray(np.flipud(ampl_db_pk_height)))
    logging.debug(f'Found {len(pks)} amplitude dips for {len(f0)} frames')

    pitch = 12*np.log2(f0/440)
    stadj = estimate_ref_pitch(fts, ats)
    a4 = 440*2**(stadj/12)
    logging.debug(f'semitone adj: {stadj:.4f}, A4 = {a4:.3f} Hz')
    # recalculate pitch based on new reference
    pitch = 12*np.log2(f0/a4)
    # Round pitch (integer semitones)
    rpitch = np.round(pitch)
    # nans break median filter
    rpitch[np.isnan(rpitch)]=-100

    # Filtered pitch to avoid short jumps
    medwind_len = int(2*np.round(median_filter_wind_sec/fts.dt/2)+1)
    assert(medwind_len>0)
    logging.debug(f'Median window: {medwind_len} frames')
    fpitch = sig.medfilt(rpitch,medwind_len)

    # pitch jumps
    pitch_jumps = np.flatnonzero(np.diff(fpitch)!=0)

    # match pitch jumps to corresponding amplitude dips
    # tolerance is medwind_len
    m = bf_matcher(pks, pitch_jumps, maxd=medwind_len)
    mdf = pd.DataFrame([{'amp_pk_idx':ai,'df_idx':fi,'amp_pos':pks[ai],
                         'df_pos':pitch_jumps[fi]} for ai,fi in m])
    mdf.sort_values('df_idx').reset_index()

    # build extended table with unmatched peaks and jumps
    adf = pd.DataFrame({'amp_pos':pks,'amp_val':amp[pks]}).rename_axis('amp_pk_idx')
    pdf = pd.DataFrame({'df_pos':pitch_jumps}).rename_axis('df_pk_idx')
    matchdf = pd.DataFrame(m,columns=['amp_pk_idx','df_pk_idx'])
    bdf=pdf.merge(adf.merge(matchdf, on='amp_pk_idx', how='outer'), on='df_pk_idx', how='outer')
    bdf['mean_pos'] = bdf[['df_pos','amp_pos']].mean(skipna=True,axis=1)
    bdf=bdf.sort_values('mean_pos').reset_index()

    # build table with note characteristics
    notes = []
    for idx in bdf[:-1].index:
        row_l = bdf.loc[idx]
        row_r = bdf.loc[idx+1]
        ist = int(np.nanmax([row_l.amp_pos,row_l.df_pos]))
        iend = int(np.nanmin([row_r.amp_pos,row_r.df_pos]))
        pn = pitch[ist:iend]
        pn = pn[~np.isnan(pn)]
        if len(pn)<1:
            continue
        notes.append({'start':ist,
                    'end':iend,
                    'round_pitch':np.median(fpitch[ist:iend]),
                    'mean_pitch':np.mean(pn),
                    'median_pitch':np.median(pn),
                    'std_pitch':np.std(pn),
                    'iqr_pitch':np.diff(np.percentile(pn,[25,75]))[0],
                    'mean_ampl':np.nanmean(adb[ist:iend]),
                    'std_ampl':np.nanstd(adb[ist:iend])
                    })
                    
    df=pd.DataFrame(notes)
    # convert to times
    df['start'] = df.start*fts.dt
    df['end'] = df.end*fts.dt
    #df['label'] = df.index
    df['duration']=df['end']-df['start']
    df['detune']=df['mean_pitch']-df['round_pitch']

    # discard short notes and invalid pitch
    dff = df[(df.duration>median_filter_wind_sec)&
             (df.round_pitch>-99)&
             (df.mean_ampl>athr)].copy()
    dff.reset_index(inplace=True)
    dff.rename(columns={'index':'orig_index'}, inplace=True)
    dff.index.name = 'index'
    dff['label'] = (dff.index+1).astype(str)
    
    return dff

    
def estimate_ref_pitch(fts, ats):
    """Estimate reference pitch in fts (value of A4)
    Estimates less than a semitone adjustment around 440 Hz

    Args:
        fts (frequency timeseries): Timeseries with frequency values and times
        ats (amplitude time series): timeseries with amplitude values and times
    Returns:
        semitone adjustment
    """
    f0=fts.v
    # convert to semitones
    pitch = 12*np.log2(f0/440)
    pitch[ats.v < ats.percentile(95)/100] = np.nan
    # don't care about octaves
    chroma = (np.mod(pitch,12))
    meanchr = np.mod(np.angle(np.nanmean(np.exp(1j*chroma*2*np.pi)))/2/np.pi+.5,1)-.5
    return meanchr


def read_timeseries(filename, labels=['barrel_f0','barrel_ampl']):
    tsdict = {}
    with open(filename,'rb') as f:
        tsl = pickle.load(f)
        for label, ts in tsl.items():
            if labels is None:
                tsdict[label] = SampledTimeSeries(ts['v'],dt=ts['dt'],t_start=ts['t0'],label=label)
            else:
                if label in labels:
                    tsdict[label] = SampledTimeSeries(ts['v'],dt=ts['dt'],t_start=ts['t0'],label=label)
    return tsdict


def output_silence_db(filename):
    tsdict = read_timeseries(filename, labels=['barrel_f0','barrel_ampl'])
    fts = tsdict['barrel_f0']
    ats = tsdict['barrel_ampl']
    a_sil, a_voice = estimate_silence_voice_meanvals(ats)
    print(f'Silence value: {a_sil}')
    print(f'Voice value: {a_voice}')
    sil_frames = np.sum(ats.v<a_sil)
    voice_frames = np.sum(ats.v>=a_sil)
    print(f'Total time: {ats.t[-1]-ats.t[0]:.4f}')
    print(f'Silence time: {sil_frames*ats.dt:.4f}')
    print(f'Voice time: {voice_frames*ats.dt:.4f}')
    athr = np.sqrt(a_sil*a_voice)
    crossing_up, crossing_down = ats.crossing_times(athr)
    print(f'Voice start: {crossing_up[0]:.4f}')
    print(f'Voice end: {crossing_down[-1]:.4f}')
    print(f'Number of crossings: {len(crossing_up)}')
    
    
def write_textgrid(df, output, tmax=None):
    """
    generates a textgrid with intervals starting in tu and ending in td
    where tu and td are taken from a dataframe containing at least the columns:
    * "start" (in seconds)
    * "end" (in seconds)
    * "label"
    if there is a column called "excerpt" a second tier will be generated 
    containing all notes in excerpt
    """
    #print (intervals)
    if tmax is None:
        tmax = df.start.max()


    tg = tgt.TextGrid()
    inttier = tgt.IntervalTier(start_time=0, end_time=tmax, name="note")
    tg.add_tier(inttier)
    for irow, row in df.iterrows():
        u = row.start
        d = row.end
        text = str(row.label)
        inttier.add_interval(tgt.Interval(u, d, text))

    if "clip" in df.columns:
        cliptier = tgt.IntervalTier(start_time=0, end_time=tmax, name="clip")
        tg.add_tier(cliptier)
        for clip, group in df.groupby('clip'):
            s = group.start.min()
            e = group.end.max()
            text = clip
            cliptier.add_interval(tgt.Interval(s, e, text))
            

    tgt.write_to_file(tg, output)

def write_csv(df, output):
    df.to_csv(output)

def output_segmentation(df, output=None, tmax=None):

    if output:
        file_type = os.path.splitext(output)[-1][1:].lower()
        if file_type == 'textgrid':
            write_textgrid(df, output, tmax=tmax) 
        elif file_type == 'csv':
            write_csv(df,output)
        else:
            logging.error(f'Unrecognised output type {file_type}')
    else:
        from io import StringIO
        output = StringIO()
        df.to_csv(output)
        output.seek(0)
        print(output.read())
    
def segment_notes(filename, output=None):
    
    tsdict = read_timeseries(filename, labels=None)
    fts, ats = get_ref_series(tsdict)
    df = segment_notes_in_timeseries(fts,ats)
    tmax = fts.t[-1]
    return df, tmax 
        
       
def read_mel_js(melody_yaml_file):
    """Read a melody YAML file
    """     
    with open(melody_yaml_file) as f:
        mel_yml = yaml.safe_load(f)

    return mel_yml

def mel_js_to_df(mel_js, melody=None):
    """Select a melody from melody js
       and convert to DataFrame

    Args:
        mel_js ([type]): Melody js in nested dictionary format
        melody ([type], optional): Title of melody (or None if js is single melody)
    """
    if melody is not None:
        mel_js = mel_js[melody]

    meldf = pd.DataFrame(mel_js['notes'])
    meldf = meldf[meldf['pitch'] != 0]
    meldf['round_pitch'] = meldf.pitch.apply(lambda x: lr.note_to_midi(x)-71)
    return meldf

def read_score(melody_yaml_file, melody=None):
    """Reads a score from a YAML score

    Args:
        melody_yaml_file (str): Filename
        melody (str, optional): Title of melody in YAML collection. 
            (defaults to using the whole file as melody)
            
        returns a Dataframe with notes, midi pitches,durations
            and any other per-note information on YAML
    """
    mel_js = read_mel_js(melody_yaml_file)
    meldf = mel_js_to_df(mel_js, melody)
        
    return meldf


def match_melodies(score, played, nrep=1, calc_scores=False):
    """Match score to played melody

    Args:
        score (int iterable): pitch sequences in score
        played (int iterable): played pitch sequence
        nrep (int, optional): number of repetitions of score. Defaults to 1.
        calc_scores (bool, optional): Return number of misses. Defaults to False.

    Returns:
        match dataframe: pandas DataFrame with:
            * score (expected) note (NaN if unmatched)
            * played note (NaN if not on score)
            * index of note in score
            * index of note in played sequence
            * repetition number
    """
    x = np.tile(score, nrep)
    indices = np.arange(len(score))
    sidx = np.tile(indices, nrep)
    repno = np.cumsum(sidx==0)
    rx, ry, ix, iy = nw_matcher(x, played, gap=2)

    ixr = []
    rn  = []
    for ii in ix:
        if ii is None: 
            ixr.append(np.nan)
            rn.append(np.nan)
        else:
            ixr.append(sidx[ii])
            rn.append(repno[ii])
        
    dfmatch = pd.DataFrame({'expected_note':rx,
                            'note':ry,
                            'expected_index':ixr,
                            'found_index':iy,
                            'rep_no':rn})

    if calc_scores:
        # Calculate scores of matches
        ndel = sum([(yy is None) for xx,yy in zip(ix,iy)])
        nins = sum([(xx is None) for xx,yy in zip(ix,iy)])
        ndiff = sum([xx!=yy for xx,yy in zip(rx,ry)])
        return dfmatch, ndiff+nins+ndel
    else:
        return dfmatch
    
        
def guess_nrep(expected, played, min_nrep=1, max_nrep=20):
    reps = np.arange(min_nrep, max_nrep)
    scores = []
    matches = []
    for n in reps:
        match, score = match_melodies(expected, played, nrep=n, calc_scores=True)
        scores.append(score)
        matches.append(match)
    return reps[np.argmin(scores)], matches[np.argmin(scores)]
        

def auto_match_melodies(score_df, played_df):
    """Extract pitch from dataframes, guess number of repetitions
       and match notes

    Args:
        score_df (pandas DataFrme): DataFrame containing a "round_pitch" column
        played_df (pandas DataFrame): DataFrame containing a "round_pitch" column
    """
    score_pitch = score_df.round_pitch.values
    played_pitch = played_df.round_pitch.values

    # First guess of number of repetitions given by quotient of notes
    nrep1 = int(np.round(len(played_pitch)/len(score_pitch)))
    nrep_min = max(1,int(nrep1*.5))
    nrep_max = max(int(nrep1*1.5),3)
    nrep, matchdf = guess_nrep(score_pitch, played_pitch, min_nrep=nrep_min, max_nrep=nrep_max)
    return matchdf
    


def reassign_same_notes(dfr):
    # Keep track of notes to join
    xnote=[]
    prev_note= -1000
    prev_exp = -1000
    exp_note = -1000
    for ii, row in dfr.iterrows():
        if not np.isnan(row.expected_note):
            # Try to find an expected note for group
            exp_note = row.expected_note
        if (row.note == prev_note) & np.isnan(row.expected_note):
            xnote.append(ii)
        else:
            if (not np.isnan(row.expected_note)) & (row.note == prev_note):
                xnote.append(ii)
            if len(xnote)>1:
                print(ii,exp_note,xnote,ynote,row.note)
                tstart = (dfr.loc[xnote[0]].start)
                tend = (dfr.loc[xnote[-1]].end)
                noteidx = xnote[-1]
                dfj = dfr.loc[xnote]
                print(dfj.start.values)
                print(dfj.expected_index.values)
                print(tstart,tend)
                dfr.loc[noteidx,'start'] = tstart
                dfr.loc[noteidx,'end'] = tend
                dfr.loc[noteidx,'found_index'] = dfr.loc[xnote[-1],'found_index']
                dfr.loc[noteidx,'index'] = dfr.loc[xnote[-1],'index']
                dfr.loc[noteidx,'note'] = dfr.loc[xnote[-1],'note']
                dfr.loc[noteidx,'expected_note'] = exp_note
                try:    
                    dfr.loc[noteidx,'expected_index'] = dfj.loc[~dfj.expected_index.isna(),'expected_index'].iloc[0]
                    dfr.drop(xnote[:-1],inplace=True)
                except IndexError:
                    dfr.drop(xnote,inplace=True)
                exp_note=np.nan
                xnote=[]
            else:
                    
                xnote=[ii]
            #for noteidx in xnote:
        prev_note = row.note
        prev_exp = row.expected_note
    return dfr.copy()

def reassign_same_notes(dfr):
    # Keep track of notes to join
    xnote=[]
    prev_note=-1000
    coalesce = False
    for ii, row in dfr.iterrows():
        if (row.note == prev_note) & np.isnan(row.expected_note):
            xnote.append(ii)
        if row.note != prev_note:
            coalesce = True
        if (row.note == prev_note) & (not np.isnan(row.expected_note)):
            xnote.append(ii)
            coalesce = True

        if coalesce:
            if len(xnote)>1:
                print(ii,xnote)
                tstart = (dfr.loc[xnote[0]].start)
                tend = (dfr.loc[xnote[-1]].end)
                noteidx = xnote[-1]
                dfj = dfr.loc[xnote]
                print(dfj.start.values)
                print(dfj.expected_index.values)
                print(tstart,tend)
                dfr.loc[noteidx,'start'] = tstart
                dfr.loc[noteidx,'end'] = tend
                dfr.loc[noteidx,'found_index'] = dfr.loc[xnote[-1],'found_index']
                dfr.loc[noteidx,'index'] = dfr.loc[xnote[-1],'index']
                dfr.loc[noteidx,'note'] = dfr.loc[xnote[-1],'note']
                try:    
                    dfr.loc[noteidx,'expected_index'] = dfj.loc[~dfj.expected_index.isna(),'expected_index'].iloc[0]
                    dfr.loc[noteidx,'expected_note'] = dfj.loc[~dfj.expected_note.isna(),'expected_note'].iloc[0]
                    dfr.drop(xnote[:-1],inplace=True)
                except IndexError:
                    dfr.drop(xnote,inplace=True)
        if np.isnan(row.expected_note):
            xnote=[ii]
        else:
            xnote = []
        coalesce=False
        prev_note = row.note
    return dfr.copy()

def get_ref_series(tsdict):
    
    f0_label = [l for l in tsdict if l.find('_f0')>-1][0]
    amp_label = f0_label.replace('_f0','_ampl')

    fts = tsdict[f0_label]
    ats = tsdict[amp_label]
    return fts, ats
    
def segment_with_clips(filename, clip_tier, meljs=None, output=None):
    """Perform note segmentation and align with a score

    Args:
        filename (str): A signals pickle file
        clips (list of (t_start, t_end, melody)): clip-level segmentation with melody as labels
        meljs: json with melodies
        output (str, optional): Output file (Defaults to outputing to stdout)
    Returns:
        note dataframe
    """
    
    tsdict = read_timeseries(filename, labels=None)
    fts, ats = get_ref_series(tsdict)
        
    df = segment_notes_in_timeseries(fts,ats)
    matchdfs = []

    for ii, interval in enumerate(clip_tier.intervals):
        t_clip = interval.start_time
        t_clip_end = interval.end_time
        melnames = list(meljs.keys())
        label = interval.text + f' ({ii})'
        tune = melnames[[label.find(x)>-1 for x in melnames].index(True)]
        
        dfclip = df[(df.start>t_clip)&(df.end<t_clip_end)]
        meldf = mel_js_to_df(meljs, tune)
        score_pitch = meldf.round_pitch.values
        played_pitch = dfclip.round_pitch.values
        dfmatch = match_melodies(score_pitch, played_pitch)
        dfmatch = dfmatch.join(meldf,on='expected_index',rsuffix='_exp')
        dfmatch['found_index_in_clip'] = dfmatch.found_index
        idx = ~dfmatch.found_index_in_clip.isna()
        dfmatch.loc[idx,'found_index'] = dfclip.iloc[dfmatch[idx].found_index].index
        dfmatch['clip'] = label
        matchdfs.append(dfmatch)
        
    matchdf = pd.concat(matchdfs,ignore_index=True)
    dfm = matchdf.join(df.reset_index(),on='found_index',how='left',lsuffix='_exp')
    
    # discard unmatched notes
    #dff = dfm[~dfm.expected_index.isna()]
    dfr = reassign_same_notes(dfm)
    dfr = dfr[(~dfr.expected_index.isna())&(~dfr.found_index.isna())]
    idx = ~dfr.expected_note.isna()
    dfr.loc[idx,'exp_note_name'] = dfr.loc[idx,'expected_note'].apply(lambda x : midi_to_note(x+71))
    dfr['label'] = dfr.apply(lambda row: "{expected_index:.0f}({rep_no:.0f}) - {exp_note_name}".format(**row),axis=1)
    #dfr['clip'] = dfr.rep_no.apply(lambda x: "{:.0f}".format(x))
    
    tmax = fts.t[-1]
    
    return dfr, tmax

 
def segment_and_align(filename, meldf, tune=None, output=None):
    """Perform note segmentation and align with a score

    Args:
        filename (str): A signals pickle file
        melody_js (str):  melody js dict
        tune (str, optional): A melody to extract from the YAML file. 
            Defaults to file being the whole melody.
        output (str, optional): Output file (Defaults to outputing to stdout)
    Returns:
        note dataframe
    """
    
    tsdict = read_timeseries(filename, labels=None)
    fts, ats = get_ref_series(tsdict)
        
    df = segment_notes_in_timeseries(fts,ats)
    
    dfmatch = auto_match_melodies(meldf, df)
    dfm = (dfmatch.join(df.reset_index(),on='found_index',how='left') 
                 .join(meldf,on='expected_index',rsuffix='_exp'))

    # discard unmatched notes
    #dff = dfm[~dfm.expected_index.isna()]
    dfr = reassign_same_notes(dfm)
    dfr = dfr[(~dfr.expected_index.isna())&(~dfr.found_index.isna())]
    idx = ~dfr.expected_note.isna()
    dfr.loc[idx,'exp_note_name'] = dfr.loc[idx,'expected_note'].apply(lambda x : midi_to_note(x+71))
    dfr['label'] = dfr.apply(lambda row: "{expected_index:.0f}({rep_no:.0f}) - {exp_note_name}".format(**row),axis=1)
    dfr['clip'] = dfr.rep_no.apply(lambda x: "{:.0f}".format(x))
    
    tmax = fts.t[-1]
    
    return dfr, tmax


def process_csv(csvfile, melody_file, root_dir='.', output_dir=None, test=False):
    
    if output_dir is None:
        output_dir = root_dir
    
    df = pd.read_csv(csvfile, index_col=0)

    mel_js = read_mel_js(melody_file)

    for irow, row in df.iterrows():
        #print(row)
        basename = os.path.splitext(row.filename)[0]
        tsfile = root_dir + '/' + basename +'_ts.pickle'
        outfile = output_dir + '/' + basename + '_notes.TextGrid'
        logging.info("Processing "+tsfile)
        this_dir = os.path.split(outfile)[0]
        tune = row.tune
        if not test:
            Path(this_dir).mkdir(parents=True, exist_ok=True)
            try:
                meldf = mel_js_to_df(mel_js, tune)
            except KeyError:
                meldf = None
            if meldf is not None:
                dfa, tmax = segment_and_align(tsfile, meldf, output=outfile)
            else:
                dfa, tmax = segment_notes(tsfile)
            
            output_segmentation(dfa, output=outfile, tmax=tmax)
        else:
            if os.path.isfile(tsfile):
                logging.debug(f"{tsfile} not found")
            logging.debug(f"Will output to {this_dir}")


def parse_args():
    # Same main parser as usual
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose', action='store_true')

    sub_parsers = parser.add_subparsers(dest='command', help='commands', required=True)    
    
    parser_sil= sub_parsers.add_parser('silence', help='Estimate silence and voice thresholds')
    parser_sil.add_argument('file', help='timeseries file')

    parser_notes = sub_parsers.add_parser('notes', help='Segment notes')
    parser_notes.add_argument('file', help='timeseries file')
    parser_notes.add_argument('-m','--melody-file',  help='YAML file containing melodies (leave out for note segmentation without alignment)')
    parser_notes.add_argument('-t','--tune',  help='melody name')
    parser_notes.add_argument('-o','--output',  help='output file')
    
    parser_csv = sub_parsers.add_parser('csv', help='File list from csv')
    parser_csv.add_argument('file', help='csv file (contains file, melody columns)')
    parser_csv.add_argument('-m','--melody-file',  help='YAML file containing melodies (leave out for note segmentation without alignment)')
    parser_csv.add_argument('-r','--root-dir', help='root folder (defaults to .)',
                            default='.')
    parser_csv.add_argument('-o','--output',  help='output dir')

    parser_csv = sub_parsers.add_parser('clips', help='Read textgrid with melody names in tier')
    parser_csv.add_argument('file', help='timeseries file')
    parser_csv.add_argument('-m','--melody-file',  help='YAML file containing melodies (leave out for note segmentation without alignment)')
    parser_csv.add_argument('-r','--root-dir', help='root folder (defaults to .)',
                            default='.')
    parser_csv.add_argument('-t','--textgrid-file', help='textgrid with excerpt segmentation',
                            default='.')
    parser_csv.add_argument('-n','--tier-name', help='tier with melody names',
                            default='clip')
    parser_csv.add_argument('-o','--output',  help='output file')
    return parser.parse_args()
    

def get_tier(textgrid_file, tier_name):
    tg = tgt.read_textgrid(textgrid_file)
    return tg.get_tier_by_name(tier_name)


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    if args.command == 'silence':
        output_silence_db(args.file)
    elif args.command == 'notes':
        if args.melody_file:
            melody = read_score(args.melody_file, args.tune)
            df, tmax = segment_and_align(args.file, melody)
        else:
            df, tmax = segment_notes(args.file)
        output_segmentation(df, tmax=tmax, output=args.output)
    elif args.command == 'clips':
        melodies = read_mel_js(args.melody_file)
        tier = get_tier(args.textgrid_file, args.tier_name)
        df, tmax = segment_with_clips(args.file, tier, melodies)
        output_segmentation(df, tmax=tmax, output=args.output)
    elif args.command == 'csv':
        process_csv(args.file, args.melody_file, args.root_dir)
    else:
        logging.error('Unknown command: run "note_matcher -h"')


if __name__ == '__main__':
    main()