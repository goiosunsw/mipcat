"""
Perform the segmentations of multiple files into independent musical excerpts and output them as text grid files
"""

import sys
import os
import argparse
import re
import numpy as np
import yaml
import tgt
import logging
import difflib
import scipy.signal as sig
from scipy.io import wavfile
import librosa as lr
from pypevoc.SoundUtils import RMSWind


from timeseries import SampledTimeSeries

logging.basicConfig(level=logging.INFO)


def nextpow2(x):
    """
    Smallest 2**n above x
    
    >>> nextpow2(1024)
    1024
    
    >>> nextpow2(1025)
    2048
    """
    return int(2**(np.ceil(np.log2(x))))

# def tolerant_union_with_precedence(x,y,tolerance=1):
#     """
#     Returns the union of x and y within a tolerance, i.e if an element of 
#     x is closer than tolerance to an element of y only the element of x 
#     is present
    
#     >>> tolerant_union_with_precedence([0.0,1.0,2.0,3.0],[0.0,1.1,5.0,6.0])
#     array([0., 1., 2., 3., 5., 6.])
    
#     >>> tolerant_union_with_precedence([0.0,0.5,1.0,2.0,4.0],[-1.2,0.7,1.1,3.0,4.1])
#     array([-1.2, 0., 0.5, 1., 2., 3., 4.])
#     """
#     x = np.asarray(x)
#     y = np.asarray(y)
    
#     try:
#         ylasti=(y-x[0]<tolerance).nonzero()[0][0]
#         z=y[:ylasti].tolist()
#     except IndexError:
#         ylasti=0

#     for xx in x:
#         try:
#             #yi = np.argmin(np.abs(xx-y[ylasti:]))+ylasti
#             yi = ((y-xx)<tolerance).nonzero()[0][0]
#         except IndexError:
#             break

#         mindist = y[yi]-xx
#         #print(y[ylasti:yi],xx,)
#         newy = y[ylasti:yi]
#         z.extend(newy[newy-xx<-tolerance])
#         z.append(xx)
#         try:
#             ylasti = ((y-xx)>tolerance).nonzero()[0][0]
#         except IndexError:
#             ylasti = len(y)-1
#     z.extend(y[ylasti:])

#     return np.asarray(z)
        
        
def tolerant_union_with_precedence(x, y, tolerance=1):
    """
    Returns the union of x and y within a tolerance, i.e if an element of 
    x is closer than tolerance to an element of y only the element of x 
    is present
    
    ### FIXME
    This still needs some fixing for the cases where an element is 
       yi == xj+-tolerance
    
    >>> tolerant_union_with_precedence([0.0,1.0,2.0,3.0],[0.0,1.1,5.0,6.0])
    array([0., 1., 2., 3., 5., 6.])
    
    >>> tolerant_union_with_precedence([0.0,0.5,1.0,2.0,4.0],
                                       [-1.2,0.7,1.1,3.0,4.1])
    array([-1.2, 0., 0.5, 1., 2., 3., 4.])
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = []
    lasty = y[0]-2*tolerance
    
    for xx in x:
        # add all elements of y from lasty to x-tolerance
        z.extend(y[(y >= lasty) & (y < xx-tolerance)])
        # add current x
        z.append(xx)
        # set lasty to xx 
        lasty = xx+tolerance
        
    # add remaining elements of y
    z.extend(y[(y > lasty)])

    return np.asarray(z)


def read_runsheet(filename):
    yaml_dir = os.path.split(os.path.abspath(filename))[0]
    with open(filename, 'r') as f:
        yml = yaml.full_load(f)
    channel_lists = yml['channel_desc']
    wave_files = yml['wave_files']
    try:
        root_dir = yml['root_dir']
    except KeyError:
        root_dir = '.'
        
    try:
        melodies_file = os.path.join(yaml_dir, yml['melodies_file'])
    except KeyError:
        logging.warning('Meoldies file not defined in YAML')
        return channel_lists, wave_files, root_dir
    
    try:
        with open(melodies_file, 'r') as f:
            melodies = yaml.full_load(f)
    except FileNotFoundError:
        logging.warning(f'Could not find meoldies file {melodies_file}')
        return channel_lists, wave_files, root_dir
    
    for wd in wave_files:
        try:
            wd['melody'] = melodies[wd['tune']]
        except KeyError:
            logging.warning(f'Meoldy not defined for  {wd["tune"]}')
    
    return channel_lists, wave_files, root_dir


def get_eg_regions(tg_filename, tiername='clip'):
    """
    Get a region times from a textgrid file
    """
    tg = tgt.read_textgrid(tg_filename)
    tier = tg.get_tier_by_name(tiername)
    tu = [x.start_time for x in tier]
    td = [x.end_time for x in tier]
    sch = [x.text for x in tier]
    return tu, td, sch


def get_wave(filename, text=None, index=None, text_index=0, 
             tg_suffix='_eg.TextGrid', tiername='clip', channel=None):
    """
    Return a wavefile corresponding to a regio in a filename
    
    Either provide the index number of the region or a regex (as name)
    to search in the region list. If several are found, text_index 
    provides the index number of one of these regions 
    """

    if text is None and index is None:
        raise TypeError('Either text or index must e provided')
    
    ts, te, sch = get_eg_regions(os.path.join(root_dir, os.path.splitext(filename)[0]+tg_suffix))
        
    if text:
        index = [ii for ii, x in enumerate(sch) if re.match(text, x)][text_index]
    
    tstart = ts[index]
    tend = te[index]
    print(tstart, tend, sch[index])
    sr, wall = wavfile.read(filename)
    if channel is None:
        return sr, wall[int(tstart*sr):int(tend*sr), :]
    else:
        return sr, wall[int(tstart*sr):int(tend*sr), channel]


def segment_file(w, sr, channels=['barrel', 'mouthpiece', 'mouth'], channel_desc=None, 
                 time_step=2, high_percentile=90, silence_factor_thresh=0.1):
    """
    Segment a file finding regions of silence
    
    channel desc gives the description of channels in the wave file
    channel is the channel to use for segmentation (list for priority)
    time step is the resolution of the amplitude envelopes
    """
    
    for channel in channels:
        sch = [x['name'] for x in channel_desc].index(channel)
        ww = w[:, sch] / np.max(np.abs(w[:, sch]))
        logging.info(f'Segmentation: trying channel {sch} ({channel})')

        frame_length = nextpow2(time_step*sr)
        hop_length = frame_length//2

        cme = lr.feature.rms(y=ww, frame_length=frame_length, 
                             hop_length=hop_length, center=True, 
                             pad_mode='reflect')[0]
        ts = SampledTimeSeries(cme, t=lr.frames_to_time(np.arange(len(cme)),
                                                        sr=sr, hop_length=hop_length))

        tu, td = ts.crossing_times(ts.percentile(high_percentile)*silence_factor_thresh)
        if len(tu) > 0 and len(td) > 0:
            break
            
    if tu[0] > td[0]:
        tu = np.concatenate([[0], tu])

    if tu[-1] > td[-1]:
        td = np.concatenate([td, [len(ww)/sr]])
    
    return tu, td, sch


class NoteSegmenter(object):
    def __init__(self, sr, hop_duration_sec=0.01,  min_note_dur_sec=.05,
                 min_phrase_dur_sec=1.):
        self.sr = sr
        self.hop_length = nextpow2(sr*hop_duration_sec)
        self.min_note_dur_sec = min_note_dur_sec
        self.min_phrase_dur_sec = min_phrase_dur_sec
        self.transpose_semitones = 2

    @property
    def time(self):
        return lr.frames_to_time(self.hop_length, np.arange(self.chroma))
        
    def _calc_chroma_and_energy(self, w):
        hop_dur = self.hop_length/self.sr
        self.hop_dur = hop_dur
        
        cmx = lr.feature.chroma_cqt(w, sr=self.sr, hop_length=self.hop_length)
        cme = lr.feature.rms(y=w, frame_length=2048, hop_length=self.hop_length, 
                             center=True, pad_mode='reflect')[0]

        self.chroma = cmx
        self.energy = cme
        
    def get_chroma_segments(self, w):
        """
        Detect pitch transitions using the chorma derivative
        
        hop_length is the distance between points
        min_note_dur_sec is the minimum length of a note 
        (pitch transitions within this duration will not be detected)
        """
        try:
            self.chroma
        except AttributeError:
            self._calc_chroma_and_energy(w)
    
        # Chroma flow
        rad = int(np.round(self.min_note_dur_sec/self.hop_dur/2))
        df = np.sum(np.abs(np.diff(self.chroma)), axis=0)
        dfpk, feat = sig.find_peaks(df, distance=rad, prominence=.3)#height=.3)
        tdfpk = lr.frames_to_time(dfpk+.5, sr=self.sr, hop_length=self.hop_length)

        return tdfpk
    
    def get_chroma_energy_segments(self, w, energy_min_ratio=.1):
        try:
            self.chroma
        except AttributeError:
            self._calc_chroma_and_energy(w)
            
        # Amplitude minima
        rad = int(np.round(self.min_note_dur_sec/self.hop_dur/2))
        emins, feat = sig.find_peaks(-self.energy, distance=rad,
                                     prominence=np.max(self.energy)*energy_min_ratio)
        temins = lr.frames_to_time(emins, sr=self.sr, hop_length=self.hop_length)
        eminvals = self.energy[emins]

        # Chroma flow
        df = np.sum(np.abs(np.diff(self.chroma)), axis=0)
        dfpk, feat = sig.find_peaks(df, distance=rad, prominence=.3)#height=.3)
        tdfpk = lr.frames_to_time(dfpk+.5, sr=self.sr, hop_length=self.hop_length)

        return tolerant_union_with_precedence(temins, tdfpk, 
                                              tolerance=self.min_note_dur_sec)
 
    def get_phrasing_segments(self, w):
        cme = self.energy
        ethr = np.percentile(cme/100, 99)
        onsets = lr.frames_to_time(np.flatnonzero((cme[:-1] <= ethr) &
                                                  (cme[1:] > ethr)), 
                                   sr=self.sr, hop_length=self.hop_length)
        offsets = lr.frames_to_time(np.flatnonzero((cme[:-1] > ethr) & 
                                                   (cme[1:] <= ethr)),
                                    sr=self.sr, hop_length=self.hop_length)

        phlims = []
        phrase_dur_min = self.min_phrase_dur_sec

        tmax = len(w)/self.sr

        for ons in onsets:
            try:
                ofs = offsets[[x >= ons for x in offsets].index(True)]
                phlims.append((ons, ofs))
            except ValueError:
                phlims.append((ons, tmax))

        sillims = []
        last_sil = 0

        for phl in phlims:
            sillims.append((last_sil, phl[0]))
            last_sil = phl[1]
        if last_sil < tmax:
            sillims.append((last_sil, tmax))
                
        t = np.arange(len(w))/self.sr
        return phlims, sillims

    def segment_notes(self, w, mode='combined', energy_min_ratio=0.1):
        if mode == 'chroma':
            csegs = self.get_chroma_segments(w)
        elif mode == 'combined':
            csegs = self.get_chroma_energy_segments(w, energy_min_ratio=energy_min_ratio)
        else:
            logging.error(f'mode {mode} not available for note sgmentation')
            
        phlims, sillims = self.get_phrasing_segments(w)

        note_times = []
        rad = int(np.round(self.min_note_dur_sec/self.hop_dur))

        for phl in phlims:
            ntrs = csegs[(csegs >= phl[0]+self.min_note_dur_sec) & 
                         (csegs <= phl[1]-self.min_note_dur_sec)]
            last_note = phl[0]
            for nt in ntrs:
                note_times.append((last_note, nt))
                last_note = nt
            note_times.append((last_note, phl[1]))
        self.note_times = note_times
        return note_times

    def get_pitch_track(self, w, fmin=100, fmax=1500):
        #pitches, magnitudes = lr.core.piptrack(w,sr=self.sr,threshold=.2,hop_length=self.hop_length,
        #                                       fmax=fmax,fmin=fmin)
        #self.ptrack = (np.max(pitches,axis=0))
        
        f0 = lr.yin(w, sr=self.sr, fmin=fmin, fmax=fmax, hop_length=self.hop_length)
        self.ptrack = f0

        self.chromas = (np.argmax(self.chroma, axis=0))
        return self.ptrack, self.chromas

    def get_times(self):
        return lr.frames_to_time(np.arange(len(self.ptrack)), sr=self.sr,
                                 hop_length=self.hop_length)

    def get_note_frequencies(self, note_times):
        tchr = lr.frames_to_time(np.arange(len(self.chromas)), sr=self.sr,
                                 hop_length=self.hop_length)
        tp = lr.frames_to_time(np.arange(len(self.ptrack)), sr=self.sr,
                               hop_length=self.hop_length)

        midis = []
        freqs = []

        for ts, te in note_times:
            ch = self.chromas[(tchr >= ts) & (tchr <= te)]
            p = self.ptrack[(tp >= ts) & (tp <= te)]
            midi = lr.hz_to_midi(np.median(p*2**(2/12)))
            freqs.append(np.median(p*2**(2/12)))
            try:
                midis.append(int(round(midi)))
            except OverflowError:
                midis.append(0)
            print(f'{ts:7.3f} - {te:7.3f} :: {min(ch):2d} - {max(ch):2d} ({np.median(ch):2.0f}) :: {np.mean(p):6.1f} +- {np.std(p):6.2f} :: {midi:3.0f}')

        self.freqs = freqs
        return freqs, midis

    def pitch_adjust(self, score, max_semitone_adj=2, semitone_step=.1):
        """
        Find the optimal pitch adjustment to match score to recording

        return the number of semitones to add to recorded
        """
        maxv = 0
        maxa = None
        
        for stadj in np.arange(-max_semitone_adj, max_semitone_adj+semitone_step,
                               semitone_step):
            midis = np.round(lr.hz_to_midi(np.array(self.freqs)*2**(stadj/12))).astype('i')
            sm = difflib.SequenceMatcher(None, midis, score)
            rat = sm.ratio()
            if rat > maxv:
                maxv = rat
                maxa = stadj
        return maxa 

    # def score_align(self, score):
    #     maxa = self.pitch_adjust(score)
    #     s2 = score.copy()
    #     s1 = np.round(lr.hz_to_midi(np.array(self.freqs)*2**(maxa/12))).astype('i').tolist()


    #     sm = difflib.SequenceMatcher(None,s1,s2)
    #     print(sm.ratio())

    #     melody_notes = [[] for xx in s1]
    #     aligned_note_times = []

    #     for tag, i1, i2, j1, j2 in (sm.get_opcodes()):
    #         if tag == 'insert':
    #             #melody_notes[i1].extend(range(j1,j2))
    #             print(f"notes to be added between {i1-1} and {i1}: {j1}:{j2} ({s2[j1:j2]})")
    #             split_note = None
    #             if i1>1:
    #                 if s2[j1]==s1[i1-1]:
    #                     print(f"[PREV] note at {j1} ({s2[j1]}) is the same as {i1-1} ({s1[i1-1]})")
    #                     split_note = i1-1
    #             if i1<len(s1)-1:
    #                 if s2[j2-1]==s1[i1]:
    #                     print(f"[NEXT] note at {j2-1} ({s2[j2-1]}) is the same as {i1} ({s1[i1]})")
    #                     split_note = i1
                        
    #             # find an amplitude minimum at note 'split_note'
    #             split_times = np.flatnonzero((self.energy_minima_times > self.note_times[split_note][0] +self.min_note_dur_sec)&
    #                                          (self.energy_minima_times < self.note_times[split_note][1]-self.min_note_dur_sec))       
    #             split_vals = self.energy_minima[split_times]
    #             split_order = np.argsort(split_vals)
    #             print(f'Number of missing notes: {j2-j1}')
    #             split_times = self.energy_minima_times[split_times[split_order[:j2-j1]]]
    #             print (f"Split at times {split_times}")
    #             split_indices = np.arange(j1,j2+1)#[split_order[:j2-j1]]
                
    #             for tt,jj in sorted(zip(split_times,split_indices)):
    #                 aligned_note_times.append((tt,s2[jj]))
            
    #         elif tag == 'delete':
    #             for ii in (range(i1,i2)):
    #                 print(f"{ii} ({s1[ii]}) : -- ")
            
    #         else:
    #             for ii,jj in zip(range(i1,i2),range(j1,j2)):
    #                 melody_notes[ii].append(jj)
    #                 tt = self.note_times[ii]
    #                 aligned_note_times.append((tt[0],s1[ii]))
    #                 #aligned_note_times.append(self.note_times[jj])
    #                 print(f"{ii} ({s1[ii]}) : {jj} ({[s2[ji] for ji in melody_notes[ii]]}) ")
    #     return aligned_note_times

    def score_align(self, score):
        """
        Aligns the performance notes (stored in the object) to the score
        Requires: 
            NoteSegmenter.segment_notes()
            NoteSegmenter.get_note_frequencies()
        """
        maxa = self.pitch_adjust(score)
        s2 = score.copy()
        s1 = np.round(lr.hz_to_midi(np.array(self.freqs)*2**(maxa/12))).astype('i').tolist()

        sm = difflib.SequenceMatcher(None, s1, s2)
        print(f'Similarity ratio between performance and score: {sm.ratio()}')
        print(f'Optimal pitch adjustmen: {maxa} semitones')

        melody_notes = [[] for xx in s1]
        aligned_note_times = []

        for tag, i1, i2, j1, j2 in (sm.get_opcodes()):
            if tag == 'insert':
                #melody_notes[i1].extend(range(j1,j2))
                print(f"notes to be added between {i1-1} and {i1}: {j1}:{j2} ({s2[j1:j2]})")
                          
            elif tag == 'delete':
                for ii in (range(i1, i2)):
                    print(f"{ii} ({s1[ii]}) : -- ")
            
            else:
                for ii, jj in zip(range(i1, i2), range(j1, j2)):
                    melody_notes[ii].append(jj)
                    tt = self.note_times[ii]
                    aligned_note_times.append(tt)
                    #aligned_note_times.append(self.note_times[jj])
                    print(f"{ii} ({s1[ii]}) : {jj} ({[s2[ji] for ji in melody_notes[ii]]}) ")
                    
        return aligned_note_times
    
    def segment_and_align(self, ww, score):
        nsegs = self.segment_notes(ww)
        self.get_pitch_track(ww)
        freqs, midis = self.get_note_frequencies(nsegs)
        score_midi = [lr.note_to_midi(x['pitch']) + self.transpose_semitones for x in score]
        na_times = self.score_align(score_midi)
        
        aligned_notes = []
        for ii, (ts, te) in enumerate(na_times):
            aligned_notes.append((ts, te, f"{ii} - {score[ii]['pitch']}"))
        
        return aligned_notes


def generate_textgrid(intervals, tmax=None):
    """
    generates a textgrid with intervals starting in tu and ending in td
    where tu and td are taken from the list intervals of the form
    [(name1,[tu1,td1]),(name2,[tu2,td2]),...]
    """
    #print (intervals)
    if tmax is None:
        tmax = 0
        for label, intv in intervals:
            if len(intv) > 0:
                tmax = max(tmax, max(intv[1])) 


    tg = tgt.TextGrid()
    for name, intv in intervals:
        inttier = tgt.IntervalTier(start_time=0, end_time=tmax, name=name)
        tg.add_tier(inttier)

        for ii, it in enumerate(zip(*intv)):
            print(it)
            u = it[0]
            d = it[1]
            try:
                text = it[2]
            except IndexError:
                text = f'{name} {ii}'
            inttier.add_interval(tgt.Interval(u, d, text))

    return tg

         
def segment_regions_from_yaml(runsheet, outputdir='.', time_step=2.0, root_dir=None):
    channel_lists, wavfiles, yaml_root_dir = read_runsheet(runsheet)
    if root_dir is not None:
        root_dir = os.path.join(root_dir, yaml_root_dir)
    
    if not os.path.isdir(outputdir):
        raise FileNotFoundError 

    print (f"Outputing to {outputdir}")
    for wdesc in wavfiles:
        try:
            chdesc_lbl = wdesc['channels']
        except KeyError: 
            chdesc_lbl = 'default'

        channel_desc = channel_lists[chdesc_lbl]

        filename = wdesc['filename']
        filepath = os.path.join(root_dir, filename)
        logging.info(f'Processing {filepath}')
        
        sr, w = wavfile.read(filepath)
        rch = [x['name'] for x in channel_desc].index('external')
        wr = w[:, rch] / np.max(np.abs(w[:, rch]))

        tu, td, sch = segment_file(w, sr, channel_desc=channel_desc, 
                                   time_step=time_step)
                                 
        tg = generate_textgrid([('clip', (tu, td))])
        
        basefilename = os.path.splitext(os.path.split(filepath)[1])[0]
        baseoutputpath = os.path.join(outputdir, basefilename)
        
        #print(file)
        # for ii, (tstart, tend) in enumerate(zip(tu,td)):
        #     label = f'clip {ii}'
        #     print(f"{tstart:7.3f}, {tend:7.3f}, {label}")
        outname = baseoutputpath+'_ext.wav'
        wavfile.write(outname, sr, (wr/max(abs(wr))*2**15).astype('int16'))
        logging.info(f"Writing ext mic file to {outname}") 

        outname = baseoutputpath+'.TextGrid'
        tgt.write_to_file(tg, outname)
        logging.info(f"Writing ext mic file to {outname}") 


def process_yaml(runsheet, outputdir=None, full=True, time_step=2.0,
                 root_dir=None):
    channel_lists, wavfiles, yaml_root_dir = read_runsheet(runsheet)
    if full:
        logging.info('Regenerating textgirds')

    if root_dir is not None:
        root_dir = os.path.join(root_dir, yaml_root_dir)
    
    if outputdir is None:
        outputdir = root_dir
   
    logging.info('Outputing to '+outputdir)
    if not os.path.isdir(outputdir):
        raise FileNotFoundError 

    for wdesc in wavfiles:
        try:
            chdesc_lbl = wdesc['channels']
        except KeyError: 
            chdesc_lbl = 'default'

        channel_desc = channel_lists[chdesc_lbl]

        filename = wdesc['filename']
        filepath = os.path.join(root_dir, filename)
        logging.info(f'Processing {filepath}')
        
        sr, w = wavfile.read(filepath)
        rch = [x['name'] for x in channel_desc].index('external')
        wr = w[:, rch] / np.max(np.abs(w[:, rch]))
        wr -= np.mean(wr)

        # segment on which channel?
        try:
            seg_channels = [wdesc['segment_on']]
        except KeyError:
            seg_channels = ['barrel', 'mouthpiece', 'external', 'mouth']

        try:
            textgrid_file = wdesc['excerpt_textgrid']
        except KeyError:
            textgrid_file = None

        if full or textgrid_file is None:
            tu, td, sch = segment_file(w, sr, channels=seg_channels,
                                       channel_desc=channel_desc, time_step=time_step)
            labels = [f'excerpt {ii}' for ii, tt in enumerate(tu)]
        else:
            tg_filename = wdesc['excerpt_textgrid']
            tu, td, labels = get_eg_regions(os.path.join(root_dir,  tg_filename))
            sch = [x['name'] for x in channel_desc].index('barrel')
            
        allnsegs = []
        for tst, tend in zip(tu, td):
            ist = int(tst*sr)
            iend = int(tend*sr)
            ww = w[ist:iend, sch].astype('float')
            if len(ww) == 0:
                logging.warning(f'Empty region in {filename}: {tst}:{tend}')
                continue
            ww -= np.mean(ww)
            try:
                segs = segment_notes_in_region(ww, sr, wdesc['melody']['notes'])
            except TypeError:
                continue
            allnsegs.extend([(x[0]+tst, x[1]+tst, x[2]) for x in segs])
                                 
        tg = generate_textgrid([('clip', (tu, td, labels)), 
                                ('note', tuple([*zip(*allnsegs)]))])
        
        # basefilename = os.path.splitext(os.path.split(filepath)[1])[0]
        basefilename = os.path.splitext(filepath)[0]
        baseoutputpath = os.path.join(outputdir, basefilename)
        
        wavfile.write(baseoutputpath+'_ext.wav', sr, 
                      (wr/max(abs(wr))*2**15).astype('int16'))
        tgt.write_to_file(tg, baseoutputpath+'.TextGrid')
 

def process_region_from_file(filename, region, region_index=0, channel=0):
    try:
        region+1
        sr, ww = get_wave(filename, index=region, channel=channel)
    except ValueError:
        sr, ww = get_wave(filename, text=region, 
                          text_index=region_index, channel=channel)
    ww = ww/np.max(np.abs(ww))
    nsegs = NoteSegmenter(sr).segment_notes(ww)
    for ts, te in nsegs:
        print(f"{ts:.3f}, {te:.3f}")


def segment_notes_in_region(ww, sr, score):
    ww = ww/np.max(np.abs(ww))
    nsegs = NoteSegmenter(sr).segment_and_align(ww, score)
    return nsegs


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="WAV filename or YAML runsheet")
    parser.add_argument("-n", '--notes-only', action='store_true', 
                        help="Do not segment excerpts. Load information from existing textgrids")
    parser.add_argument("-r", '--regions-only', action='store_true', 
                        help="Do not segment notes")
    parser.add_argument("-R", '--region', default=0,
                        help="Region regex or filename (when WAV file provided)")
    parser.add_argument("-t", '--textgrid', 
                        help="Textgrid with regions (when WAV file provided)")
    parser.add_argument("-c", '--channel', type=int, default=0, 
                        help="Channel number in file")
    parser.add_argument("-s", '--min_silence_sec', type=float, default=2.0, 
                        help="Minimum silence duration in seconds")
    parser.add_argument("-o", "--output")
    parser.add_argument("-d", "--root_dir", help="root directory (bypass yaml)" )
    return parser.parse_args()

    
if __name__ == '__main__':
    args = parse_arguments()
    
    infile = args.infile
    
    root_dir = '.'
    outputdir = None
    
    if args.output:
        outputdir = os.path.realpath(args.output)
    else:
        outputdir = None
        
    try:
        runsheet = args.infile
        channel_lists, wavfiles, root_dir = read_runsheet(runsheet)
    except ValueError:
        runsheet = None
        process_region_from_file(infile, args.region, channel=args.channel)
        exit(0)
        
    if args.root_dir:
        root_dir = args.root_dir

    if args.regions_only:
        segment_regions_from_yaml(runsheet, time_step=args.min_silence_sec,
                                  root_dir=root_dir)
        exit(0)
    
    process_yaml(runsheet, 
                 full=not(args.notes_only), 
                 time_step=args.min_silence_sec, 
                 outputdir=outputdir,
                 root_dir=root_dir)
        
    
       
