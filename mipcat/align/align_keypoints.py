#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
    Find the offset between two sound files
    by analysisng key points (peaks) in spectrogram
    
    The aim of this script is to compare long audio files
    and find the right offset betwen the two files. 
    
    It can be used for example to align audio between one camera
    and a reference microphone
"""
import os
import argparse
import numpy as np
import librosa as lr
import scipy.signal as sig
from scipy.ndimage import maximum_filter
from scipy.ndimage import binary_erosion
from sklearn.neighbors import KDTree
from collections import Counter

DEBUG = False
EPS = 1e-12


def wav_load(filename, sr=None, channel=0, res_type='kaiser_fast'):
    w, sr = lr.load(filename, mono=False, sr=sr, res_type=res_type)
    if len(w.shape)>1:
        w = w[channel,:]
    return w, sr


def detect_peaks(image, thresh=None, xrad=2, yrad=1):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    neighborhood = np.ones((xrad,yrad))

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    if thresh is not None:
        detected_peaks = local_max & (image>thresh)

    return detected_peaks


def get_markers(w,sr,twind=0.0464,pkpct=98,frad=60.0,trad=0.3):
    nfft = 2**(np.round(np.log2(sr*twind)))
    nhop = nfft//2
    if DEBUG:
        print(f'nhop = {nhop}')
    
    fs, ts, ss = sig.spectrogram(w,nfft=nfft,fs=sr,
                                 window=np.hanning(nfft),noverlap=nfft-nhop)
    ss[ss<EPS] = EPS
    dbss = 10*np.log10(ss)
    #dbmax = np.max(dbss)
    
    nf = int(np.round(frad/(sr/nfft)))
    nt = int(np.round(trad/(ts[1]-ts[0])))
    
    spks=detect_peaks(dbss, xrad=nf, yrad=nt, thresh=np.percentile(dbss,pkpct))
    
    npks = np.sum(spks)
    feat = np.zeros((npks,3))
    
    for ii,(xx,yy) in enumerate(zip(*np.where(spks))):
        feat[ii,:] = np.array([yy,xx,dbss[xx,yy]])
        
    return feat, nhop


def marker_differences(markers, max_dist=50, min_dist=2, include_ampl=False):
    feat = []
    for marker in markers:
        dist = markers[:,0]-marker[0]
        near_markers = np.flatnonzero((dist<=max_dist)&(dist>=min_dist))
        for rel_idx in near_markers:
            if include_ampl:
                feat.append([marker[0], markers[rel_idx,0] - marker[0], 
                             marker[1], markers[rel_idx,1], 
                             markers[rel_idx,2] - marker[2]])
            else:
                feat.append([marker[0], markers[rel_idx,0] - marker[0], marker[1], markers[rel_idx,1]])
    return np.array(feat,dtype='int')


def get_features(wlist, sr, min_dist_sec=0.0, max_dist_sec=1.0):
    """
    Generates features for a list of waveforms.
    Features are pairs of spectral peaks

    Features are tables with the following columns:
    * Frame index
    * Frame difference
    * Origin frequency
    * Destination frequency
    """
    feats = [] 
    for ii, w in enumerate(wlist):
        feat, nhop = get_markers(w, sr=sr)
        if DEBUG:
            print(f"file {ii}: {len(feat)} peaks")

        max_dist = int(max_dist_sec*sr/nhop)
        min_dist = int(min_dist_sec*sr/nhop)

        fd = marker_differences(feat, min_dist=min_dist, max_dist=max_dist, 
                                include_ampl=False)
        if DEBUG:
            print(f"file {ii}: {len(fd)} features")

        feats.append(fd)

    return feats, nhop


def compare_features(feat1, feat2, dist_min=2, stat_rad=2):
    """
    Given a list of features, return the frame delta
    that correpsonding to the most matching features

    Args:
        feat1 (numpy array): first feature set
        feat2 (numpy array): reference feature set
        dist_min (int, optional): Maximum distance between features. Defaults to 2.
        stat_rad (int, optional): Radius for statistics of matches. Defaults to 2.

        features are numpy arrays contining N_points x N_features
        Features should contain the time index in first column and other features in other columns
    Returns:
        delay: estimated delay 
        frac: fraction of matches within delay
    """    
    kd = KDTree(feat1[:,1:4])
    dist,ind = kd.query(feat2[:,1:4])

    x = feat1[ind[dist[:,0]<=dist_min,0],0]
    y = feat2[dist[:,0]<=dist_min,0]
    ctr = Counter(y-x)
    if DEBUG:
        for ii, (delay, n) in enumerate(ctr.most_common()):
            print(f'{delay:4d}: n={n:6d}')
            if ii > 10:
                break
    dtr = ctr.most_common()[0][0]
    nvec = (y-x)[np.abs(y-x-dtr) < stat_rad] 
    dti = np.mean(nvec)

    return dti, len(nvec)/len(x)


def wave_delta(w1, w2, sr, delta_t_max=1.0):
    feat, nhop = get_features([w1,w2], sr=sr, min_dist_sec=.05, max_dist_sec=delta_t_max)
    dfr, frac = compare_features(feat[0], feat[1])
    if DEBUG:
        print('resolution:', nhop/sr)
    return dfr*nhop/sr, frac


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="delayed audio filename")
    parser.add_argument("file2", help="reference audio filename")
    
    parser.add_argument("-d", "--chunk-duration", type=float, default=10,
                        help="Duration of chunks in delayed file")
    parser.add_argument("-n", "--n-chunks", type=int, default=3,
                        help="Number of chunks in delayed file to use for comparison")
    parser.add_argument("-c", "--channel", type=int, default=0,
                        help="Channel number for reference signal")
    parser.add_argument("-t", "--table", action="store_true", 
                        help="Output comma separated file names, delay and number of matches to add to a CSV file")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output extra information")
    return parser.parse_args()


def compare_files(file1, file2, sr=22050, channel1=0, channel2=0, delta_t_max=1.0):
    """
    Find delay between file1 and file2, 
    using channels channel1 and channel2.

    returns dt: the position of file1 in file2
    """

    wref, sr = wav_load(file2, sr=sr, channel=channel2)
    if DEBUG:
        print(f'Read {file1}')
    wdest, sr = wav_load(file1, sr=sr, channel=channel1)
    if DEBUG:
        print(f'Read {file2}')
    return wave_delta(wdest, wref, sr=sr, delta_t_max=delta_t_max)


def compare_files_multi_chunk():

    wref, sr = wav_load(args.file1, sr=sr, channel=args.channel)
    ref_duration = len(wref)/sr
    dest_duration = lr.get_duration(filename=args.file2)
    
    if DEBUG:
        print(f'reference :{ref_duration:.3f} sec')
        print(f'destination :{dest_duration:.3f} sec')

    min_dist = ref_duration/args.n_chunks/2
    ti = interesting_points(wref, sr=sr, min_dist=2.0, n_peaks=args.n_chunks, 
                          tvar=args.chunk_interval)
    ti -= args.chunk_interval/2

    delay, val, sr, stats = compare_dest_to_ref(args.file2, wref, 
                                     chunk_duration_sec=args.chunk_interval,
                                     chunk_times=ti, sr=sr)


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        DEBUG=args.verbose

    dt, frac = compare_files(args.file1, args.file2, channel1=0, 
                       channel2=args.channel, delta_t_max=5.0)
    if args.table:
        f2_dir, f2_name = os.path.split(args.file2)
        rel_file1 = os.path.relpath(args.file1, f2_dir)
        print(f"{args.file2}, {rel_file1}, {args.channel}, {dt}, {frac}")
    else:
        print(f'{dt} sec, {frac*100:.2f} % matches')

