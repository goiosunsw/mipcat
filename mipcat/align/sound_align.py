#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
    Find the offset between two sound files
    
    The aim of this script is to compare long audio files
    and find the right offset betwen the two files. 
    
    It can be used for example to align audio between one camera
    and a reference microphone
"""

import argparse
import random
import numpy as np
import librosa as lr
import scipy.signal as sig


DEBUG = False


def nextpow2(x):
    return int(2**(np.ceil(np.log2(x))))


def clip(x,xlo=None,xhi=None):
    y=x
    if xhi is not None:
        y[x>xhi]=xhi
    if xlo is not None:
        y[x<xlo]=xlo
    return y


def normalize(x, mode='std'):
    if mode == 'std':
        mx = np.mean(x)
        sx = np.std(x)
    elif mode == 'iqr':
        mx = np.median(x)
        sx = np.percentile(x,75)-np.percentile(x,25)
    elif mode == 'minmax':
        u = np.max(x)
        l = np.min(x)
        mx = l
        sx = (u-l)
    else:
        raise ValueError(f'Unsupported mode {mode}')
    return (x-mx)/sx


def function_in_window(x, fun, nwind=2):
    xw = np.zeros(x.shape) 
    xt = np.stack([x[n:len(x)-nwind+n] for n in range(nwind+1)])
    x1 = fun(xt,axis=0)
    npadl = nwind//2
    xw[npadl:-nwind+npadl] = x1

    return xw


def correlation(x,y):
    """
    FFT based cross-correlation (faster)

    Args:
        x (float array): first signal
        y (float array): another signal

    Returns:
        correlation (real values)
    """
    lx = len(x)
    ly = len(y)
    nmin = min(lx,ly)
    padlen = nextpow2(lx + ly)
    x_ = np.zeros(padlen)
    y_ = np.zeros(padlen)
    x_[:lx] = x
    y_[:ly] = y
    sxy = np.fft.fft(x_) * np.conj(np.fft.fft(y_))
    cxy = np.fft.ifft(sxy)
    return np.real(cxy)/nmin


def delay_from_corr(cxy):
    """
    Calculate delay from the peak of correlation

    Args:
        cxy (float array): correlation vector

    Returns:
        delay (int): delay in samples
        max (float): maximum of correlation at delay
    """
    mx = np.argmax(cxy)
    v = cxy[mx] 

    if mx > len(cxy)/2:
        return mx-len(cxy), v
    else:
        return mx, v    

def normal_load(sound_file, sr=None, min_noise=1e-8, channel=0):
    """Load a sound file and apply normalisations: 
    
    * convert to float from -1 to 1
    * add faint noise to avoid infinity in logs

    Args:
        sound_file (str): sound file name
        sr (float, optional): Sample rate. 
            Defaults to the sample rate of the file.
        min_noise (float, optional): Minimal noise to add t wave file, 
                                     avoiding infinite logs
    """
    # Load waveform
    w, sr = lr.load(sound_file, sr=sr, mono=False)
    print(sound_file, w.shape)
    if len(w.shape)>1:
        if DEBUG:
            print(f'Using channel {channel}')
        w = w[channel,:]
    w = w/np.max(np.abs(w))
    w = w + np.random.rand(*(w.shape))*min_noise
    return w, sr


def amplitude_peaks(sound_file, n_peaks=2, min_dist=1.0, sr=None, twind=0.2, thop=0.1):
    """Find amplitude peaks in a sound file, with a minimum 
    separation

    Args:
        sound_file (str): sound file name
        n_peaks (int, optional): Nuber of peaks. Defaults to 2.
        min_dist (float, optional): Minimum distance between peaks. Defaults to 1.0.
        sr (float, optional): Sampling rate to use for file read
        twind (float, optional): Window time
        thop (float, optional): hop between ampltiude estimations
    """

    nwind = nextpow2(twind*sr)
    nhop = int(thop*sr)
    nvar = int(tvar/thop)
    
    w, sr = lr.load(sound_file, sr=sr)
    
    wa = lr.feature.rms(w, frame_length=nwind, hop_length=nhop)
    wv = function_in_window(wa, np.std, nvar)
    wan = normalize(wa)
    ta = lr.core.frames_to_time(np.array(len(wa)),sr=sr,hop_length=nhop)

    pks = sig.find_peaks(wa)#,distance=min_dist)
    
    
def interesting_points(w, sr=1.0, n_peaks=2, min_dist=1.0, 
                       twind=0.2, thop=0.1, tvar=None, fmax=None,
                       fmin=50,min_noise=1e-8):
    """Find amplitude peaks in a sound file, with a minimum 
    separation

    Args:
        w (float array): wave vector
        sr (float, optional): Sampling rate to use for file read
                              Defaults to the rate of the file
        n_peaks (int, optional): Nuber of peaks. 
                                 Defaults to 2.
        min_dist (float, optional): Minimum distance between points. 
                                    Defaults to 1.0.
        twind (float, optional): Analysis window time 
                                 Defaults to 0.2 sec
        thop (float, optional): hop between ampltiude estimations
                                Defaults to 0.1 sec
        tvar (float, optional): varaibility window length (sec)
                                Defaults to min_dist
        fmax (float, optional): Frequency cutoff for variablily calcs. (Hz)
                                Defaults to sr/2
        fmin (float, optional): Minimum frequency (mainly determines fft size)
    """

    if tvar is None:
        tvar = min_dist
    

    # Calculate analysis parameters
    n_fft = nextpow2(sr/fmin)
    nwind = min(n_fft, nextpow2(twind*sr))
    nhop = int(thop*sr)
    nvar = int(tvar/thop)
    if fmax is None:
        fmax = sr/2

    # Spectrogram
    ws=20*np.log10(np.abs(lr.stft(w, hop_length=nhop , n_fft=n_fft,
                                  win_length=nwind)))
    # Amplitude
    wa = np.sum(ws,axis=0)
    # Time vector
    ta = lr.frames_to_time(np.arange(len(wa)), sr=sr, 
                           hop_length=nhop)
    
    # Index of max frequency
    fidx = int(np.ceil(fmax/sr*n_fft))
    fminidx = int(np.round(fmin/sr*n_fft))
    
    # remove extremely low values
    minv = np.percentile(ws,10)/2
    ws[ws<minv] = minv
    
    wsv = function_in_window(ws.T, np.nanstd, nvar).T
    wv = np.sum(np.abs(wsv[fminidx:fidx,:]),axis=0)

    pks,_ = sig.find_peaks(wv, distance=min_dist/thop)
    pks = pks[(ta[pks] > tvar) & (ta[pks] < max(ta) - tvar)]
    
    # select highest amplitude peaks
    pka = wa[pks]
    idx = np.flipud(np.argsort(pka))
    
    return np.sort(ta[pks[idx][:n_peaks]])

# def compare_chunks(w1, w2):
#     """
#     Performs a correlation between the two excerpts
#     and returns the minimum offset and value

#     Returns:
#         delay, maximum correlation
#     """
#     xc = correlation(w1, w2, mode='full')
#     nmin = min(len(w1), len(w2))

#     xb = np.ones(len(w1) + len(w2) - 1) * nmin
#     xb[:nmin] = np.arange(1, nmin+1)
#     xb[-nmin:] = np.arange(nmin, 0, -1)

#     #xcorr = xc/xb
#     maxi = np.argmax(xc)
#     maxv = np.max(xc)

#     delay = maxi - nmin + 1

#     return delay, maxv

def compare_chunks(w1, w2):
    """
    Performs a correlation between the two excerpts
    and returns the minimum offset and value

    Returns:
        delay, maximum correlation
    """

    cxy = correlation(normalize(w1), normalize(w2))
    return delay_from_corr(cxy)

def find_chunk_in_wave_per_chunks(w, chunk, dest_mul=2):
    """
    Finds the position of a chunk in a file, by
    reading sequentially chunks of the file and performing
    compare

    Args:
        w (numpy array): long audio vector 
                        in which to find reference
        chunk (numpy array): reference chunk to look 
                            for in dest
        dest_mul (float): destination block size relative 
                        to chunk
    Returns:
        delay: delay between chunk and file
        value: quality of the match

    Example:
        >>> wdest = np.random.randn(16384)
        
        >>> wchunk = wdest[3000:3000+1024] + np.random.randn(1024) * 0.1
        
        >>> find_chunk_in_wave_per_chunks(wdest,wchunk)[0]
        3000
      
    """

    nref = len(chunk)
    ndest = len(w)
    
    nw = int(nref*dest_mul)
    
    pos = 0 
    
    vals = []
    dels = []
    
    while pos < ndest - nw:
        cdest = w[pos:pos+nw]
        thisdel, thisval = compare_chunks(cdest,chunk)
        dels.append(thisdel)
        vals.append(thisval)
        pos += nref
        if DEBUG > 1:
            print(pos,thisdel,thisval)
        
    maxidx, maxval = max(((ii, vv) for ii, vv in enumerate(vals)), key = lambda x:x[1]) 
    stats = {'average_corr': np.mean(vals),}
    
    return dels[maxidx] + maxidx*nref, maxval, stats


def compare_dest_to_ref(dest, wref, chunk_size=1.0, 
                        chunk_times=None, sr=None, nchunks=3):
    """
    Compares two files and finds the best match location based on 
    two chunks from the destination file (usually the longest)
    
    If not sure which the longest file is, use helper function
    compare_files

    Args:
        dest (str): destination filename
        wref (float array): reference wave
        dest_chunk_st (float): starting time (sec) of the first chunk
                              (default:0.0 sec)
        dest_chunk_end (float): ending time (sec) of the last chunk
                                (default:end)
        chunk_length (float): length of the chunk (1.0 sec)
    """

    dest_duration = lr.get_duration(filename=dest)

    assert(dest_duration>chunk_size)
    
    if chunk_times is None:
        dest_duration = lr.get_duration(filename=dest)
        chunk_times = np.linspace(0, dest_duration-chunk_size, nchunks)


    # Load chunks of the destination file
    delays = []
    values = []
    stats = []
    for chunk_time in chunk_times:
        try:
            wd, _ = lr.load(dest, sr=sr, offset=chunk_time, 
                            duration=chunk_size)
        except ValueError:
            print(f'Chunk not read at {chunk_time}')
            break
        delay, value, s = find_chunk_in_wave_per_chunks(wref, wd)
        delays.append(delay)
        values.append(value)
        stats.append({'average':s['average_corr'],
                       'max':value,
                       'time':chunk_time})
        if DEBUG:
            print(f'chunk at time {chunk_time:.3f}: avg={s["average_corr"]:.3f}, max={value}, delay={delay/sr-chunk_time}')
        
        
    # Use the maximum correlation to determine delay of whole file
    maxidx, maxval = max(((ii, vv) for ii, vv in enumerate(values)), key = lambda x:x[1]) 
    reldel = delays[maxidx]
    delay = reldel-chunk_times[maxidx]*sr

    
    return delay, maxval, sr, stats
    

def test(ndest=2**18, nchunk=2**14, noiselev=.1):
    wdest = np.random.randn(ndest)
    pos = random.randint(0, ndest-nchunk)
    wchunk = wdest[pos:pos+nchunk] + np.random.randn(nchunk) * noiselev
    print("excerpt from pos:", pos)
    print(find_chunk_in_wave_per_chunks(wdest,wchunk))
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="delayed audio filename")
    parser.add_argument("file2", help="reference audio filename")
    
    parser.add_argument("-c", "--chunk-interval", type=float, default=1.0,
                        help="Chunk duration in seconds used for comparison")
    parser.add_argument("-o", "--overlap", type=float, default=0.5,
                        help="Overlap between chunks of the destination")
    parser.add_argument("-n", "--n-chunks", type=int, default=3,
                        help="Number of chunks in destination to use for comparison")
    parser.add_argument("-t", "--track-no-ref", type=int, default=0,
                        help="Track number for reference signal")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output extra information")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        DEBUG=args.verbose

    # Read the reference file
    sr = 16000
    wref, sr = normal_load(args.file2, sr=sr, channel=args.track_no_ref)
    ref_duration = len(wref)/sr
    dest_duration = lr.get_duration(filename=args.file1)
    
    if DEBUG:
        print(f'reference :{ref_duration:.3f} sec')
        print(f'destination :{dest_duration:.3f} sec')

    min_dist = ref_duration/args.n_chunks/2
    ti = interesting_points(wref, sr=sr, min_dist=2.0, n_peaks=args.n_chunks, 
                          tvar=args.chunk_interval)
    ti -= args.chunk_interval/2

    delay, val, sr, stats = compare_dest_to_ref(args.file1, wref, 
                                     chunk_size=args.chunk_interval,
                                     chunk_times=ti, sr=sr)

    print(f"Delay: {delay/sr} sec ({delay} samples)")



