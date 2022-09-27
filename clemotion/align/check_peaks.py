import sys
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr

NFFT=2048*8

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="audio filename")
    
    parser.add_argument("-c", "--channel", type=int, default=0,
                        help="Channel number for reference signal")
    parser.add_argument("-n", "--nfft", type=int, default=2048,
                        help="fft window for spectrogram")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output extra information")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    wavfile = args.filename
    w, sr = lr.load(wavfile,sr=None)
    if len(w.shape) > 1:
        w = w[args.channel,:]
    
    pkfile = os.path.splitext(wavfile)[0] + '_spec_peaks.pickle'
    with open(pkfile, 'rb') as f:
        data = pickle.load(f)
    
    plt.figure()
    plt.specgram(w,Fs=sr,NFFT=NFFT,window=np.hanning(NFFT))

    time = data['peaks'][:,0]*data['dt']
    freqs = data['peaks'][:,1]*data['df']
    plt.plot(time, freqs, '.r')
    plt.show()