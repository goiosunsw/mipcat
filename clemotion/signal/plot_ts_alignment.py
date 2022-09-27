import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import librosa as lr
import tgt
import pickle
from timeseries import SampledTimeSeries

def parse_args():
    # Same main parser as usual
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--wav-file',  help='wav file')
    parser.add_argument('-t','--timeseries', help='time series file (pickle)')
    parser.add_argument('-s','--segmentation',  help='segmentation file (TexGrid)')
    parser.add_argument('-c','--channel', default=6, help='channel to plot from wav')

    return parser.parse_args()


def read_timeseries(filename):
    tsdict = {}
    with open(filename,'rb') as f:
        tsl = pickle.load(f)
        for label, ts in tsl.items():
            if label.find('_f0') > -1:
                chan = label.replace('_f0','')
                tsdict['f0'] = SampledTimeSeries(ts['v'],dt=ts['dt'],t_start=ts['t0'],label=label)
        for label, ts in tsl.items():
            if label.find(chan+'_ampl'):
                tsdict['ampl'] = SampledTimeSeries(ts['v'],dt=ts['dt'],t_start=ts['t0'],label=label)
    return tsdict

def plot(args):
    fig, ax = plt.subplots(3,sharex=True)
    if args.wav_file:
        w, sr = lr.load(args.wav_file, mono=False, sr=None)
        try:
            t = np.arange(w.shape[1])/sr
            ax[0].plot(t,w[args.channel])
        except IndexError:
            t = np.arange(w.shape[0])/sr
            ax[0].plot(t,w)
            

    if args.timeseries:
        tsd = read_timeseries(args.timeseries)
        tsd['ampl'].apply(lambda x: 20*np.log10(x)).plot(ax=ax[1])
        tsd['f0'].plot(ax=ax[2])
        
    if args.segmentation:
        tg = tgt.read_textgrid(args.segmentation) 
        for tier in tg.tiers:
            if tier.name == 'clip':
                for intv in tier.intervals:
                    for axi in ax:
                        axi.axvspan(intv.start_time,intv.end_time,alpha=.1,color='green')
            if tier.name == 'notes' or tier.name == 'note':
                for intv in tier.intervals:
                    for axi in ax:
                        axi.axvspan(intv.start_time,intv.end_time,alpha=.2,color='red')
    
    plt.show()

if __name__ == '__main__':

    args = parse_args()

    plot(args)
    
