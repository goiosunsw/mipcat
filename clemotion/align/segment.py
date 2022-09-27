import numpy as np
import tgt
import os
import argparse
import logging
from chunked_fingerprinter import ChunkedFingerprinter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="audio filename")
    
    parser.add_argument("-c", "--channel", type=int, default=0,
                        help="Channel number for reference signal")
    parser.add_argument("-m", "--min_silence", type=float, default=0.5,
                        help="Minimum duration of silence")

    parser.add_argument("-t", "--textgrid", action='store_true',
                        help="write Praat TextGrid")
    parser.add_argument("-C", "--CSV", action='store_true',
                        help="write CSV")
    parser.add_argument("-T", "--TSV", action='store_true',
                        help="write TSV (Audacity)")
    parser.add_argument("-o", "--output", type=str, default='',
                        help="Output file")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output extra information")
    return parser.parse_args()


def segment_file(filename, channel, pct=5, min_silence=.5):
    
    cf = ChunkedFingerprinter(filename, channel=channel)
    cf.process()

    a = np.log(cf.power)
    t = np.arange(len(a))*cf.dt + cf.dt/2
    min_fr = int(min_silence/cf.dt)

    amin = np.percentile(a,pct)
    amax = np.percentile(a,(100-pct))

    ath = (amin+amax)/2
    clusters = a>ath
    
    i=0
    off=True
    nframes = len(clusters)
    
    tst=[]
    tend=[]
    
    while i+min_fr < nframes:
        if clusters[i]>0 and off:
            tst.append(t[max(0,i-1)])
            off=False
            i+=1
        elif clusters[i]<=0 and not off:
            if all(clusters[i:i+min_fr] <=0 ):
                off=True
                tend.append(t[i])
                i+=min_fr
            else:
                i+=1
        else:
            i+=1
    if not off:
        tend.append(t[-1])
    return tst,tend

def write_CSV(filename, tst, tend, sep=','):
    with open(filename,'x') as f:
        for ii ,(ts, te) in enumerate(zip(tst,tend)):
            f.write(f'{ts}{sep}{te}{sep}Region{ii}\n')
        
def write_TextGrid(filename, tst, tend, tmax=None):
    if tmax is None:
        tmax = np.max(tend)

    tg = tgt.TextGrid()

    inttier = tgt.IntervalTier(start_time=0, end_time=tmax, name='regions')
    tg.add_tier(inttier)

    for ii, it in enumerate(zip(tst,tend)):
        u = it[0]
        d = it[1]
        try:
            text = it[2]
        except IndexError:
            text = f'Region {ii}'
        inttier.add_interval(tgt.Interval(u, d, text))
    tgt.write_to_file(tg, filename)


if __name__ == '__main__':
    args = parse_args()

    if args.output:
        output = args.output
    else:
        output = os.path.splitext(args.filename)[0] + '_reg'
    
    tst, tend = segment_file(args.filename, args.channel, min_silence=args.min_silence)

    output_done = False
    if args.textgrid:
        write_TextGrid(output+'.TextGrid',tst,tend)
        output_done = True
    if args.CSV:
        write_CSV(output+'.CSV',tst,tend)
        output_done = True
    if args.TSV:
        write_CSV(output+'.txt',tst,tend,sep='\t')
        output_done = True

    if not output_done:
        for ts, te in zip(tst,tend):
            print(f'{ts:8.4f}, {te:8.4f}')

    