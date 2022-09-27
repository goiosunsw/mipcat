#!/usr/bin/env python3
"""
    Generate timeseries for a list of files from a minimal CSV containing:
    * file (file path)
    * channel set
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import librosa as lr
import yaml
#import .timeseries_generator as tsg
from . import timeseries_generator as tsg

BASEL_CHANNEL_PRIORITY = ['barrel', 'external', 'farfield']
DEFAULT_BASE_CHAN = 'farfield'


def get_base_chan(chan_desc, wav_file_name=None):

    if wav_file_name:
        w,sr = lr.load(wav_file_name,duration=1.0, sr=None, mono=False)
        nchans = w.shape[0]
        chan_desc = chan_desc[:nchans]
        
    base_chan = DEFAULT_BASE_CHAN

    for base_chname in BASEL_CHANNEL_PRIORITY:
        for chan in chan_desc:
            if chan['keep'] and chan['name'] == base_chname:
                return chan['name']
    return base_chan


def get_chans(channel_desc_file, set_name, wav_file_name=None):
    basepath = os.path.split(os.path.abspath(channel_desc_file))[0]
    with open(channel_desc_file, 'r') as f:
        channel_desc = yaml.safe_load(f)

    if wav_file_name:
        w,sr = lr.load(wav_file_name,duration=1.0, sr=None)
        nchans = w.shape[0]
        channel_desc = channel_desc[:nchans]

    channel_set = channel_desc[set_name]
    base_chan = get_base_chan(channel_set)

    return channel_set, base_chan


def process_single(filename, chan_set=None, output=None):
    if output is None:
        output = os.path.splitext()[0]+'_ts.pickle'
    
    base_chan = get_base_chan(chan_set, wav_file_name=filename)
    tsl = tsg.process_wav(filename, chan_set, base_chan=base_chan)
    tsg.ts_to_pickle(tsl, output)
    

def process_csv(csvfile, chan_desc_file, outputdir, test=False, root_dir='.'):
    
    df = pd.read_csv(csvfile, index_col=0)

    with open(chan_desc_file, 'r') as f:
        chan_desc = yaml.safe_load(f)

    for irow, row in df.iterrows():
        #print(row)
        wav_path = root_dir + '/' + row.filename
        logging.info("Processing "+wav_path)
        chan_set = chan_desc[row.channel_set]
        outputfile = outputdir + '/' + os.path.splitext(row.filename)[0]+'_ts.pickle'
        this_dir = os.path.split(outputfile)[0]
        if not test:
            Path(this_dir).mkdir(parents=True, exist_ok=True)
            process_single(wav_path, chan_set, output=outputfile)
        else:
            if os.path.isfile(wav_path):
                logging.debug(f"{wav_path} not found")
            logging.debug(f"Will output to {this_dir}")

    

def parse_args():
    # Same main parser as usual
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose', action='store_true')
    parser.add_argument('-t', '--test', help='dry-run', action='store_true')

    sub_parsers = parser.add_subparsers(dest='command', help='commands', required=True)    
    
    parser_csv = sub_parsers.add_parser('csv', help='File list from csv')
    parser_csv.add_argument('file', help='csv file (contains file and channel_set columns)')
    parser_csv.add_argument('-c','--channel-file',  help='YAML file containing channel descriptions')
    parser_csv.add_argument('-r','--root-dir', help='root folder (defaults to .)',
                            default='.')
    parser_csv.add_argument('-o','--output',  help='output dir')

    parser_file = sub_parsers.add_parser('single', help='generate from single file')
    parser_file.add_argument('file', help='wav file')
    parser_file.add_argument('-c','--channel-file',  help='YAML file containing channel descriptions',
                             required=True)
    parser_file.add_argument('-s','--set',  help='set from channel set file (otherwise single set)')
    parser_file.add_argument('-o','--output',  help='output file')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    if args.verbose or args.test:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    if args.command == 'single':
        chan_set, base_chan = get_chans(args.channel_file, args.set)
        process_single(args.file, chan_set, args.output)
    elif args.command == 'csv':
        process_csv(args.file, args.channel_file, args.output, test=args.test, root_dir=args.root_dir)
