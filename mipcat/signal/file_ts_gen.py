#!/usr/bin/env python3
"""
    Call time series generator for a single wve file with a channel
    configuration YAML
"""

import os
import sys
import logging
from pathlib import Path
import yaml
import timeseries_generator as tsg

BASEL_CHANNEL_PRIORITY = ['barrel', 'external']
DEFAULT_BASE_CHAN = 'external'


def get_base_chan(chan_desc):

    base_chan = DEFAULT_BASE_CHAN

    for base_chname in BASEL_CHANNEL_PRIORITY:
        for chan in chan_desc:
            if chan['keep'] and chan['name'] == base_chname:
                return chan['name']
    return base_chan


def get_ts_args(root_file, fnbr):
    basepath = os.path.split(os.path.abspath(root_file))[0]
    with open(root_file, 'r') as f:
        main_yml = yaml.safe_load(f)

    channel_desc = main_yml['channel_desc']
    runsheets = main_yml['recordings']
    data_dir = main_yml['data_folder']

    # build filelist

    ii = 0
    for rec in runsheets:
        wavfiles = rec['wave_files']
        rec_dir = rec['root_dir']
        for wf in wavfiles:
            filename = os.path.join(rec_dir,wf['filename'])
            # filename = wf['filename']
            if ii == fnbr:
                print(filename)
                print('Generating config YAML')
                if wf['instrument'] == 'lab':
                    ch_conf='default'
                else:
                    ch_conf='generic_clarinet'
                    
                chan_desc = channel_desc[ch_conf] 

                cfg_dict = {'channel_desc': chan_desc,
                            'base_channel': get_base_chan(chan_desc)}
                print(cfg_dict)
                return data_dir, filename, chan_desc, get_base_chan(chan_desc)
            ii+=1
    return None

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    root_file = sys.argv[1]
    fnbr = int(sys.argv[2])
    out_dir = sys.argv[3]

    data_dir, filename, chan_desc, base_chan = get_ts_args(root_file, fnbr)
    fullpath = os.path.join(data_dir, filename)

    tsl = tsg.process_wav(fullpath, chan_desc, base_chan=base_chan)
    tsfile = os.path.splitext(filename)[0] + '_ts.pickle'

    out_file = os.path.join(out_dir, tsfile)
    this_dir = os.path.split(out_file)[0]
    print(this_dir)
    Path(this_dir).mkdir(parents=True, exist_ok=True)
    tsg.ts_to_pickle(tsl, out_file)


