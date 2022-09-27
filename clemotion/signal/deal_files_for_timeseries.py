#!/usr/bin/env python3
import sys
import os
import yaml

"""
    Deals files and arguments for distributed processing of 
    wave files

    takes in a combined runsheet as argument
"""

BASEL_CHANNEL_PRIORITY = ['barrel', 'external']
DEFAULT_BASE_CHAN = 'external'


def get_base_chan(chan_desc):

    base_chan = DEFAULT_BASE_CHAN

    for base_chname in BASEL_CHANNEL_PRIORITY:
        for chan in chan_desc:
            if chan['keep'] and chan['name'] == base_chname:
                return chan['name']
    return base_chan

if __name__ == '__main__':
    root_file = sys.argv[1]
    fnbr = int(sys.argv[2])

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
        for wf in wavfiles:
            filename = os.path.join(data_dir,wf['filename'])
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
                exit(0)
            ii+=1
    exit(1)



            


