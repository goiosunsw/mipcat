import os
import sys
import argparse
import yaml
import tgt

IMPORT_REGIONS = True
CHANNEL_DESC = 'channel_desc.yaml'
MELODIES_FILE = 'melodies.yaml'
TG_ROOT_DIR = '/srv/scratch/z3227932/Data/2021_SensorClarinet/'

if __name__ == "__main__":
    root_file = sys.argv[1]
    basepath = os.path.split(os.path.abspath(root_file))[0]
    with open(root_file, 'r') as f:
        main_yml = yaml.safe_load(f)

    with open(os.path.join(basepath, CHANNEL_DESC), 'r') as f:
        chan_desc = yaml.safe_load(f)

    with open(os.path.join(basepath, MELODIES_FILE), 'r') as f:
        melodies = yaml.safe_load(f)
    melody_files = set()

    # chan_desc = {}
    recs = []

    for dd in main_yml['runsheets']:
        with open(os.path.join(basepath, dd['runsheet']), 'r') as f:
            this_yml = yaml.safe_load(f)
        melody_files.add(this_yml['melodies_file'])
        del this_yml['melodies_file']
        # chan_desc.update(this_yml['channel_desc'])
        del this_yml['channel_desc']

        if IMPORT_REGIONS:
            for fd in this_yml['wave_files']:
                
                tg_filename = fd['notes_textgrid']
                
                tg = tgt.read_textgrid(os.path.join(TG_ROOT_DIR,
                                                    this_yml['root_dir'],
                                                    tg_filename))
                tier = tg.get_tier_by_name('clip')
                annots = []
                for annot in tier.annotations:
                    d = {}
                    d['start_time'] = float(annot.start_time)
                    d['end_time'] = float(annot.end_time)
                    d['label'] = annot.text
                    annots.append(d)
                fd['regions'] = annots
        
        this_dd = this_yml#['runsheet']
        this_root = this_dd['root_dir']
        recs.append( this_dd)
        del dd['runsheet']
        recs[-1].update(dd)

    # main_yml['melody_files'] = list(melody_files)
    # melodies = {}
    # for mf in melody_files:
    #     with open(os.path.join(basepath, mf), 'r') as f:
    #         this_mel = yaml.safe_load(f)
    #     melodies.update(this_mel)
    # main_yml['melodies'] = melodies


    main_yml['channel_desc'] = chan_desc
    main_yml['recordings'] = recs
    del main_yml['runsheets']

    print(yaml.dump(main_yml, sort_keys=False))
