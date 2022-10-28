import argparse
import json
import os
import re
import pandas as pd
from mipcat.video.mouthpiece_tracker import MouthpieceTracker

def process_file(video_file, config, output_file):
    print(video_file)
    processor = MouthpieceTracker(progress=True)
    processor.set_video_source(video_file)
    processor.read_config(config)
    processor.run()
    processor.to_pickle(output_file)

def parse_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("video_list", help="CSV wilth video file info")
    ap.add_argument("-o", "--output", default="./",
                        help="output folder (also containing configs)")
    ap.add_argument("-r", "--root", default="./",
                        help="video root folder")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    outdir = args.output
    if outdir[-1] != os.path.sep:
        outdir += os.path.sep
                    
    df = pd.read_csv(args.video_list, header=0, index_col=None)
    df = df[(df['view']=='Endoscope')&(df['view_type']=='new')]
    print(df.shape)

    for irow, row in df.iterrows():
        video_path = f"{args.root}/{row.subj_dir}/{row['view']}/{row.filename}"
        cfg_fn = os.path.splitext(row.filename)[0] + "_conf.json"
        output_fn = os.path.splitext(row.filename)[0] + "_results.pickle"
        config_path = f"{args.output}/{row.subj_dir}/{row['view']}/{cfg_fn}"
        output_path = f"{args.output}/{row.subj_dir}/{row['view']}/{output_fn}"
        
        process_file(video_path, config=config_path, output_file=output_path)