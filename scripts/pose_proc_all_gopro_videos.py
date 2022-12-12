import argparse
import json
import os
import re
import pandas as pd
from mipcat.video.face.mp_pose_detect import pose_detect_on_file, pickle_results

def process_file(video_file, output_file, rotate=0, crop=None):
    print(video_file)
    results = pose_detect_on_file(video_file, rotate=rotate)    
    pickle_results(results, output_file)

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
    gpdir = 'GOPRO'
    outdir = args.output
    if outdir[-1] != os.path.sep:
        outdir += os.path.sep
                    
    df = pd.read_csv(args.video_list, header=0, index_col=None)
    df = df[((df['view']=='Front')|(df['view']=='Side'))&(df['view_type']=='new')]
    print(df.shape)

    for irow, row in df.iterrows():
        rot = 0
        if row['view'] == 'Side':
            rot=270
        video_path = f"{args.root}/{row.subj_dir}/{gpdir}/{row['view']}/{row.filename}"
        output_fn = os.path.splitext(row.filename)[0] + "_pose.pickle"
        output_path = f"{args.output}/{row.subj_dir}/{gpdir}/{row['view']}/{output_fn}"
        
        crop = None

        process_file(video_path,  output_file=output_path, rotate=rot, crop=crop)
