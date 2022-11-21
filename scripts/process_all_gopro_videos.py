import argparse
import json
import os
import re
import pandas as pd
from mipcat.video.aruco_tracker import ArucoTracker

def process_file(video_file, output_file, crop=None):
    print(video_file)
    
    trk = ArucoTracker(video_file, progress=True, crop=crop)
    try:
        trk.run()
    except AttributeError:
        import traceback
        traceback.print_exc()
    finally:
        trk.to_pickle(output_file)

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
        video_path = f"{args.root}/{row.subj_dir}/{gpdir}/{row['view']}/{row.filename}"
        output_fn = os.path.splitext(row.filename)[0] + "_markers.pickle"
        output_path = f"{args.output}/{row.subj_dir}/{gpdir}/{row['view']}/{output_fn}"
        
        crop = None
        if row['view'] == 'Side':
            crop = [300,0,2000,2000]
        process_file(video_path,  output_file=output_path, crop=crop)
