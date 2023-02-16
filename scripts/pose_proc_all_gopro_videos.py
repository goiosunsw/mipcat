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
                    
    with open(args.video_list, 'r') as fh:
        for line in fh:
            rot = 0
            if line.find('Side') >-1:
                rot=270
            video_path = args.root+line.strip()
            output_path = f"{args.output}/{line.strip()}"
            output_path = os.path.splitext(output_path)[0]+'_pose.pickle'
            
            crop = None

            process_file(video_path,  output_file=output_path, rotate=rot, crop=crop)
