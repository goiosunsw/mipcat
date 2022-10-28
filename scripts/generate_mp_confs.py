import argparse
import json
import os
import re
from pathlib import Path
import json
import cv2
import mipcat
from mipcat.video.template_trackers import MultiAngleTemplateTracker
from mipcat.video import template_trackers

DEFAULT_CONF = {
    "hue" : [48,85],
    "saturation" : [39, 255],
    "value" : [0,255],
    "close_rad" : 1.0,
    "pos" : 0
}

FILE_PATTERN = r'.*endoscope.*\.mp4'

use_templates = ['template_25.png','template_20.png','template_30.png']

sep = os.path.sep

base_templ_path = '/'.join(os.path.abspath(template_trackers.__file__).split(sep)[:-2])+'/resources/'
templ_fn = [base_templ_path + t for t in use_templates]
templates = [cv2.imread(tt) for tt in templ_fn]
for nn, tt in zip(use_templates,templates):
    print(tt.shape)

def find_template(file):
    cap =  cv2.VideoCapture(cv2.samples.findFile(file))
    ret, img = cap.read()
    print(img.shape)
    rects = []
    vals = []
    for template in templates:
        h, w = template.shape[:2]
        trk = template_trackers.MultiAngleTemplateTracker(template, angle_step=15, size_fact=1.3, n_size=5)
        cent, angle = trk.match(img)
        cent = [int(c) for c in cent]
        rects.append([int(cent[0]-w/2), int(cent[1]-h/2), w, h])
        vals.append(trk.val)
    return rects, vals

def parse_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("root", help="root folder for search")
    ap.add_argument("-o", "--output", default="./",
                        help="start time in seconds")
    ap.add_argument("-e", "--end_sec", type=float, default=0.0,
                        help="end time in seconds")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    outdir = args.output
    if outdir[-1] != os.path.sep:
        outdir += os.path.sep
    for root, dirs, files in os.walk(args.root):
        for filename in files:
            path = os.path.join(root,filename)
            if re.match(FILE_PATTERN, path.lower()):
                print(path)
                jsfile = os.path.splitext(path)[0]+'_conf.json'
                try:
                    with open(jsfile,'r') as f:
                        js = json.load(f)
                    newjs = {}
                    for k, v in js.items():
                        if k in DEFAULT_CONF:
                            newjs[k] = v
                except IOError:
                    newjs = DEFAULT_CONF
                try:
                    rects, vals = find_template(path)
                    rect = rects[0]
                    if "rect" not in newjs:
                        newjs["rect"] = [[rect[0],rect[1]],[rect[0]+rect[2],rect[1]+rect[3]]]
                    newjs["other_rects"] = {nn:{'rect': rect, 'val': val} 
                                            for nn,rect, val in zip(use_templates, rects, vals)}
                except AttributeError:
                    print("could not read "+path)
                thisoutdir = root.replace(args.root, outdir)
                newjsfile = os.path.splitext(filename)[0]+'_conf.json'
                outfile = os.path.join(thisoutdir, newjsfile)
                
                #print(outfile)
                Path(os.path.split(outfile)[0]).mkdir(parents=True, exist_ok=True)
                
                with open(outfile,"w") as f:
                    json.dump(newjs, f)
                    
