import pandas as pd
import os
import argparse
import wave
import re
import pickle
import numpy as np
from mipcat.signal.timeseries_generator import ts_from_pickle, ts_to_pickle
from timeseries import SampledTimeSeries, TimeSeries
from scipy.io import wavfile
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import RANSACRegressor
import scipy.signal as sig
import librosa as lr

templ_names = ['template_20.png', 'template_25.png', 'template_30.png'] 
scales = {}

def get_video_data(video_file, rot=0):
    p
        try:
            with open(pose_file, 'rb') as f:
                pose_data = pickle.load(f)
        except FileNotFoundError:
            print(f"Not found : {pose_file}")
            continue
        try:
            with open(aruco_file, 'rb') as f:
                aruco_data = pickle.load(f)
        except FileNotFoundError:
            print(f"Not found : {aruco_file}")
            continue

def process_mouthpiece_data(area_data):
    time = np.array([x['time'] for x in area_data])
    area = np.array([x['area'] for x in area_data])
        
    # Scale data
    tp=np.array([[x['other_rects'][tn]['rect'] for tn in templ_names] for x in area_data if  'other_rects' in x])
    tt=np.array([x['time'] for x in area_data if  'other_rects' in x])

    dists = np.sqrt(np.sum(np.diff(np.array([tp[:,:,0]+(tp[:,:,2])/2, tp[:,:,1]+(tp[:,:,3])/2]).transpose([1,2,0]),axis=1)**2,axis=2))
    
    # Average strip width
    sw = np.min(np.median(np.array([x['minRect'][1] for x in area_data]),axis=0))
    print("Strip width (px) :", sw)

    # Equiv. strip length
    len_px = area/sw

    if dists.shape[0]>1:
        
        #xf = UnivariateSpline(tt,dists[:,ii])
        X = np.array([tt]).T
        mod = RANSACRegressor().fit(X,dists[:,0])
        #print(mod.score(X,dists[:,ii]))
        print(np.sum(mod.inlier_mask_)/X.shape[0], "% inliers")
        print(mod.estimator_.coef_, mod.estimator_.intercept_)
        scales[vid_file] = np.mean(mod.predict(X))

        len_mm = len_px*5/mod.predict(np.array([time]).T)
    elif dists.shape[0]==1:
        # Use first distance: it should have less obscuring
        len_mm = len_px*5/dists[0,0]
    else:
        len_mm = len_px*5/np.mean(list(scales.values()))

    return time, len_mm

def cut_mouthpiece_timeseries(time, len_mm, ts_path, t_start):

    tsl = ts_from_pickle(ts_path)
    t_end = t_start+tsl[0].t[-1]
    print(f"{t_start:8.3f}:{t_end:8.3f} : {row['pct']:6.2f} : {wavpath}")
    idx = (time>t_start)&(time<t_end)
    try:
        ats = TimeSeries(t=time[idx]-t_start,v=len_mm[idx])
    except ValueError:
        ats = None
    return ats



def parse_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("filename", help="csv file with wav to video alignment")
    ap.add_argument("root", help="root folder for search")
    ap.add_argument("-o", "--output", default="note_list.csv",
                        help="output file")
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()

    wvdfo = pd.read_csv(args.filename,index_col=0)
    # Add view information
    wvdfo['view'] = "mouthpiece"
    wvdfo.loc[wvdfo.video_path.str.lower().str.contains('gopro'),'view'] = 'GOPRO'
    wvdfo.loc[wvdfo.video_path.str.lower().str.contains('side'),'view'] = 'side'
    wvdfo.loc[wvdfo.video_path.str.lower().str.contains('front'),'view'] = 'front'

    # List video files for each wavfile
    wav_vid_df = wvdfo.pivot(index='wav_path',columns='view',values='video_path')[["side","front"]]
    wav_vid_df = wav_vid_df[(~wav_vid_df.isna()).any(axis=1)]
    for (side_video, front_video), wavs in wav_vid_df.reset_index().groupby(['side','front']):
        side_data = get_video_data(side_video, rot=270)
        front_data = get_video_data(front_video, rot=0)
        for w in wavs.wav_path:
            print("  "+w)
            angle = process_gopro_data(front_data, side_data)
    for vid_file, grp in wvdf.groupby('video_path'):
        
        time, angle = process_gopro_data(aruco_data, pose_data)

        for irow, row in grp.iterrows():
            wavpath = row['wav_path']
            ts_path = os.path.join(args.root,wavpath.replace('.wav','_ts.pickle'))
            out_path = os.path.join(args.root,wavpath.replace('.wav','_mouthpiece_ts.pickle'))
            t_start = -row['delay']
            try:
                ats = cut_mouthpiece_timeseries(time, len_mm, ts_path, t_start)
            except FileNotFoundError:
                print(f"Not found: {ts_path}")
                continue
            print(f"Writing to {out_path}")
            with open(out_path,'wb') as f:
                pickle.dump({'t':ats.t,'v':ats.v}, f) 
             
    
