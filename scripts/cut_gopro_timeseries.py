import pandas as pd
import os
import argparse
import wave
import re
import pickle
import numpy as np
from numpy import ma
from mipcat.signal.timeseries_generator import ts_from_pickle, ts_to_pickle
from timeseries import SampledTimeSeries as TimeSeries
from scipy.io import wavfile
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import RANSACRegressor
import scipy.signal as sig
import librosa as lr
from pykalman import KalmanFilter


VIEW_CONFIG = {
    'Side': {
        'marker_ids': [5, 9],
        'rotate': 270
    },
    'Front': {
        'marker_ids': [6, 10],
        'rotate': 0
    }
}

def kfilter(x, observation_error=0.05, observation_cov=0.001,
               pos_proc_error=0.15, pos_proc_xy_cov=0.015,
               vel_proc_error=0.25, vel_proc_xy_cov=0.05,
               pos_vel_proc_xx_cov=0.1, pos_vel_proc_xy_cov=0.025,
               outl_cov_fact=100000, outl_pos_thresh=50):
    """
    Use Kalman to filter a XY position sequence.
    
    observation_error and observation_cov is the estimated error/cov of the measured position
    the 6 _proc_error and _proc_cov are coefficients for the covariance matrix of the model 
    (model is x_ij+1 = x_ij+v_ij*t)
    outl_cov_fact is a factor applied to process (model) transition matrix. Matrix is divided by this factor,
    making for a more aggressive filtering
    Outliers that move further than outl_pos_thresh of the filtered model are treated as missing observations
    """
    preds = np.zeros(x.shape)
    for ii in range(x.shape[1]):
        # Find nan values 
        meas = ma.array(x[:,ii,:].copy())
        naidx = np.flatnonzero(np.isnan(meas[:,0]))
        ndiff = np.diff(naidx)

        # Calculate initial state estimate based on non-nan values
        if len(ndiff)>0 and np.max(ndiff)>1:
            amax = np.argmax(ndiff)
            naidx[amax]
            mfit = meas[naidx[amax]+1:naidx[amax+1]-1]
        else:
            mfit = meas[~np.isnan(meas[:,0]),:]
        kf = KalmanFilter(initial_state_mean=np.concatenate((np.nanmean(mfit,axis=0),[0,0])),
                                                            n_dim_obs=2,n_dim_state=4)

        ### Aggressive filtering to detect outliers
        # transition matrices
        kf.transition_matrices = np.array([[1,0,1,0],[0,1,0,1],
                                           [0,0,1,0],[0,0,0,1]])

        # mask nan-values
        meas[np.isnan(meas)]=ma.masked

        kf.observation_covariance = np.array([[observation_error, observation_cov],
                                              [observation_cov, observation_error]])

        kf.transition_covariance = np.array([[pos_proc_error, pos_proc_xy_cov, pos_vel_proc_xx_cov, pos_vel_proc_xy_cov],
                                             [pos_proc_xy_cov, pos_proc_error, pos_vel_proc_xy_cov, pos_vel_proc_xx_cov],
                                             [pos_vel_proc_xx_cov, pos_vel_proc_xy_cov, vel_proc_error, vel_proc_xy_cov],
                                             [pos_vel_proc_xy_cov, pos_vel_proc_xx_cov, vel_proc_xy_cov, vel_proc_error]])/outl_cov_fact
        #pred_m,pred_cov=kf.em(mfit).smooth(meas)
        pred_m,pred_cov=kf.smooth(meas)
        pred_r = pred_m

        # Remove outliers from final filtering
        outl = np.sqrt(np.sum((meas-pred_m[:,:2])**2,axis=1))>outl_pos_thresh
        print(f"Outliers: {np.sum(outl)}/{len(outl)}")

        meas[outl,:]=ma.masked

        ### Final filtering 
        kf = KalmanFilter(initial_state_mean=np.concatenate((np.nanmean(mfit,axis=0),[0,0])),
                                                            n_dim_obs=2,n_dim_state=4)
        kf.transition_matrices = np.array([[1,0,1,0],[0,1,0,1],
                                           [0,0,1,0],[0,0,0,1]])

        kf.observation_covariance = np.array([[observation_error, observation_cov],
                                              [observation_cov, observation_error]])

        kf.transition_covariance = np.array([[pos_proc_error, pos_proc_xy_cov, pos_vel_proc_xx_cov, pos_vel_proc_xy_cov],
                                             [pos_proc_xy_cov, pos_proc_error, pos_vel_proc_xy_cov, pos_vel_proc_xx_cov],
                                             [pos_vel_proc_xx_cov, pos_vel_proc_xy_cov, vel_proc_error, vel_proc_xy_cov],
                                             [pos_vel_proc_xy_cov, pos_vel_proc_xx_cov, vel_proc_xy_cov, vel_proc_error]])

        pred_m,pred_cov=kf.smooth(meas)
    
        preds[:,ii,:] = pred_m[:,:2]
    return preds

def process_gopro_data(mrk_side, bdy_side):
    side_markers = VIEW_CONFIG['Side']['marker_ids']
    body_markers = [6,8,9,10]
    img_size = [1080,1920]

    mrk_coords = np.ones((len(mrk_side),len(side_markers),2))*np.nan
    face_coords = np.ones((len(mrk_side),len(body_markers),2))*np.nan
    side_times = np.ones((len(mrk_side)))*np.nan

    for ii, mdata in enumerate(mrk_side):
        for jj, mno in enumerate(side_markers):
            try:
                bbox = mdata[mno]['bbox']
                #cent = np.mean(mdata[mno]['bbox'],axis=0)
            except KeyError:
                try:
                    bbox = mdata[str(mno)]['bbox']
                except KeyError:
                    continue
            cent = np.array([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2])
            mrk_coords[ii,jj,:] = np.array([cent[1],cent[0]])
        side_times[ii] = mdata['msec']/1000.0
        
    for ii, mdata in enumerate(bdy_side):
        for jj, mno in enumerate(body_markers):
            try:
                cent = mdata['landmarks'][mno]
            except KeyError:
                continue
            face_coords[ii,jj,0] = cent['x'] * img_size[0]
            face_coords[ii,jj,1] = (1-cent['y']) * img_size[1]

    face_coords = kfilter(face_coords,observation_error=1/2)
    
    # clarinet angle
    dm = np.diff(mrk_coords, axis=1)
    cl_angle = np.arctan(dm[:,0,1]/dm[:,0,0])/np.pi*180
    # ear-to-eye angle
    dm = np.diff(face_coords, axis=1)
    face_angle = np.arctan(dm[:,0,1]/dm[:,0,0])/np.pi*180
    # clarinet to mouth distances
    mouth_middle = (face_coords[:,2] + face_coords[:,3])/2
    cl_length = (np.sqrt(np.sum(np.diff(mrk_coords,axis=1)**2, axis=2)))[:,0]
    # lateral distance (perpendicular to clarinet)
    lat_cl_mouth_dist = np.abs((mrk_coords[:,1,0]-mrk_coords[:,0,0])*(mrk_coords[:,0,1]-mouth_middle[:,1]) -
                 (mrk_coords[:,1,1]-mrk_coords[:,0,1])*(mrk_coords[:,0,0]-mouth_middle[:,0]))/(
                 cl_length)
    # longitudinal distance (marker-to-mouth distance)
    long_cl_mouth_dist = np.sqrt(np.sum((mrk_coords-mouth_middle[:,np.newaxis,:])**2, axis=2))

    return {'times': side_times,
            'cl_angle': cl_angle,
            'face_angle': face_angle,
            'lat_cl_mouth_dist': lat_cl_mouth_dist,
            'long_cl_mouth_dist': long_cl_mouth_dist}


def cut_gopro_timeseries(gp_data, ts_path, t_start):
    time = gp_data['times'] 
    tsl = ts_from_pickle(ts_path)
    t_end = t_start+tsl[0].t[-1]
    print(f"{t_start:8.3f}:{t_end:8.3f} : {row['pct']:6.2f} : {wavpath}")
    idx = (time>t_start)&(time<t_end)
    vtl = []
    for label, vals in gp_data.items(): 
        if label == 'times':
            continue
        try:
            ats = TimeSeries(t=time[idx]-t_start,v=vals[idx], label=label)
        except ValueError:
            ats = None
        vtl.append(ats)
    return vtl


def parse_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("filename", help="csv file with wav to video alignment")
    ap.add_argument("root", help="root folder for search")
    ap.add_argument("-o", "--output", default="note_list.csv",
                        help="output file")
    ap.add_argument('-a', "--aruco-suffix", default="_markers.pickle", 
                    help = "suffix for extracted aruco marker file")
    ap.add_argument('-p', "--pose-suffix", default="_pose.pickle", 
                    help = "suffix for extracted aruco marker file")
    return ap.parse_args()


if __name__ == '__main__':
    print("This is the GOPRO time_series collector")
    args = parse_args()

    wvdfo = pd.read_csv(args.filename,index_col=0)
    # select mouthpiece endoscope videos
    wvdfo = wvdfo[(wvdfo.video_path.str.contains("Side"))&(wvdfo.pct>1)]
    # selct best matches
    wvdf=wvdfo.groupby('wav_path').apply(lambda grp: grp.loc[grp.pct.idxmax()])

    for vid_file, vid_grp in wvdf.groupby('video_path'):
        side_file = os.path.join(args.root, vid_file.replace('.MP4',args.aruco_suffix))
        try:
            with open(side_file, 'rb') as f:
                mrk_side = pickle.load(f)
        except FileNotFoundError:
            print(f"Not found : {mrk_side}")
            continue
        side_pose_file = os.path.join(args.root, vid_file.replace('.MP4',args.pose_suffix))
        try:
            with open(side_pose_file, 'rb') as f:
                body_side = pickle.load(f)
        except FileNotFoundError:
            print(f"Not found : {body_side}")
            continue
        gp_data = process_gopro_data(mrk_side, body_side)   
        # process each gopro video
        for irow, row in vid_grp.iterrows():
            vid_file = row['video_path']
            if vid_file.lower().find("side") != -1:
                continue
            

        for irow, row in vid_grp.iterrows():
            wavpath = row['wav_path']
            ts_path = os.path.join(args.root,wavpath.replace('.wav','_ts.pickle'))
            out_path = os.path.join(args.root,wavpath.replace('.wav','_pose_ts.pickle'))
            t_start = -row['delay']
            try:
                tsl = cut_gopro_timeseries(gp_data, ts_path, t_start)
            except FileNotFoundError:
                print(f"Not found: {ts_path}")
                continue
            print(f"Writing to {out_path}")
            with open(out_path,'wb') as f:
                ts_to_pickle(tsl, out_path)
             
    
