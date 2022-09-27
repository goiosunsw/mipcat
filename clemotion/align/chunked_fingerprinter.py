from typing import ForwardRef
import pickle
import os
import argparse
import librosa as lr
import numpy as np
import tqdm
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import  binary_erosion
from sklearn.neighbors import KDTree

def nearpow2(x):
    return int(2**(np.round(np.log2(x))))

def get_duration(filename):
    try:
        import soundfile
        return soundfile.info(filename).duration
    except Exception:
        return None

def detect_peaks(image, thresh=None, xrad=2, yrad=1):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    neighborhood = np.ones((xrad,yrad))

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    if thresh is not None:
        detected_peaks = local_max & (image>thresh)

    return detected_peaks


def get_n_channels(filename, duration=0.01):
    w, sr = lr.load(filename, mono=False, sr=None, duration=duration)
    if len(w.shape)>1:
        return(w.shape[0])
    else:
        return 1

class ChunkedFingerprinter(object):
    def __init__(self, filename, channel=0, fmin=60, fmax=11000,
                 chunk_sec=15.0, frame_sec=0.05, pct_thresh=99,
                 trad=0.3, frad=160):

        self.sr = lr.get_samplerate(filename)
        self.n_channels = get_n_channels(filename)
        self.channel = channel
        self.frame_length = nearpow2(frame_sec*self.sr)
        self.hop = self.frame_length//2
        self.dt = self.hop/self.sr
        self.df = self.sr/self.frame_length
        self.f_idx_min = int(np.round(fmin/self.df))
        self.f_idx_max = int(np.round(fmax/self.df))
        self.frames_per_block = int(chunk_sec/frame_sec)
        duration = get_duration(filename)
        if duration is not None:
            self.n_blocks = int(np.ceil(duration*self.sr/self.hop/self.frames_per_block))
        else:
            self.n_blocks = None 
        self.stream = lr.stream(filename,
                                mono=False,
                                block_length=self.frames_per_block,
                                frame_length=self.frame_length,
                                hop_length=self.hop)
        self.thresh = 0.0
        self.pk_pct = pct_thresh
        self.frad = frad
        self.trad = trad
        self.n_f_rad = int(np.round(self.frad/(self.sr/self.frame_length)))
        self.n_t_rad = int(np.round(self.trad/(self.dt)))

    def process(self):
        feat = []
        frame_offset = 0
        ampl = []
        for y_block in tqdm.tqdm(self.stream, total=self.n_blocks):
            if self.n_channels>1:
                y_block = y_block[self.channel,:]

            if len(y_block) < self.frame_length:
                new_block = np.zeros(self.frame_length)
                new_block[:len(y_block)] = y_block
                y_block = new_block

            m_block = np.abs(lr.stft(y_block,
                                     n_fft=self.frame_length,
                                     hop_length=self.hop,
                                     center=False))**2

            # update thresholds
            this_thresh = np.percentile(m_block,self.pk_pct) 
            self.thresh = max(this_thresh, self.thresh)

            # binary image with spectrogram peaks
            spks = detect_peaks(m_block, xrad=self.n_f_rad, yrad=self.n_t_rad, 
                                thresh=self.thresh)
          
            # add to list of peaks 
            for ii,(xx,yy) in enumerate(zip(*np.where(spks))):
                if (xx > self.f_idx_min) & (xx < self.f_idx_max):
                    feat.append([frame_offset+yy, xx, m_block[xx,yy]])

            # amplitudes
            ampl.extend(np.sum(m_block[self.f_idx_min:self.f_idx_max,:],axis=0).tolist())

            frame_offset += self.frames_per_block
            
        feat = np.array(feat)

        # filter frames that were filtered with lower threshold
        feat = feat[feat[:,2]>=self.thresh]
        self.peaks = feat
        self.power = np.array(ampl) 
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="audio filename")
    
    parser.add_argument("-c", "--channel", type=int, default=0,
                        help="Channel number for reference signal")
    parser.add_argument("-o", "--output", type=str, default='',
                        help="Output file")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output extra information")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cf = ChunkedFingerprinter(args.filename, channel=args.channel)
    cf.process()
    data = {'peaks':cf.peaks,
            'power':cf.power,
            'dt':cf.dt,
            'df':cf.df}

    if args.output:
        output = args.output
    else:
        output = os.path.splitext(args.filename)[0] + '_spec_peaks.pickle'

    with open(output,'wb') as f:
        pickle.dump(data,f)
    
