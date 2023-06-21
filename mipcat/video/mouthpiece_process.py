#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Processing of video of the mouthpiece scale

    This file is meant to be used with the 'opencv-contrib' environment

    Runs a color-based processing of the reed mouthpiece video. 
    Needs a configuration input in json format
"""

import os
import argparse
import json
import pickle
import traceback
import numpy as np
import scipy.signal as sig
import cv2
from tqdm import tqdm


def length(x):
    return np.sqrt((x[2]-x[0])**2+(x[3]-x[1])**2)


def angle(x):
    return np.arctan(-(x[3]-x[1])/(x[2]-x[0]))/np.pi*180


def swap_rect_dims(rect):
    return np.array([rect[1], rect[0], rect[3], rect[2]])



class FrameProcessor(object):
    def __init__(self, config_file=None, 
                 video_file=None, pos=0,
                 progress=False):

        self.progress = progress
        self.min_area = 200

        self.lower_color = np.array([0,0,0])
        self.upper_color = np.array([179,255,255])
        self.fill_closure_rad = 1
        
        if video_file:
            self.set_video_source(video_file, from_time=pos)
        else:
            return
        if config_file is not None:
            try:
                self.read_config(config_file)
            except Exception:
                traceback.print_exc()
                
        else:
            return
        
        self.has_hsv_config = False
        
        #self.process()


    def read_config(self, jsfile):
        with open(jsfile,'r') as f:
            jsc = json.load(f)
        self._config(jsc)

    def _config(self,jsc):
        #self.set_color_range(jsc)
        self.lower_color = np.array([jsc[x][0] for x in ['hue','saturation','value']])
        self.upper_color = np.array([jsc[x][1] for x in ['hue','saturation','value']])
        
        try:
            self.anchor_pt = [int(x) for x in jsc['anchor']]
        except KeyError:
            pass
        try:
            self.fill_closure_rad = int(jsc['close_rad'])
        except KeyError:
            self.fill_closure_rad = 1

    def config_to_json(self):
        jsc = {'hue':[int(self.lower_color[0]), int(self.upper_color[0])],
                'saturation':[int(self.lower_color[1]), int(self.upper_color[1])],
                'value':[int(self.lower_color[2]), int(self.upper_color[2])],
                'colse_rad':int(self.fill_closure_rad)}
        return jsc

    def set_color_range(self, jsc):
        self.lower_color = np.array([jsc[x]['min'] for x in ['hue','saturation','value']])
        self.upper_color = np.array([jsc[x]['max'] for x in ['hue','saturation','value']])

    def color_convert(self):
        # convert to HSV
        hsv = cv2.cvtColor(self.img,cv2.COLOR_BGR2HSV)
        self.hsv = hsv
        # find regions of appropriate color
        if self.lower_color[0] <= self.upper_color[0]:
            mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        else:
            lc = self.lower_color.copy()
            uc = self.upper_color.copy()
            lc[0] = self.upper_color[0]
            uc[0] = 179
            mask1 = cv2.inRange(hsv,lc,uc)
            lc[0] = 0
            uc[0] = self.lower_color[0]
            mask2 = cv2.inRange(hsv,lc,uc)
            mask = cv2.bitwise_or(mask1,mask2)
        self.green_mask = mask

    def find_nearest_blob(self, mask, anchor_pt, minArea=0, minDist=1e12):
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        idx=np.flatnonzero((stats[1:,4]>minArea))+1
        
        for ii in idx:
            xx,yy = np.where(labels==ii)
            dists = ((yy-anchor_pt[0])**2+(xx-anchor_pt[1])**2)
            thisidx = np.argmin(dists)
            if dists[thisidx]<minDist:
                minDist = dists[thisidx]
                minpt = (yy[thisidx],xx[thisidx])
                minlab = ii
        
        return (labels==minlab), minpt


    def find_green_strip(self):
        # find blob nearest to point
        rad = self.fill_closure_rad
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(rad,rad))
            closed_mask = cv2.morphologyEx(self.green_mask,cv2.MORPH_DILATE,kernel)
            masks,new_pt = self.find_nearest_blob(closed_mask, self.anchor_pt, 
                                             minArea = self.min_area)
        except UnboundLocalError:
            ret = {'area':np.nan, 'filled_area':np.nan,
            'minRect':((np.nan,np.nan),(np.nan,np.nan),np.nan), 
                'rectPts':np.ones((4,2))*np.nan, 'mask':self.green_mask}
            return ret
        
        strip_mask = masks
        # cv2.bitwise_and(self.green_mask,masks)

        self.final_mask = strip_mask.astype(np.uint8)
        self.filled_mask = strip_mask.astype(np.uint8)

        
    def crop_strip(self, origin_pt, angle):
        h, w = self.img.shape[:2]
        lpt = origin_pt[1]-origin_pt[0]*np.tan(angle)
        rpt = origin_pt[1]+(w-origin_pt[0])*np.tan(angle)
        if rpt<0 or lpt>h:
            pt1 = (origin_pt[0]-origin_pt[1]/np.tan(angle),0)
            pt2 = (origin_pt[0]+(h-origin_pt[1])/np.tan(angle),h)
            pts = [(0,0), pt1, pt2, (0,h)]
        else:
            pt1 = (0,lpt)
            pt2 = (w,rpt)
            pts = [(0,0), pt1, pt2, (w,0)]
        #print(h,w,angle,pt1,pt2)
        pts = np.array([[pt] for pt in pts]).astype('i')
        crop_mask = np.zeros((h,w),dtype='uint8')
        cv2.fillPoly(crop_mask, [pts] ,255)
        #ret, crop_mask = cv2.threshold(crop_mask, 128, 255, cv2.THRESH_BINARY)
        self.final_mask = cv2.bitwise_and(self.final_mask,crop_mask)
        self.filled_mask = cv2.bitwise_and(self.filled_mask,crop_mask)

    def measure(self):
        # get the bounding rectangle corresponding to the blob of interest
        contours, _ = cv2.findContours(self.filled_mask.astype('uint8')*255, 
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.area = 0
        self.filled_area = 0
        
        if len(contours)<1:
            return
        
        c = np.concatenate(contours)
        minRect = cv2.minAreaRect(c)    
        pts = cv2.boxPoints(minRect)

        self.area = np.sum(self.final_mask > 0)
        self.filled_area = np.sum(self.filled_mask > 0)
        self.new_rect = minRect
        self.rect_pts = pts

    def set_video_source(self, video_file, from_time=0.0, to_time=None):
        self.video_file = video_file
        cap = cv2.VideoCapture(cv2.samples.findFile(video_file))
        ret, img = cap.read()
        self.frame_size = img.shape[1::-1]
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rate = (cap.get(cv2.CAP_PROP_FPS))

        cap.set(cv2.CAP_PROP_POS_MSEC,from_time*1000)
        self.cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if to_time is not None:
            self.max_time = to_time
        else:
            self.max_time = length/rate
            
        self.min_time = from_time
        self.frame_msec = 1000/rate
        length = int((self.max_time - self.min_time)*rate)
        print("Video length:", length)
        
        self.cap = cap
        self.n_frames = length
        #self.init_references()
        #self.process(img)
        ret, self.img = self.get_frame()

    def get_frame(self, pos=None, time=None):
        if pos is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        if time is not None:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, int(time*1000))
        ret, self.img = self.cap.read()
        self.time = self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
        if not ret:
            print("Error reading frame number ", pos)
        return ret, self.img

    def image_results(self):
        mm = self.final_mask
        ims = self.img/np.max(self.img)/2 
        ims[:,:,0] += mm/np.max(mm)/2

        ims = (ims*255).astype('uint8')

        

        #cv2.imshow("image",ims)

        return ims.astype('uint8')
    
    @property
    def roi_cent(self):
        
        # get the bounding rectangle corresponding to the blob of interest
        contours, _ = cv2.findContours(self.roi_mask, 
                                    cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        c=contours[0]
        minRect = cv2.minAreaRect(c)    

        x,y=np.where(self.roi_mask)

        w = np.max(x)-np.min(x)
        h = np.max(y)-np.min(y)

        return [minRect[0][0]-np.min(y)+w//2,minRect[0][1]-np.min(x)+h//2]

    def roi_results(self):
        return self.roi_img_prev


    def process(self, img):
        if img is None:
            ret, self.img = self.get_frame()
        else:
            self.img = img
        self.color_convert()
        self.find_green_strip()
        
        self.measure()
        self.this_res = {'area':int(self.area),
                         'filled_area':int(self.filled_area),
                         'minRect':self.new_rect}
        
    def run(self, n=None):
        if n is None:
            n = self.n_frames
        self.results=[]
        if self.progress:
            self.pbar = tqdm(total=self.n_frames)
        for ii in range(n):
            ret, image = self.get_frame()
            if not ret:
                self.pbar.write(f'Error reading frame {ii}. Skipping')
                continue
            self.process(image)
            self.this_res['time'] = self.time
            self.results.append(self.this_res)
            if self.progress:
                self.pbar.update()
        if self.progress:
            self.pbar.close()

    def to_json(self, filename=None):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        if filename is not None:
            with open(filename,'w') as f:
                json.dump(self.results,f,cls=NumpyEncoder)
        else:
            with open(os.path.splitext(self.video_file)[0]+'_mp_video_analysis.json','w') as f:    
                json.dump(self.results,f,cls=NumpyEncoder)

    def to_pickle(self, filename=None):
        if filename is None:
            filename = os.path.splitext(self.video_file)[0]+'_mp_video_analysis.pickle'
        with open(filename,'wb') as f:
            pickle.dump(self.results, f)
                
def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="movie filename")
    parser.add_argument("-s", "--start_sec", type=float, default=0.0,
                        help="start time in seconds")
    parser.add_argument("-e", "--end_sec", type=float, default=0.0,
                        help="end time in seconds")
 
    parser.add_argument("-c", "--config", 
                        help="Config json file")
    parser.add_argument("-g", "--gui", action='store_true',
                        help="GUI (test mode)")
    parser.add_argument("-o", "--output", default="", 
                        help="output file (leave empty for same name as video)")

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parse()
    vid_file = args.filename
    basename = os.path.splitext(vid_file)[0]

    if args.config:
        color_range_json_file = args.config
    else:
        color_range_json_file = basename + '_conf.json'

    if len(args.output)==0:
        output = basename+'_mp_video_analysis.json'
    else:
        output = args.output 

    if args.end_sec <= 0.001:
        end_time = None
    else:
        end_time = args.end_sec

    processor = FrameProcessor(config_file=color_range_json_file, 
                                progress=True)
    processor.set_video_source(args.filename, from_time=args.start_sec, to_time=end_time)
    if args.gui:
        processor.gui()
    else:
        processor.run()
        processor.to_json(output)

