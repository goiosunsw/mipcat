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
import numpy as np
import scipy.signal as sig
import cv2
import time
from numpy import *
from tqdm import tqdm, trange


def find_nearest_blob(mask, anchor_pt, close_rad=9, minArea=1000, mindist = 1e12):
    maskr=mask
    
    rad=close_rad
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(rad,rad))
    maskr = cv2.morphologyEx(maskr,cv2.MORPH_CLOSE,kernel)

    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(maskr)
    #avv = [np.mean(hsv[labels==rno,2]) for rno in range(numLabels)]
    #print(stats, centroids)
    idx=np.flatnonzero((stats[1:,4]>minArea))+1
    #rno=np.argmax(stats[1:,4])+1
    
    for ii in idx:
        xx,yy = np.where(labels==ii)
        dists = ((yy-anchor_pt[0])**2+(xx-anchor_pt[1])**2)
        thisidx = np.argmin(dists)
        if dists[thisidx]<mindist:
            mindist = dists[thisidx]
            minpt = (yy[thisidx],xx[thisidx])
            minlab = ii
    
    return (labels==minlab), minpt

class FrameProcessor(object):
    def __init__(self, color_range_file=None, config_file=None, 
                 video_file=None, anchor_pt=(0,0),
                 min_area=1000, filled_rad=19, scale_length=400, pos=0, blob_close_rad=9,
                 progress=False):
        if color_range_file is not None:
            self.read_color_range(color_range_file)
        self.anchor_pt=anchor_pt
        self.blob_close_rad=blob_close_rad
        self.min_area = min_area
        self.filled_rad = filled_rad
        self.scale_len = scale_length
        self.init_pos = pos
        self.progress = progress
        if video_file:
            self.set_video_source(video_file, pos=pos)
        if config_file is not None:
            self.read_config(config_file)

    def read_color_range(self, jsfile):
        with open(jsfile,'r') as f:
            jsc = json.load(f)
        self.set_color_range(jsc)
        
    def read_config(self, jsfile):
        with open(jsfile,'r') as f:
            jsc = json.load(f)
        self.lower_color = np.array([jsc[x][0] for x in ['hue','saturation','value']])
        self.upper_color = np.array([jsc[x][1] for x in ['hue','saturation','value']])
        try:
            self.anchor_pt = [int(x) for x in jsc['anchor']]
        except KeyError:
            pass
        try:
            self.blob_close_rad = int(jsc['close_rad'])
        except KeyError:
            pass

    def set_color_range(self, jsc):
        self.lower_color = np.array([jsc[x]['min'] for x in ['hue','saturation','value']])
        self.upper_color = np.array([jsc[x]['max'] for x in ['hue','saturation','value']])

    def init_references(self, img=None):
        if img is None:
            self.get_frame()
        else:
            self.img = img
        self.color_convert()
        masks, new_pt = find_nearest_blob(self.mask, self.anchor_pt,minArea=self.min_area)
        contours, _ = cv2.findContours(masks.astype('uint8')*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c=contours[0]
        minRect = cv2.minAreaRect(c)
        self.ref_rect = minRect
        self.new_rect = minRect

    def color_convert(self):
        # convert to HSV
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # find regions of appropriate color
        if self.lower_color[0] <= self.upper_color[0]:
            mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        else:
            lc = self.lower_color.copy()
            uc = self.upper_color.copy()
            #lc[0] = self.upper_color[0]
            uc[0] = 179
            mask1 = cv2.inRange(hsv,lc,uc)
            lc = self.lower_color.copy()
            uc = self.upper_color.copy()
            lc[0] = 0
            #uc[0] = self.lower_color[0]
            mask2 = cv2.inRange(hsv,lc,uc)
            mask = cv2.bitwise_or(mask1,mask2)
        self.mask = mask

    def find_blob(self):
        # find blob nearest to point
        try:
            masks, new_pt = find_nearest_blob(self.mask, self.anchor_pt, 
                                             minArea = self.min_area,
                                             close_rad = self.blob_close_rad)
        except UnboundLocalError:
            ret = {'area':np.nan, 'filled_area':np.nan,
            'minRect':((np.nan,np.nan),(np.nan,np.nan),np.nan), 
                'rectPts':np.ones((4,2))*np.nan, 'mask':self.mask}
            return ret

        self.initial_mask = masks
        # get the bounding rectangle corresponding to the blob of interest
        contours, _ = cv2.findContours(masks.astype('uint8')*255, 
                                       cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        c=contours[0]
        minRect = cv2.minAreaRect(c)    
        pts=cv2.boxPoints(minRect)

        # get mask with filled holes
        rad = self.filled_rad
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(rad,rad))
        maskr = cv2.morphologyEx(self.mask,cv2.MORPH_CLOSE,kernel)


        if minRect[1][1]>minRect[1][0]:
            xangle = (90-minRect[2])/180*pi
        else:
            xangle = minRect[2]/180*pi

        dlen = self.scale_len - max(minRect[1])

        newrect = ((minRect[0][0]+dlen/2*np.cos(xangle),
                    minRect[0][1]-dlen/2*np.sin(xangle)),
                (minRect[1][0],self.scale_len),
                minRect[2])
        newrect=minRect

        # convert rect to mask and apply to original mask
        pts=cv2.boxPoints(newrect)
        rectMask = np.zeros(self.mask.shape,dtype='uint8')
        rectMask = cv2.fillPoly(rectMask,[pts.astype('i')],(255,255,255))
        self.rect_mask = rectMask

        mm = cv2.bitwise_and(rectMask,self.mask)
        md = cv2.bitwise_and(rectMask,maskr)

        self.final_mask = mm
        self.filled_mask = md

    def measure(self):
        self.ref_rect = self.new_rect
        # get the bounding rectangle corresponding to the blob of interest
        contours, _ = cv2.findContours(self.filled_mask.astype('uint8')*255, 
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = np.concatenate(contours)
        minRect = cv2.minAreaRect(c)    
        pts = cv2.boxPoints(minRect)

        self.area = np.sum(self.final_mask > 0)
        self.filled_area = np.sum(self.filled_mask > 0)
        self.new_rect = minRect
        self.rect_pts = pts

    def set_video_source(self, video_file, pos=0):
        self.video_file = video_file
        cap = cv2.VideoCapture(cv2.samples.findFile(video_file))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video length:", length)
        self.cap = cap
        self.n_frames = length
        self.init_references()
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def get_frame(self, pos=None):
        if pos is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, self.img = self.cap.read()
        if not ret:
            print("Error reading frame number ", pos)
        return self.img

    def image_results(self):
        mm = self.final_mask
        ims = self.img/np.max(self.img)/2 
        ims[:,:,0] += mm/np.max(mm)/2

        ims = (ims*255).astype('uint8')

        # draw reference rectangle
        pts=cv2.boxPoints(self.ref_rect)
        pts = pts.reshape((-1,1,2)).astype('int32')
        cv2.polylines(ims,[pts],True,(0,0,255))
        
        # draw new recatngle
        pts=cv2.boxPoints(self.new_rect)
        pts = pts.reshape((-1,1,2)).astype('int32')
        cv2.polylines(ims,[pts],True,(0,255,0))

        #cv2.imshow("image",ims)

        return ims.astype('uint8')
        
    def process(self, img=None):
        if img is None:
            self.get_frame()
        else:
            self.img=img
        self.color_convert()
        self.find_blob()
        self.measure()
        #return self.box

    def onChange(self, trackbarValue):
        self.get_frame(trackbarValue)
        self.process(self.img)
        cv2.imshow("image", self.image_results())

    def gui(self):
        wait_time = 33
        cv2.namedWindow('image')
        cv2.createTrackbar('frame no.', 'image', 0, self.n_frames, 
                          self.onChange)

        self.onChange(0)
        
        self.get_frame(0)
        k = cv2.waitKey()
        while self.cap.isOpened():
           if k == 27:
                break

        cv2.destroyAllWindows() 

    def run(self, n=None):
        if n is None:
            n = self.n_frames
        self.results=[]
        if self.progress:
            self.pbar = tqdm(total=self.n_frames)
        for ii in range(n):
            ret, image = self.cap.read()
            if not ret:
                self.pbar.write(f'Error reading frame {ii}. Skipping')
                continue
            msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.process(image)
            self.results.append({'time':msec/1000,
                                 'area':int(self.area),
                                 'filled_area':int(self.filled_area),
                                 'minRect':self.new_rect})
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

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="movie filename")
    parser.add_argument("-n", "--frame-number", type=int, default=0,
                        help="Process frame number")
    parser.add_argument("-N", "--n-frames", type=int, default=-1,
                        help="Process only n frames")
    parser.add_argument("-c", "--config", 
                        help="Config json file")
    parser.add_argument("-g", "--gui", action='store_true',
                        help="GUI (test mode)")
    parser.add_argument("-o", "--output", default="", 
                        help="output file (leave empty for same name as video)")

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parse()
    pos = args.frame_number
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

    processor = FrameProcessor(config_file=color_range_json_file, progress=True)
    processor.set_video_source(args.filename)
    if args.gui:
        processor.gui()
    else:
        try:
            if args.n_frames<1:
                processor.run()
            else:
                processor.run(n=args.n_frames)
        except AttributeError:
            import traceback
            traceback.print_exc()
        finally:
            processor.to_json(output)

