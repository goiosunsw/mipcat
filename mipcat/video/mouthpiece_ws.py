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

def nothing(x):
    pass

def find_nearest_blob(labels, anchor_pt, close_rad=9, minArea=1000, mindist = 1e12):

    idx = np.unique(labels)
    minlab = 0
    minpt = None
    
    for ii in idx:
        if ii<1:
            continue
        area = np.sum(labels==ii)
        if area>minArea:
            xx,yy = np.where(labels==ii)
            dists = ((yy-anchor_pt[0])**2+(xx-anchor_pt[1])**2)
            thisidx = np.argmin(dists)
            if dists[thisidx]<mindist:
                mindist = dists[thisidx]
                minpt = (yy[thisidx],xx[thisidx])
                minlab = ii
    
    return (labels==minlab), minpt

class FrameProcessor(object):
    def __init__(self, color_range_file=None, video_file=None, anchor_pt=(0,0),
                 min_area=1000, filled_rad=19, scale_length=400, pos=0):
        if color_range_file is not None:
            self.read_color_range(color_range_file)
        self.anchor_pt=anchor_pt
        self.min_area = min_area
        self.filled_rad = filled_rad
        self.scale_len = scale_length
        self.init_pos = pos
        self.couple_trackbars=False
        if video_file:
            self.set_video_source(video_file, pos=pos)

    def read_color_range(self, jsfile):
        with open(jsfile,'r') as f:
            jsc = json.load(f)
        self.set_color_range(jsc)

    def set_color_range(self, jsc):
        self._lower_color = np.array([jsc[x]['min'] for x in ['hue','saturation','value']])
        self._upper_color = np.array([jsc[x]['max'] for x in ['hue','saturation','value']])

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
        hsv = cv2.cvtColor(self.img,cv2.COLOR_BGR2HSV)
        # find regions of appropriate color
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        self.mask = mask

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
        if self.show_mask:
            ims = np.uint8(self.mask)
        else:
            mm = self.final_mask
            ims = self.img/np.max(self.img)/2 
            ims[:,:,0] += mm/np.max(mm)/2

            ims = (ims*255).astype('uint8')
        cv2.circle(ims, self.anchor_pt, 3, [0,0,255])

        return ims.astype('uint8')

    def morphs(self):
        rad=5
        sure_fg = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(rad,rad)), iterations=3)
        sure_bg = cv2.morphologyEx(self.mask, cv2.MORPH_DILATE, 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(rad,rad)), iterations=3)
        unknown = cv2.subtract(sure_bg,sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        wsh = cv2.watershed(self.img, markers)
        markers[markers==1]=0
        self.filled_mask,_ = find_nearest_blob(markers, self.anchor_pt)
        self.final_mask = self.filled_mask
        
    def process(self, img=None):
        if img is None:
            self.get_frame()
        else:
            self.img=img
        self.color_convert()
        self.morphs()
        self.measure()
        #return self.box

    def onChange(self, trackbarValue):
        self.get_frame(trackbarValue)
        self.process(self.img)
        cv2.imshow("image", self.image_results())

    def mouse_click(self, event, x, y, 
                flags, param):
      
        # to check if left mouse 
        # button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            self.anchor_pt = (x,y)
          
    @property
    def upper_color(self):
        return self._upper_color

    @property
    def lower_color(self):
        return self._lower_color

    @property
    def show_mask(self):
        return cv2.getTrackbarPos('Mask', 'controls')
    
    @property
    def frame_no(self):
        return cv2.getTrackbarPos('frame no.', 'image')

    def redraw(self, val):
        # get current positions of all trackbars
        if self.couple_trackbars:
            self._lower_color = np.array([cv2.getTrackbarPos('HMin','controls'),
                                cv2.getTrackbarPos('SMin','controls'),
                                cv2.getTrackbarPos('VMin','controls')])

            self._upper_color = np.array([cv2.getTrackbarPos('HMax','controls'),
                                cv2.getTrackbarPos('SMax','controls'),
                                cv2.getTrackbarPos('VMax','controls')])

        self.onChange(self.frame_no)
    
    def gui(self):
        wait_time = 33
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.mouse_click)

        # Create control window
        #cv2.namedWindow('image')
        cv2.namedWindow('controls', flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)

        # create trackbars for color change
        cv2.createTrackbar('HMin','controls',0,179,self.redraw) # Hue is from 0-179 for Opencv
        cv2.createTrackbar('SMin','controls',0,255,self.redraw)
        cv2.createTrackbar('VMin','controls',0,255,self.redraw)
        cv2.createTrackbar('HMax','controls',0,179,self.redraw)
        cv2.createTrackbar('SMax','controls',0,255,self.redraw)
        cv2.createTrackbar('VMax','controls',0,255,self.redraw)
        cv2.createTrackbar('Process','controls',0,1,self.redraw)
        cv2.createTrackbar('Mask','controls',0,1,self.redraw)

        # Set default value for MAX HSV trackbars.
        cv2.setTrackbarPos('HMax', 'controls', self._upper_color[0])
        cv2.setTrackbarPos('SMax', 'controls', self._upper_color[1])
        cv2.setTrackbarPos('VMax', 'controls', self._upper_color[2])
        cv2.setTrackbarPos('HMin', 'controls', self._lower_color[0])
        cv2.setTrackbarPos('SMin', 'controls', self._lower_color[1])
        cv2.setTrackbarPos('VMin', 'controls', self._lower_color[2])
        cv2.setTrackbarPos('Process', 'controls', 1)

        cv2.createTrackbar('frame no.', 'image', 0, self.n_frames, 
                          self.onChange)

        self.onChange(0)
        self.couple_trackbars = True
        
        self.get_frame(0)
        k = cv2.waitKey()
        while self.cap.isOpened():
           if k == 27:
                break

        cv2.destroyAllWindows() 


def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="movie filename")
    parser.add_argument("-n", "--frame-number", type=int, default=0,
                        help="Process frame number")
    parser.add_argument("-c", "--color-range", 
                        help="Color range json file")
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

    if args.color_range:
        color_range_json_file = args.color_range
    else:
        color_range_json_file = basename + '_colormask.json'

    if len(args.output)==0:
        output = basename+'_markers.json'
    else:
        output = args.output 

    processor = FrameProcessor(color_range_file=color_range_json_file)
    processor.set_video_source(args.filename)
    if args.gui:
        processor.gui()
    else:
        try:
            processor.run()
        except AttributeError:
            import traceback
            traceback.print_exc()
        finally:
            processor.to_json(output)

