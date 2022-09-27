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
from imutils.object_detection import non_max_suppression
from .kalman_optical_tracker import OpticalTracker
import pytesseract


def length(x):
    return np.sqrt((x[2]-x[0])**2+(x[3]-x[1])**2)


def angle(x):
    return np.arctan(-(x[3]-x[1])/(x[2]-x[0]))/np.pi*180


def swap_rect_dims(rect):
    return np.array([rect[1], rect[0], rect[3], rect[2]])


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


def ticks_from_tick_borders(detected_borders, maxnotfound=10):
    all_borders, indices = complete_tick_borders(detected_borders, maxnotfound=maxnotfound)
    # first pair should be the 2 sides of one tick
    if all_borders[1]-all_borders[0] > all_borders[2]-all_borders[1]:
        all_borders = np.insert(all_borders,0,all_borders[0]-(all_borders[2]-all_borders[1]))
        indices = [None] + indices

    if (len(all_borders)//2)*2 != len(all_borders):
        all_borders = np.insert(all_borders,-1,all_borders[-1]+all_borders[-2]-all_borders[-3])
        indices += [None]
        
    ticks_measured = ((all_borders[::2]+all_borders[1::2])/2)
    pp=np.polyfit(np.arange(len(ticks_measured)),ticks_measured,2)
    ipred = np.arange(-5,len(ticks_measured)+5)
    ticks_reg = np.polyval(pp,ipred)
    return ticks_reg,ipred
    
    
    
def complete_tick_borders(detected_borders, maxnotfound=10):
    indices = [0,1,2]
    borders = [d for d in detected_borders[:3]]
    ii = 3
    
    nnotfound = 0

    while ii < len(detected_borders):
        x = detected_borders[ii]
        exp_int = borders[-2]-borders[-3]
        exp_next_int = (borders[-1]-borders[-2])
        tol = min(exp_int,exp_next_int)/2
        exp_pos = borders[-1]+exp_int
        xdiff = exp_pos - x
        if (xdiff) < -tol :
            # missing tick
            borders.append(exp_pos)
            indices.append(None)
            nnotfound+=1
            if nnotfound>maxnotfound:
                break
            continue
        else:
            nnotfound=0
            borders.append(x)
            indices.append(ii)
            ii+=1

    return np.asarray(borders), indices

def get_near_vert_segments(imgr, min_y=0, max_y=None, min_angle=45, break_tolerance=2, 
                        min_prominence=None, max_width=10, min_length=0):
    """
    Find near_vertical dark segments in (gray) image
    
    Optionally, restict search to lines min_y to max_y
    
    Arguments:
        * min_y,max_y: restrict line search to horizontal band
        * min_angle: minimum angle of segment (90=vertical)
        * break_tolerance: Number of lines where one line may be interrupted
        * min_prominence: Minimum contrast of lines
        * max_width: Maximum width of lines (for peak_search)
        * min_length: Minimum segment length
    """
    
    mindist = max(1,np.arctan(min_angle/180*np.pi)+1)
    
    if max_y is None:
        max_y = imgr.shape[0]
        
    seg_pts = []
    dy=1
    
    for ii,lno in enumerate(range(int(min_y),int(max_y),dy)):
        thisl=imgr[lno,:]
        if min_prominence is None:
            min_prom = np.diff(np.percentile(thisl,[5,50]))[0]/10
        else:
            min_prom = min_prominence
        #print(min_prom)
            
        pks,pkinfo = sig.find_peaks(-thisl,prominence=[min_prom,None],width=[None,max_width])
        
        # predict new position for this line
        xpred = []
        segidx = []
        for si,l in enumerate(seg_pts):
            if l[-1][1] >= lno-dy-break_tolerance:
                s = np.array(l)
                #print(s.shape)
                if len(l)>1:
                    poly = np.polyfit(s[:,1],s[:,0],1)
                    xpred.append(np.polyval(poly,lno))
                else:
                    xpred.append(s[0,0])
                segidx.append(si)
        xpred = np.array(xpred)
        for pk in pks:
            dists = np.abs(pk-xpred)
            if len(dists)>0:
                distmin=min(dists)
            else:
                distmin=len(thisl)

            if distmin<mindist:
                seg_pts[segidx[np.argmin(dists)]].append([pk,lno])
            else:
                seg_pts.append([[pk,lno]])

    #return seg_pts
    segments = []
    for seg in seg_pts:
        s = np.asarray(seg)
        if s.shape[0]>min_length:
            
            poly = np.polyfit(s[:,1],s[:,0],1)
            angle = np.arctan(-1/poly[0])/np.pi*180
            if angle>min_angle:
                #print(f'{angle:6.2f}, {poly[1]:5.1f}')
                
                ymax=(max(s[:,1]))
                ymin=(min(s[:,1]))
                xmax=np.polyval(poly,ymax)
                xmin=np.polyval(poly,ymin)
            
                segments.append({'y_st': ymin,
                                 'y_end': ymax,
                                 'x_st': xmin,
                                 'x_end': xmax,
                                 'angle': angle,
                                 'length': np.sqrt((xmax-xmin)**2+(ymax-ymin)**2)})
    return segments
        
        
def align_segments(segments,new_len=None):
    angles = np.array([x['angle'] for x in segments])
    medangle = np.median(angles)
    med_y = np.median(np.array([(x['y_st']+x['y_end'])/2 for x in segments]))
    newslope = -1/np.tan(medangle/180*np.pi) 

    newsegs = []
    for segment in segments:
        newseg = segment.copy()
        oldslope = -1/np.tan(segment['angle']/180*np.pi) 
        x_st = segment['x_st']
        x_end = segment['x_end']
        y_st = segment['y_st']
        y_end = segment['y_end']
        if new_len is None:
            halfl = (y_end-y_st)/2
            newseg['x_st'] = (x_st+x_end)/2 - halfl*newslope
            newseg['x_end'] = (x_st+x_end)/2 + halfl*newslope
        else:
            halfl = (new_len)/2
            med_x = x_st + (x_end-x_st)/(y_end-y_st)*(med_y-y_st)
            newseg['x_st'] = med_x - halfl*newslope
            newseg['x_end'] = med_x + halfl*newslope
            newseg['y_st'] = med_y - halfl
            newseg['y_end'] = med_y + halfl
            
        newseg['angle'] = medangle
        newsegs.append(newseg)
    return newsegs


class FrameProcessor(object):
    def __init__(self, color_range_file=None, config_file=None, 
                 video_file=None, anchor_pt=(0,0),
                 min_area=1000, filled_rad=19, scale_length=400, pos=0, blob_close_rad=9,
                 progress=False):
        if color_range_file is not None:
            self.read_color_range(color_range_file)
        self.anchor_pt=anchor_pt
        self.blob_close_rad=blob_close_rad
        self.rough_close_rad = 20
        self.min_area = min_area
        self.filled_rad = filled_rad
        self.scale_len = scale_length
        self.init_pos = pos
        self.crop_shape = None
        self.progress = progress
        # HSV boundaries for white scale
        self.lower_white = np.array([0, 0, 100])
        self.upper_white = np.array([179, 50, 255])
        self.nn_path = 'C:/Users/goios/Downloads/frozen_east_text_detection.pb'
        self.tesseract_dir = 'C:/Users/goios/Downloads'

        if video_file:
            self.set_video_source(video_file, pos=pos)
        if config_file is not None:
            self.read_config(config_file)

        print("[INFO] loading EAST text detector...")
        self.net = cv2.dnn.readNet(self.nn_path)
        self.layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        self.text_det_img_size = 320
        self.text_det_fract = 4
        self.text_to_find = ['25','30']
        self.text_boxes = {}
        self.text_boxes_orig = {}
        self.ref_cent_25 = (0,0)
        self.ref_string = self.text_to_find[0]
        self.tracker = OpticalTracker()

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
        masks, new_pt = find_nearest_blob(self.green_mask, self.anchor_pt,minArea=self.min_area)
        contours, _ = cv2.findContours(masks.astype('uint8')*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c=contours[0]
        minRect = cv2.minAreaRect(c)
        self.ref_rect = minRect
        self.new_rect = minRect

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

    def find_green_strip(self):
        # find blob nearest to point
        try:
            masks,new_pt = find_nearest_blob(self.green_mask, self.anchor_pt, 
                                             minArea = self.min_area,
                                             close_rad=self.blob_close_rad)
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
        maskr = cv2.morphologyEx(self.green_mask,cv2.MORPH_CLOSE,kernel)

        if minRect[1][1]>minRect[1][0]:
            rot_angle = minRect[2]+270
            xangle = (90-minRect[2])/180*pi
        else:
            rot_angle = minRect[2]+180
            xangle = minRect[2]/180*pi

        dlen = self.scale_len - max(minRect[1])

        newrect = ((minRect[0][0]+dlen/2*np.cos(xangle),
                    minRect[0][1]-dlen/2*np.sin(xangle)),
                (minRect[1][0],self.scale_len),
                minRect[2])
        newrect=minRect

        # convert rect to mask and apply to original mask
        pts=cv2.boxPoints(newrect)
        rectMask = np.zeros(self.green_mask.shape,dtype='uint8')
        rectMask = cv2.fillPoly(rectMask,[pts.astype('i')],(255,255,255))
        self.rect_mask = rectMask

        mm = cv2.bitwise_and(rectMask,self.green_mask)
        md = cv2.bitwise_and(rectMask,maskr)

        self.final_mask = mm
        self.filled_mask = md
        self.green_rect = newrect
        self.green_rect_pts = pts
        self.green_strip_angle = rot_angle

    def find_scale(self):
        green_cent = self.green_rect[0]
        white_mask = cv2.inRange(self.hsv, self.lower_white, self.upper_white)

        try:
            masks,new_pt = find_nearest_blob(white_mask, green_cent, 
                                             minArea = self.min_area,
                                             close_rad=self.blob_close_rad)
        except UnboundLocalError:
            ret = {'area':np.nan, 'filled_area':np.nan,
            'minRect':((np.nan,np.nan),(np.nan,np.nan),np.nan), 
                'rectPts':np.ones((4,2))*np.nan, 'mask':self.mask}
            return ret

        # combine green strip and scale for main ROI
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.rough_close_rad,
                                                          self.rough_close_rad))

        mask2 = cv2.bitwise_or(masks.astype('uint8')*255,self.initial_mask.astype('uint8')*255)
        mask2 = cv2.morphologyEx(mask2,
                                 cv2.MORPH_CLOSE, el, iterations=1)
        self.roi_mask = mask2

    def get_rotated_roi(self, cent=None):

        # get the bounding rectangle corresponding to the blob of interest
        contours, _ = cv2.findContours(self.roi_mask, 
                                    cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        c=contours[0]
        minRect = cv2.minAreaRect(c)    
        pts=cv2.boxPoints(minRect)

        margFrac = .5

        x,y=np.where(self.roi_mask)

        w = np.max(x)-np.min(x)
        h = np.max(y)-np.min(y)

        marg = max(int(margFrac*w), int(margFrac*h))

        if self.crop_shape is None:
            crop_w  = np.max(x) - np.min(x) + 2*marg
            crop_h  = np.max(y) - np.min(y) + 2*marg
            self.crop_shape = (crop_w, crop_h)

            
        crop_w = self.crop_shape[0]
        crop_h = self.crop_shape[1]
        if cent is None:
            cent = [crop_w/2, crop_h/2]
            refrect = self.text_boxes_orig[self.ref_string]
        else:
            refrect = np.array([np.min(x), np.min(y), np.max(x), np.max(y)])
        refcent = (refrect[0]+refrect[2])/2, (refrect[1]+refrect[3])/2
        cropul = (int(refcent[0]-crop_w/2), int(refcent[1]-crop_h/2)) 
        croplr = (int(refcent[0]+crop_w/2), int(refcent[1]+crop_h/2)) 
        self.cropullr = np.concatenate((cropul,croplr))
        cropullr = self.cropullr
        srccrop = [cropul[0], cropul[1], min(croplr[0], self.img.shape[1]),
                                         min(croplr[1], self.img.shape[0])]

        # Cropped ROIs
        imcr = np.zeros((crop_h, crop_w, self.img.shape[2]),dtype='uint8')
#        imcr[0:(srccrop[2]-srccrop[0]), 0:(srccrop[3]-srccrop[1]),:] = (self.img*(self.roi_mask[:,:,np.newaxis]//255))[srccrop[0]:srccrop[2],
#                                                                           srccrop[1]:srccrop[3],:].astype('uint8')
        imcr[0:(srccrop[3]-srccrop[1]), 0:(srccrop[2]-srccrop[0]),:] = (self.img*(self.roi_mask[:,:,np.newaxis]//255))[srccrop[1]:srccrop[3],
                                                                           srccrop[0]:srccrop[2],:].astype('uint8')
        #imcg = np.zeros((crop_w, crop_h, self.img.shape[2]))
        #imcg[0:(srccrop[2]-srccrop[0]), 0:(srccrop[3]-srccrop[1]),:] = (self.img*(self.green_mask[:,:,np.newaxis]//255))[srccrop[0]:srccrop[2],
        #                                                                   srccrop[1]:srccrop[3],:].astype('uint8')
        #imcg = (self.img*(self.green_mask[:,:,np.newaxis].astype('uint8')))[cropullr[0]:cropullr[2],
        #                                                                                cropullr[1]:cropullr[3],:]

        # rotated ROIs
        newW=newH=self.text_det_img_size
        marg=10

        #cent = [minRect[0][0]-np.min(y)+w//2,minRect[0][1]-np.min(x)+h//2]
        rotM = cv2.getRotationMatrix2D(cent,self.green_strip_angle,1)
        rotMi = cv2.invertAffineTransform(rotM)
        self.rotMi = rotMi

        imrot = cv2.warpAffine(imcr, rotM, [crop_w, crop_h])#np.asarray(minRect[1]).astype(int)*2)
        #grrot = cv2.warpAffine(imcg, rotM, [newW, newH])
        newbbcnr = [int(cent[1]-minRect[1][1]/2)-marg,int(cent[0]-minRect[1][0]/2)-marg,
                    int(cent[1]+minRect[1][1]/2)+marg,int(cent[0]+minRect[1][0]/2)+marg,]
        rotpts=np.asarray([rotM.dot(list(pt) +[1,]) 
                           for pt in self.green_rect_pts-np.asarray([cropullr[0],cropullr[1],])])
        self.tick_y_max = np.min(rotpts[:,1])

        self.roi_img = imrot
        self.roi_img_prev = imrot.copy()
        cv2.polylines(self.roi_img_prev, [rotpts.astype(int)],1,[0,255,0], 2)

        
    def decode_predictions(self, scores, geometry, min_confidence=.5):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability,
                # ignore it
                if scoresData[x] < min_confidence:
                    continue
                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * self.text_det_fract, y * self.text_det_fract)
                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)
         
    def find_numbers(self):
        W = H = self.text_det_img_size
        text_crop_u = (self.crop_shape[0] - self.text_det_img_size)//2
        text_crop_l = (self.crop_shape[1] - self.text_det_img_size)//2
        roi_crop = self.roi_img[text_crop_l:text_crop_l+self.text_det_img_size,
                                text_crop_u:text_crop_u+self.text_det_img_size]
        print(self.roi_img.shape,roi_crop.shape,text_crop_l,text_crop_u)
        blob = cv2.dnn.blobFromImage(roi_crop, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layerNames)

        # likely position of text
        rects, confidences = self.decode_predictions(scores, geometry)
        # reduce to single connected boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        padding = 0.0
        orig = roi_crop#self.roi_img
        origW, origH = orig.shape[:2]
        rW,rH =1,1

        # convert boxes to original image size and do OCR
        # initialize the list of results
        results = []
        # loop over the bounding boxes
        self.text_boxes_orig={}
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * padding)
            dY = int((endY - startY) * padding)
            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX + text_crop_l)
            startY = max(0, startY - dY + text_crop_u)
            endX = min(origW, endX + (dX * 2)+ text_crop_l)
            endY = min(origH, endY + (dY * 2)+ text_crop_u)
            # extract the actual padded ROI
            roi = orig[startY:endY, startX:endX]
            
            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 1, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text
            config = (f"-l eng --oem 1 --psm 7 --tessdata-dir {self.tesseract_dir}")
            text = pytesseract.image_to_string(roi, config=config)
            # add the bounding box coordinates and OCR'd text to the list
            # of results
            results.append(((startX, startY, endX, endY), text))
            self.tick_y_min = 0
            for r, text in results:
                cv2.rectangle(self.roi_img_prev, r[0:2], r[2:4], (0, 255, 0), 2)
                cv2.putText(self.roi_img_prev, text, (r[0], r[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                for tf in self.text_to_find:
                    if text.find(tf)>-1:
                        #print(f'Found text {tf}')
                        bbox=np.asarray(r)
                        self.tick_y_min = max(self.tick_y_min, np.max(bbox[[1,3]]))
                        self.text_boxes[tf] = bbox
                        (bbox[0]+bbox[2])//2,(bbox[1]+bbox[3])//2
                        w = bbox[2]-bbox[0]
                        h = bbox[3]-bbox[1]
                        r = max(w,h)*.7

                        cent=(self.rotMi.dot(np.array([((bbox[0]+bbox[2])//2,
                                                (bbox[1]+bbox[3])//2,1,)]).T).T+np.array([self.cropullr[0],self.cropullr[1]]))[0]
                        br = np.array([int(cent[0]-r),int(cent[1]-r),int(cent[0]+r),int(cent[1]+r)])
                        self.text_boxes_orig[tf] = br
                    else:
                        bbox = None

                    

    def detect_ticks_sd(self):
        lsd = cv2.createLineSegmentDetector(2)
        imgr = cv2.cvtColor(self.roi_img, cv2.COLOR_BGR2GRAY)

        lines, *info = lsd.detect(imgr)
        miny = self.tick_y_min
        maxy = self.tick_y_max
        print(miny,maxy)

        # reference label
        ref_cent = self.ref_cent_25
        ref_dist = 25
        for lab, bbox in self.text_boxes.items():
            try:
                ref_cent = (bbox[0]+bbox[2])/2,bbox[3]
                ref_dist = float(lab)
                if ref_dist == 25:
                    self.ref_cent_25 = ref_cent
                break
            except TypeError:
                pass
        
        allticks = sorted([(ii,
                            min(l[0,0],l[0,2]), 
                            max(l[0,0],l[0,2]),
                            min(np.sqrt(np.sum((l[0,0:2]-ref_cent)**2)),
                            np.sqrt(np.sum((l[0,2:4]-ref_cent)**2))),
                            angle(l[0]),length(l[0]))
                           for ii,l in enumerate(lines)
                           if l[0,1]>miny and l[0,1]<maxy and 
                              l[0,3]>miny and l[0,3]<maxy and 
                              (angle(l[0])>70 or angle(l[0])<-90) 
                          ], key=lambda x:x[1]) 
        allticks = np.asarray(allticks)
        tick_borders_x = (allticks)[:,1]
        lsd.drawSegments(self.roi_img_prev,np.array([lines[int(l[0])] for l in allticks]))

        ticks_x, ticks_idx = ticks_from_tick_borders(tick_borders_x)

        l=60
        x0=np.mean([max(lines[int(l[0])][0,[1,3]]) for l in allticks])
        ang = np.pi-np.median([l[4] for l in allticks])/180*np.pi

        for xm in ticks_x:
            cv2.line(self.roi_img_prev,(int(xm),int(x0)),(int(xm+l*cos(ang)),int(x0+l*sin(ang))),(0,0,255),1)

    def detect_ticks(self):
        
        imgr = cv2.cvtColor(self.roi_img, cv2.COLOR_BGR2GRAY)
        miny = self.tick_y_min
        maxy = self.tick_y_max
        segments = get_near_vert_segments(imgr,min_length=5,min_angle=75,min_y=miny,max_y=maxy)
        for seg in sorted(segments,key=lambda x:x['x_end']):
            cv2.line(self.roi_img_prev,(int(seg['x_st']),int(seg['y_st'])),(int(seg['x_end']),int(seg['y_end'])),(0,0,255))


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

    def set_video_source(self, video_file, from_time=0.0, to_time=None):
        self.video_file = video_file
        cap = cv2.VideoCapture(cv2.samples.findFile(video_file))
        ret, img = cap.read()
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
        self.init_references()
        self.process_full(img)

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
        cv2.polylines(ims,[pts],True,(0,191,0))
        for lab, brf in self.text_boxes_orig.items():
            br = np.round(brf).astype('int')
            cv2.rectangle(ims,br[0:2],br[2:4],(0,255,0),2)

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

    def displace_number_box(self):
        new_rect = self.tracker.update(self.img)
        self.text_boxes_orig[self.ref_string] = (new_rect)
        
        
    def process_full(self, img=None):
        if img is None:
            self.get_frame()
        else:
            self.img=img
        self.color_convert()
        self.find_green_strip()
        self.find_scale()
        self.get_rotated_roi(cent=self.roi_cent)
        self.find_numbers()
        self.detect_ticks()
        self.measure()
        print(self.text_boxes_orig)
        try:
            self.tracker.set_template(img, (self.text_boxes_orig[self.ref_string]))
            self.ref_string = '25'
        except IndexError:
            pass
        #return self.box

    def process(self, img):
        if img is None:
            self.get_frame()
        else:
            self.img=img
        self.color_convert()
        self.find_green_strip()

        self.displace_number_box()
        self.get_rotated_roi()
        self.detect_ticks()
        self.measure()
        #return self.box
        

    def onChange(self, trackbarValue):
        self.get_frame(trackbarValue)
        if self.tracker.template is None:
            self.process_full(self.img)
        else:
            self.process(self.img)
        cv2.imshow("image", self.image_results())
        cv2.imshow("rotated ROI", self.roi_results())

    def gui(self):
        wait_time = 33
        cv2.namedWindow('image')
        cv2.createTrackbar('frame no.', 'image', 0, self.n_frames, 
                          self.onChange)

        cv2.namedWindow('rotated ROI')
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

