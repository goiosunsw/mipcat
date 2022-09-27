#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    GUI for adjusting parameters of the mouthpiece color based processor
    mouthpiece_color_based_process.py

    This file is meant to be used with the 'opencv-contrib' environment

    Once parameters are adjusted press "Validate" to create a json file that
    can be used directly with the script
"""

import os
import sys
import json
import argparse
import PySimpleGUI as sg
import cv2
import numpy as np


def ensure_max_dim(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if h<height and w<width:
        return image

    try:
        img = np.zeros((height,width,image.shape[2]),dtype=image.dtype)
    except IndexError:
        
        img = np.zeros((height,width),dtype=image.dtype)

    rh = 1
    rw = 1
    if width is None and height is None:
        return image
    if height is not None:
        rh = height / float(h)
        #dim = (int(w * r), height)
    if width is not None:
        rw = width / float(w)
        #dim = (width, int(h * r))
    r = min(rh,rw)
    dim = (int(w*r),int(h*r)) 
    print(dim)
    i2 = cv2.resize(image, dim, interpolation=inter)
    img[:i2.shape[0],:i2.shape[1],:] = i2

    return i2


class MarkerInfo(object):
    def __init__(self, markerfile):
        with open(markerfile,'r') as f:
            self.markers = json.load(f)
        self.times = np.array([x['msec']/1000 for x in self.markers])
        
    def get_markers(self,time):
        # FIXME: this won't work if the markers have only been extracted for part of the video
        try:
            idx = np.flatnonzero(self.times>time)[0]
        except IndexError:
            idx = -1
        return self.markers[idx]
    
    def draw_markers(self, img, time):
        markers = self.get_markers(time)
        for mid, mrk in markers.items():
            if mid == 'msec':
                continue
            try: 
                cnr = mrk['corners']
                img = cv2.polylines(img,[np.int32(cnr)],True,255,3, cv2.LINE_AA)
                x, y, w, h = mrk['bbox']
                cv2.putText(img,f"{mid}", (x+w+10,y+h+10), 0, 1.0, (0,255,0))
            except KeyError:
                x, y, w, h = mrk['bbox']
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.putText(img,f"{mid}", (x+w+10,y+h+10), 0, 1.0, (0,255,0))
        return img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="movie filename", default="")
    parser.add_argument("-s", "--start_sec", type=float, default=0.0,
                        help="start time in seconds")
    parser.add_argument("-m", "--marker-file", default="", 
                        help="marker file (leave empty for same name as video)")

    return parser.parse_args()

def main():

    args = parse_args()
    vidfile = args.filename
    if len(args.marker_file)<1:
        markerfile = os.path.splitext(vidfile)[0]+'_markers.json'
    else:
        markerfile = args.marker_file
    markers = MarkerInfo(markerfile)


    cap = cv2.VideoCapture(vidfile)
    video_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pos = 0
    ret, img = cap.read()
    w,h,*_ = img.shape
    
    gw = 800
    gh = 600    

    # define the window layout
    layout = [
      [sg.Text('OpenCV Demo', size=(60, 1), justification='center')],
      [sg.Graph((gw,gh),(0,gw), (gh,0), key='-GRAPH-', enable_events=True, drag_submits=False)],
      [sg.Check('process', default=False, size=(10, 1), key='-PROCESS-', enable_events=True),
       sg.Slider((0, video_len), pos, 1, orientation='h', size=(40, 15), key='-POS-', enable_events=True),
       sg.Text('0.0',justification='left',size=(10,1), key='-SEC-')],
      [sg.Text('Rotation',size=(10,1),justification='right'),
       sg.Combo(['0','90','180','270'], default_value='0', key='-ROT-')]
    ]

    window = sg.Window('Video browser', layout)
    graph_elem = window['-GRAPH-']      # type: sg.Graph
    
    event, values = window.read(timeout=20)
    img = cv2.resize(img,None,fx=gw/img.shape[1],fy=gh/img.shape[0]) 

    imgbytes=cv2.imencode('.ppm', img)[1].tobytes()       # on some ports, will need to change to png
    a_id = graph_elem.draw_image(data=imgbytes, location=(0,0))    # draw new image


    cap.set(cv2.CAP_PROP_POS_FRAMES,values['-POS-'])

    while True:
        event, values = window.read(timeout=20)
        if event in ('Exit', None):
            break

        if event in ('-POS-', '-ROT-'):
            cap.set(cv2.CAP_PROP_POS_FRAMES,values['-POS-'])
            ret, img = cap.read()
            if img is None:
                continue
            
            sec = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
            window['-SEC-'].update(f'{sec:.03f}')

            time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            markers.draw_markers(img, time_msec/1000)
           
            if values['-ROT-'] == '90':
                rot = cv2.ROTATE_90_CLOCKWISE
                img = cv2.rotate(img, rot)
            elif values['-ROT-'] == '180':
                rot = cv2.ROTATE_180
                img = cv2.rotate(img, rot)
            elif values['-ROT-'] == '270':
                rot = cv2.ROTATE_90_COUNTERCLOCKWISE
                img = cv2.rotate(img, rot)
            #img = cv2.resize(img,None,fx=gw/img.shape[1],fy=gh/img.shape[0]) 
            img = ensure_max_dim(img, width=gw, height=gh)
            imgbytes=cv2.imencode('.ppm', img)[1].tobytes()       # on some ports, will need to change to png
            if a_id:
                graph_elem.delete_figure(a_id)             # delete previous image
            a_id = graph_elem.draw_image(data=imgbytes, location=(0,0))    # draw new image
            graph_elem.send_figure_to_back(a_id)            # move image to the "bottom" of all other drawings
            last_pos = values['-POS-']

    window.close()


if __name__ == '__main__':
    main()