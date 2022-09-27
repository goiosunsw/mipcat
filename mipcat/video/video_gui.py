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
import PySimpleGUI as sg
import cv2
import numpy as np
from mouthpiece_color_based_process import FrameProcessor

def process(image, fp):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, lower, upper)
    # output = cv2.bitwise_and(image,image, mask= mask)
    fp.img = image
    fp.process(image)
    return fp.image_results()


def main():

    vidfile = sys.argv[1]
    conffile = os.path.splitext(vidfile)[0]+'_conf.json'


    cap = cv2.VideoCapture(vidfile)
    video_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pos = 0
    ret, img = cap.read()
    w,h,*_ = img.shape
    
    try:
        with open(conffile,'r') as f:
            confjs = json.load(f)
            print(confjs)
    except IOError:
        confjs = {'hue':  [0, 179],
            'saturation': [0, 255],
            'value': [0, 255],
            'close_rad':1,
            'anchor':(w//2,h//2)}


    # define the window layout
    layout = [
      [sg.Text('OpenCV Demo', size=(60, 1), justification='center')],
      [sg.Graph((600,450),(0,450), (600,0), key='-GRAPH-', enable_events=True, drag_submits=False)],
      [sg.Check('process', default=False, size=(10, 1), key='-PROCESS-', enable_events=True),
       sg.Slider((0, video_len), pos, 1, orientation='h', size=(40, 15), key='-POS-', enable_events=True),
       sg.Text('0.0', size=(10, 1), justification='left', key='-SEC-')],
      [sg.Text('Hue', size=(10, 1), justification='right'),
       sg.Slider((0,179),confjs['hue'][0], 1, orientation='h', size=(30, 15), key='-HUE_LO-', enable_events=True),
       sg.Slider((0,179),confjs['hue'][1], 1, orientation='h', size=(30, 15), key='-HUE_HI-', enable_events=True)],
      [sg.Text('Saturation', size=(10, 1), justification='right'),
       sg.Slider((0,255),confjs['saturation'][0], 1, orientation='h', size=(30, 15), key='-SAT_LO-', enable_events=True),
       sg.Slider((0,255),confjs['saturation'][1], 1, orientation='h', size=(30, 15), key='-SAT_HI-', enable_events=True)],
      [sg.Text('Value', size=(10, 1), justification='right'),
       sg.Slider((0,255),confjs['value'][0], 1, orientation='h', size=(30, 15), key='-VAL_LO-', enable_events=True),
       sg.Slider((0,255),confjs['value'][1], 1, orientation='h', size=(30, 15), key='-VAL_HI-', enable_events=True)],
      [sg.Text('Blob close radius', size=(10,1),justification='right'),
       sg.Slider((1,30),confjs['close_rad'],1, orientation='h', size=(30,15), key='-BRAD-',enable_events=True)],
      [sg.Button('Validate'), sg.Button('Delete')]
    ]

    window = sg.Window('Video browser', layout)
    graph_elem = window['-GRAPH-']      # type: sg.Graph
    
    event, values = window.read(timeout=20)

    imgbytes=cv2.imencode('.ppm', img)[1].tobytes()       # on some ports, will need to change to png
    a_id = graph_elem.draw_image(data=imgbytes, location=(0,0))    # draw new image

    fp = FrameProcessor()
    fp.anchor_pt = confjs['anchor']
    points=[confjs['anchor']]
    p_id = graph_elem.draw_circle(points[-1], 5, fill_color='red', line_color='red')
    ptids=[(p_id)]

    jsc = {'hue': {'min': values['-HUE_LO-'],
                    'max': values['-HUE_HI-']},
            'saturation': {'min': values['-SAT_LO-'],
                            'max': values['-SAT_HI-']},
            'value': {'min': values['-VAL_LO-'],
                        'max': values['-VAL_HI-']}}
    fp.set_color_range(jsc)
    fp.blob_close_rad = int(values['-BRAD-'])

    cap.set(cv2.CAP_PROP_POS_FRAMES,values['-POS-'])
    fp.init_references(img)

    while True:
        event, values = window.read(timeout=20)
        if event in ('Exit', None):
            break


        if event in ['-POS-', '-PROCESS-', '-HUE_LO-', '-HUE_HI-', '-SAT_LO-', '-SAT_HI-',
                     '-VAL_LO-', '-VAL_HI-','-BRAD-']:
            cap.set(cv2.CAP_PROP_POS_FRAMES,values['-POS-'])
            ret, img = cap.read()
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            sec=msec/1000.
            window['-SEC-'].update(f'{sec:.3f}')
            #print(msec)
            if values['-PROCESS-']:
                jsc = {'hue': {'min': values['-HUE_LO-'],
                               'max': values['-HUE_HI-']},
                       'saturation': {'min': values['-SAT_LO-'],
                                      'max': values['-SAT_HI-']},
                       'value': {'min': values['-VAL_LO-'],
                                 'max': values['-VAL_HI-']}}
                fp.set_color_range(jsc)
                fp.blob_close_rad = int(values['-BRAD-'])
                img = process(img,fp)
            imgbytes=cv2.imencode('.ppm', img)[1].tobytes()       # on some ports, will need to change to png
            if a_id:
                graph_elem.delete_figure(a_id)             # delete previous image
            a_id = graph_elem.draw_image(data=imgbytes, location=(0,0))    # draw new image
            graph_elem.send_figure_to_back(a_id)            # move image to the "bottom" of all other drawings
            last_pos = values['-POS-']

        if event == '-GRAPH-':
            fp.anchor_pt = values['-GRAPH-']
            points.append(values['-GRAPH-'])
            p_id = graph_elem.draw_circle(points[-1], 5, fill_color='red', line_color='red')
            ptids.append(p_id)
        
        if event == 'Validate':
            dd = {'anchor':points[-1],
                  'hue':[values['-HUE_LO-'], values['-HUE_HI-']],
                  'saturation':[values['-SAT_LO-'], values['-SAT_HI-']],
                  'value':[values['-VAL_LO-'], values['-VAL_HI-']],
                  'close_rad':values['-BRAD-']
                  }
            outfile = os.path.splitext(vidfile)[0]+'_conf.json'
            with open(outfile,'w') as f:
                json.dump(dd, f)
                print('Config written to {}'.format(outfile))

        if event == 'Delete':
            points = []
            for p_id in ptids:
                graph_elem.delete_figure(p_id)





    window.close()


if __name__ == '__main__':
    main()