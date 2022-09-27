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
from mouthpiece_process import FrameProcessor
from template_trackers import template_track_rot, rotate_image


SLIDER_HEIGHT=15


    

def show_image(img, container, a_id=None, t_id=None, l_id=None):
    for s_id in [ a_id, t_id, l_id]:
        if s_id is not None:
            container.delete_figure(s_id)
    imgbytes=cv2.imencode('.ppm', img)[1].tobytes()       # on some ports, will need to change to png
    a_id = container.draw_image(data=imgbytes, location=(0,0))    # draw new image
    container.send_figure_to_back(a_id)
    return a_id

def rect_angle(rot_rect):
    """Convert rotated rect angle (up to 90 deg) to 180 deg angle
       by using long side information

    Args:
        rot_rect (openCV rotatedRect): _description_
    """
    
    if rot_rect[1][1]>rot_rect[1][0]:
        xangle = (90+rot_rect[2])
    else:
        xangle = rot_rect[2]

    return xangle
    
def reset_frame_proc(fp):
    
    rect = fp.new_rect
    template_time = confjs['pos']
    cap.set(cv2.CAP_PROP_POS_MSEC,template_time*1000)
    ret, img = cap.read()

    start_point, end_point = rect
    window["-INFO-"].update(value=f"grabbed rectangle from {start_point} to {end_point}")
    #visual_rect = prior_rect
    ref_pos = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
    dragging = False
    prior_rect = None
    template = img[min(start_point[1],end_point[1]):max(start_point[1],end_point[1]),
                    min(start_point[0],end_point[0]):max(start_point[0],end_point[0])]
    anchor = ((start_point[0]+end_point[0])/2,
                    (start_point[1]+end_point[1])/2)
    start_point, end_point = None, None  # enable grabbing a new rect
    fp.process(img)
    original_angle = rect_angle(fp.new_rect)

def main():
    img=None

    vidfile = sys.argv[1]
    conffile = os.path.splitext(vidfile)[0]+'_conf.json'


    cap = cv2.VideoCapture(vidfile)
    video_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fr_per_sec = cap.get(cv2.CAP_PROP_FPS)
    video_dur = video_len/fr_per_sec
    
    pos = 0.0
    ref_pos = pos
    original_angle = 0.0
    ret, img = cap.read()
    w,h,*_ = img.shape
    template=None
    t_id = None
    l_id = None
    line_len = 100
    templ_cent = (0,0)
    line_angle = 0
    line_end = (0,0)
    angle = 0
    val = 0
    
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

    pos_slider = sg.Slider((0, video_dur), pos, .1, orientation='h', size=(40, 15), key='-POS-', enable_events=True)

    # define the window layout
    left_col = [
      [sg.Text('Mouthpiece video configurator', size=(60, 1), justification='center')],
       [sg.Graph((600,450),(0,450), (600,0), key='-GRAPH-', enable_events=True, drag_submits=True)],
        [sg.Check('process', default=False, size=(10, 1), key='-PROCESS-', enable_events=True),
        pos_slider,
        sg.Text('0.0', size=(15, 1), justification='left', key='-SEC-')],
      [sg.Button('Save'), sg.Button('Back to ref'), sg.Button('Print'),sg.Button('Template')],
      [sg.Text(key='-INFO-', size=(60, 1))]
    ]
    right_col = [
        [sg.Text('Hue', size=(SLIDER_HEIGHT, 1), justification='right'),
         sg.Column([
          [sg.Slider((0,179),confjs['hue'][0], 1, orientation='h', size=(30, SLIDER_HEIGHT), key='-HUE_LO-', enable_events=True)],
          [sg.Slider((0,179),confjs['hue'][1], 1, orientation='h', size=(30, SLIDER_HEIGHT), key='-HUE_HI-', enable_events=True)]])],
        [sg.Text('Saturation', size=(SLIDER_HEIGHT, 1), justification='right'),
         sg.Column([
          [sg.Slider((0,255),confjs['saturation'][0], 1, orientation='h', size=(30, SLIDER_HEIGHT), key='-SAT_LO-', enable_events=True)],
          [sg.Slider((0,255),confjs['saturation'][1], 1, orientation='h', size=(30, SLIDER_HEIGHT), key='-SAT_HI-', enable_events=True)]])],
        [sg.Text('Value', size=(SLIDER_HEIGHT, 1), justification='right'),
         sg.Column([
          [sg.Slider((0,255),confjs['value'][0], 1, orientation='h', size=(30, SLIDER_HEIGHT), key='-VAL_LO-', enable_events=True)],
          [sg.Slider((0,255),confjs['value'][1], 1, orientation='h', size=(30, SLIDER_HEIGHT), key='-VAL_HI-', enable_events=True)]])],
        [sg.Text('Blob close radius', size=(SLIDER_HEIGHT,1),justification='right'),
        sg.Slider((1,30),confjs['close_rad'],1, orientation='h', size=(30,SLIDER_HEIGHT), key='-BRAD-',enable_events=True)],
       ]

    layout = [[sg.Column(left_col),sg.Column(right_col)]]

    window = sg.Window('Video browser', layout)
    graph_elem = window['-GRAPH-']      # type: sg.Graph
    
    event, values = window.read(timeout=20)

    a_id = show_image(img, graph_elem)

    dragging = False
    start_point = end_point = prior_rect = None
    rect = None
    visual_rect = None

    fp = FrameProcessor()
    try:
        fp.read_config(conffile)
    except FileNotFoundError:
        pass
    try:
        rect = confjs['rect']
        template_time = confjs['pos']
        cap.set(cv2.CAP_PROP_POS_MSEC,template_time*1000)
        ret, img = cap.read()

        start_point, end_point = rect
        window["-INFO-"].update(value=f"grabbed rectangle from {start_point} to {end_point}")
        #visual_rect = prior_rect
        ref_pos = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
        dragging = False
        prior_rect = None
        template = img[min(start_point[1],end_point[1]):max(start_point[1],end_point[1]),
                        min(start_point[0],end_point[0]):max(start_point[0],end_point[0])]
        anchor = ((start_point[0]+end_point[0])/2,
                        (start_point[1]+end_point[1])/2)
        start_point, end_point = None, None  # enable grabbing a new rect
        fp.process(img)
        original_angle = rect_angle(fp.new_rect)
    except KeyError:
        print("template not found")
        anchor = confjs['anchor']
        
        fp.anchor_pt = anchor
    


    def process(image, fp, template=None):
        print(image.shape,template.shape)
        nonlocal original_angle, templ_cent, line_angle, val, pos
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, lower, upper)
        # output = cv2.bitwise_and(image,image, mask= mask)
        fp.img = image
        fp.process(image)
        angle = rect_angle(fp.new_rect)
        # matched template 
        val, pos = template_track_rot(image, template, original_angle-angle)
        templ_shape = template.shape
        # green_area crop direction
        templ_cent = (pos[0]+templ_shape[1]/2,pos[1]+templ_shape[0]/2)
        line_angle = (angle-90)/180*np.pi
        fp.crop_strip(templ_cent, line_angle)
        return fp.image_results()

    def redraw(pos_msec=None):
        nonlocal img
        nonlocal a_id, l_id, t_id, l_id
        nonlocal angle, line_end
        nonlocal template
        if pos_msec is None:
            pos_msec = int(values['-POS-']*1000)
        cap.set(cv2.CAP_PROP_POS_MSEC,pos_msec)
        ret, img = cap.read()
        proc_img = img
        if not ret:
            return a_id
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        sec=msec/1000.
        window['-SEC-'].update(f'{sec:.3f}')
        #print(msec)
        if values['-PROCESS-'] or (not hasattr(fp, 'lower_color')):
            jsc = {'hue': {'min': values['-HUE_LO-'],
                            'max': values['-HUE_HI-']},
                    'saturation': {'min': values['-SAT_LO-'],
                                    'max': values['-SAT_HI-']},
                    'value': {'min': values['-VAL_LO-'],
                                'max': values['-VAL_HI-']}}
            fp.set_color_range(jsc)
            fp.fill_closure_rad = int(values['-BRAD-'])
            a_id = show_image(proc_img, graph_elem, a_id, t_id, l_id)
        if values['-PROCESS-']:
            proc_img = process(proc_img,fp,template=template)
            a_id = show_image(proc_img, graph_elem, a_id, t_id, l_id)
            angle = rect_angle(fp.new_rect)
            templ_shape = template.shape
            templ_cent = (pos[0]+templ_shape[1]/2,pos[1]+templ_shape[0]/2)
            line_angle = (angle-90)/180*np.pi
            line_end = (templ_cent[0]+line_len*np.cos(line_angle),templ_cent[1]+line_len*np.sin(line_angle))
            t_id = graph_elem.draw_rectangle([pos[0],pos[1]],[pos[0]+templ_shape[1],pos[1]+templ_shape[0]],fill_color=None,line_color='blue')
            l_id = graph_elem.draw_line(templ_cent,line_end, color='green')
            window["-INFO-"].update(value=f"Area: {fp.area}; Filled: {fp.area/fp.filled_area*100:5.2f}%; Angle: {angle:4.1f} deg.; Match: {val*100:5.2f}%")
        else:
            a_id = show_image(img, graph_elem, a_id)

        return a_id 

    # a_id = show_image(img, graph_elem, a_id)
    a_id = redraw()
    #fp.init_references(img)



    while True:
        event, values = window.read(timeout=20)
        if event in ('Exit', None):
            break


        if event in ['-POS-', '-PROCESS-', '-HUE_LO-', '-HUE_HI-', '-SAT_LO-', '-SAT_HI-',
                     '-VAL_LO-', '-VAL_HI-','-BRAD-']:
            a_id = redraw()
            last_pos = values['-POS-']

        if event == '-GRAPH-':
            x, y = values["-GRAPH-"]
            if not dragging:
                start_point = (x, y)
                dragging = True
                drag_figures = graph_elem.get_figures_at_location((x,y))
                lastxy = x, y
            else:
                end_point = (x, y)
            if prior_rect:
                graph_elem.delete_figure(prior_rect)
            if t_id:
                graph_elem.delete_figure(t_id)
            delta_x, delta_y = x - lastxy[0], y - lastxy[1]
            lastxy = x,y
            if None not in (start_point, end_point):
                if visual_rect:
                    graph_elem.delete_figure(visual_rect)
                    visual_rect = None
                prior_rect = graph_elem.draw_rectangle(start_point, end_point,fill_color=None, line_color='red')
            window["-INFO-"].update(value=f"mouse {values['-GRAPH-']}")
        elif event.endswith('+UP'):  # The drawing has ended because mouse up
            window["-INFO-"].update(value=f"grabbed rectangle from {start_point} to {end_point}")
            rect = (start_point,end_point)
            visual_rect = prior_rect
            ref_pos = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
            dragging = False
            prior_rect = None
            template = img[min(start_point[1],end_point[1]):max(start_point[1],end_point[1]),
                           min(start_point[0],end_point[0]):max(start_point[0],end_point[0])]
            fp.anchor_pt = ((start_point[0]+end_point[0])/2,
                            (start_point[1]+end_point[1])/2)
            start_point, end_point = None, None  # enable grabbing a new rect
            fp.process(img)
            original_angle = rect_angle(fp.new_rect)

        if event == 'Template':
            angle = rect_angle(fp.new_rect)
            a_id = show_image(rotate_image(template,original_angle-angle)[0], graph_elem, a_id, t_id)
            

        if event == 'Back to ref':
            pos_var = window['-POS-'].update(ref_pos)
            pos_msec = ref_pos*1000
            a_id = redraw(pos_msec)
        if event == 'Save' or event == 'Print':
            dd = {'pos':ref_pos,
                  'rect':rect,
                  'hue':[values['-HUE_LO-'], values['-HUE_HI-']],
                  'saturation':[values['-SAT_LO-'], values['-SAT_HI-']],
                  'value':[values['-VAL_LO-'], values['-VAL_HI-']],
                  'close_rad':values['-BRAD-']
                  }
            if event == 'Save':
                outfile = os.path.splitext(vidfile)[0]+'_conf.json'
                with open(outfile,'w') as f:
                    json.dump(dd, f)
                    print('Config written to {}'.format(outfile))
            else:
                sg.Popup(dd)






    window.close()


if __name__ == '__main__':
    main()