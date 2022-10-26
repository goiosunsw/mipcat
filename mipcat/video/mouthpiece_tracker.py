import os
import argparse
import numpy as np
import cv2
from .mouthpiece_process import FrameProcessor
from .template_trackers import template_track_rot
from .template_trackers import MultiAngleTemplateTracker
from . import template_trackers

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

class MouthpieceTracker(FrameProcessor):
    def __init__(self, nbr_templates_every=30, *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.template_rect = None
        self.template_time = 0.0

        self.has_template_config = False
        self.nbr_templates_every = nbr_templates_every
        if self.nbr_templates_every:
            self.load_nbr_templates()
        self.last_nbr_template_time = -1000
        
    def load_nbr_templates(self):
        
        use_templates = ['template_20.png','template_25.png','template_30.png']

        base_templ_path = '/'.join(os.path.abspath(template_trackers.__file__).split('\\')[:-2])+'/resources/'
        templ_fn = [base_templ_path + t for t in use_templates]
        self.nbr_templates = [cv2.imread(tt) for tt in templ_fn]
        self.nbr_names = use_templates

    def detect_nbrs(self, img):
        rects = []
        vals = []
        angles = []
        for template in self.nbr_templates:
            h, w = template.shape[:2]
            trk = MultiAngleTemplateTracker(template, angle_step=15, size_fact=1.3, n_size=5)
            cent, angle = trk.match(img)
            cent = [int(c) for c in cent]
            rects.append([int(cent[0]-w/2), int(cent[1]-h/2), w, h])
            vals.append(trk.val)
            angles.append(angle)
        return rects, vals, angles

    def set_template_from_time(self, rect, time=0.0):
        self.template_rect = rect
        self.template_time = time
        ret, img = self.get_frame(time=time)
        self.template = img[rect[1]:rect[1]+rect[3], 
                            rect[0]:rect[0]+rect[2]]
        self.has_template_config = True
        self.anchor_pt = [rect[0]+rect[2]//2,rect[1]+rect[3]//2]
        self.recalc_template_refs()

    def recalc_template_refs(self):
        # below could be replaced by simplr self.process
        #self.process(self.img)
        self.color_convert()
        self.find_green_strip()
        self.measure()
        self.original_angle = rect_angle(self.new_rect)

    def set_color_range(self, jsc):
        super().set_color_range(jsc)
        self.recalc_template_refs()

    def find_template(self, img):
        angle = rect_angle(self.new_rect)
        templ_shape = self.template.shape
        # matched template 
        val, pos = template_track_rot(img, self.template,
                                      self.original_angle-angle)

        # green_area crop direction
        self.templ_rect = [pos[0], pos[1], templ_shape[1],templ_shape[0]]
        self.templ_cent = (pos[0]+templ_shape[1]/2,pos[1]+templ_shape[0]/2)
        self.line_angle = (angle-90)/180*np.pi

    def _config(self, jsc):
        super()._config(jsc)
        try:    
            template_pts = jsc['rect']
        except KeyError:
            return
        try:
            self.template_time = jsc['pos']
        except KeyError:
            self.template_time = 0

        ul = (min(template_pts[0][0], template_pts[1][0]),
              min(template_pts[0][1], template_pts[1][1]) )
        rect = [*ul,
                abs(template_pts[1][0]-template_pts[0][0]),
                abs(template_pts[1][1]-template_pts[0][1])]
        self.anchor_pt = [rect[0]+rect[2]//2,rect[1]+rect[3]//2]
        self.template_rect = rect
        self.set_template_from_time(rect, self.template_time)
        self.has_template_config = True
        print("Done template config")
        print(rect)

    def config_to_json(self):
        jsc = super().config_to_json()
        jsc['rect'] = [[self.template_rect[0], self.template_rect[1]],
                       [self.template_rect[0] + self.template_rect[2],
                        self.template_rect[1] + self.template_rect[3]]]
        jsc['pos'] = self.template_time
        return jsc
        
    def process(self, img=None):
        super().process(img)
        if self.has_template_config:
            if img is None:
                img = self.img
            self.find_template(img)
            self.crop_strip(self.templ_cent, self.line_angle)
            self.measure()
            
        self.this_res = {'area':int(self.area),
                         'filled_area':int(self.filled_area),
                         'minRect':self.new_rect}

        if self.time-self.last_nbr_template_time > self.nbr_templates_every:
            self.last_nbr_template_time = self.time
            rects, vals, angles = self.detect_nbrs(img)
            self.this_res['other_rects'] =  {nn:{'rect': rect, 'val': float(val), 'angle': float(angle)} 
                                             for nn,rect, val, angle in zip(self.nbr_names, 
                                                                    rects, vals, angles)}

    def image_results(self):
        ims = super().image_results()
        if self.has_template_config:
            ims = cv2.circle(ims, np.array(self.templ_cent, dtype='int'), 3, [0,0,255])
        return ims
        

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="movie filename")
    parser.add_argument("-s", "--start_sec", type=float, default=0.0,
                        help="start time in seconds")
    parser.add_argument("-e", "--end_sec", type=float, default=0.0,
                        help="end time in seconds")
 
    parser.add_argument("-c", "--config", 
                        help="Config json file")
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

    processor = MouthpieceTracker(progress=True)
    processor.set_video_source(args.filename, from_time=args.start_sec, to_time=end_time)
    processor.read_config(color_range_json_file)
    processor.run()
    processor.to_json(output)

