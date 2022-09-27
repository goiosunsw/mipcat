import sys
import os
import pickle
import argparse
import json
import numpy as np
import cv2
import logging
import traceback
from tqdm import trange
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class ArucoTracker(object):
    def __init__(self, from_time=0.0, to_time=None, track_undetected=False,
                 output=None):
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        arucoParams = cv2.aruco.DetectorParameters_create()
        arucoParams.adaptiveThreshConstant = 5
        arucoParams.adaptiveThreshWinSizeMin = 3
        arucoParams.adaptiveThreshWinSizeStep= 5
        arucoParams.minMarkerPerimeterRate = 0.03
        self.arucoParams = arucoParams
        self.template_method = cv2.TM_SQDIFF
        self.track_undetected = track_undetected

        self.output = output
        if self.output is None:
            self.output = 'aruco_output.json'
        
        self.unique_ids = set()
        self.markers = []
        self.centers = []
        self.bboxes = []
        self.last_templates = {}
        self.cert_templates = {}
        self.last_bbox = {}
        self.cert_bbox = {}
        
        
    def aruco_detect(self,image):
        (corners_, ids_, rejected) = cv2.aruco.detectMarkers(image, self.arucoDict,
            parameters=self.arucoParams)
        
        corners = corners_
        try:
            ids = ids_.flatten()
        except AttributeError:
            ids = np.array([])
        
        self.unique_ids.update(tuple(ids.tolist()))
        mrks = {}
        for iid in self.unique_ids:
            try:
                idx = np.flatnonzero(ids==iid)[0]
                mrks[iid] = {'corners':corners[idx], 'source':'aruco'}
            except IndexError:
                mrks[iid] = {}
                
        for iid,mrk in mrks.items():
            if 'corners' in mrk:
                bbox = cv2.boundingRect(mrk['corners'])
                mrk['bbox'] = bbox 
                template = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
                self.cert_templates[iid] = template
                self.cert_bbox[iid] = bbox
                self.last_templates[iid] = template
                self.last_bbox[iid] = bbox
                
        self.markers.append(mrks)
                
    def template_track_remaining(self, image):
        markers = self.markers[-1]
        remaining_ids = [k for k,v in markers.items() if len(v)==0]

        for iid in remaining_ids:
            template = self.last_templates[iid]
            bbox = self.last_bbox[iid]
            pval,pos = self.template_track(image, template)
            bbox = (pos[0],pos[1],bbox[2],bbox[3])
            template = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
            self.last_bbox[iid] = bbox
            self.last_templates[iid] = template
            markers[iid] = {'bbox':bbox,'source':'template','template_val':pval}
        
    def template_track(self, image, template):
        # Apply template Matching
        res = cv2.matchTemplate(image,template,self.template_method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if self.template_method == cv2.TM_SQDIFF or self.template_method == cv2.TM_SQDIFF_NORMED:
            return min_val, min_loc
        else:
            return max_val, max_loc
        
        
    def detect(self,image):
        self.aruco_detect(image)
        if self.track_undetected:
            self.template_track_remaining(image)
                
    
    def to_json(self, filename=None):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

            
        with open(self.output_file, 'w') as f:    
            #json.dump(rets,f,default=default_ser)
            
            if filename is not None:
                with open(filename,'w') as f:
                    json.dump(self.markers,f,cls=NumpyEncoder)
            else:
                return json.dumps(self.markers,cls=NumpyEncoder)
        
    def center_array(self):
        cents=[{k:(v['bbox'][0]+v['bbox'][2]/2,v['bbox'][1]+v['bbox'][3]/2) 
                for k,v in mrk.items() if type(v) is dict} for mrk in self.markers]
        uids = list(self.unique_ids)
        x=np.nan*np.ones((len(cents),len(uids),2))
        for idx,cidx in enumerate(uids):
            for ii,c in enumerate(cents):
                try:
                    x[ii,idx,:] = c[cidx]
                except KeyError:
                    pass
        return x

    def draw_on_image(self, image):
        mrk = self.markers[-1]
        for id, m in mrk.items():
            corners = np.array(m['corners'][0]).astype('i')
            print(corners)
            # Draw contour
            for j in range(corners.shape[0]):
                next = (j + 1) % (corners.shape[0])
                cv2.line(image, (corners[j]), (corners[next]), (0, 255, 0), 5)


def ensure_max_dim(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if h<height and w<width:
        return image

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="movie filename")
    parser.add_argument("-m", "--pos-msec", type=int, default=0,
                        help="Process frame number")
    parser.add_argument("-g", "--gui", action='store_true',
                        help="GUI (test mode)")
    parser.add_argument("-r", "--rotate", type=int, default=0,
                        help="Rotate image")
    parser.add_argument("-s", "--single", action='store_true', 
                        help="output information about single frame")
    parser.add_argument("-a", "--aruco", action='store_true', 
                        help="add aruco tracking")
    parser.add_argument("-o", "--output", type=str,
                        help="output file")
                    

    return parser.parse_args()

def output_results(results, time=0.0, output=sys.stdout):
    output.write(f"time: {time/1000:.4f}")
    if results is None or results.pose_landmarks is None:
        return
    for ii, lm in enumerate(results.pose_landmarks.landmark):
        line = f"""    
    landmark {ii}:        
        x: {lm.x}, 
        y: {lm.y}, 
        z: {lm.z}, 
        visibility: {lm.visibility}"""
        output.write(line)
    output.write("\n")

def pickle_results(result_list, output=''):
    with open(output, 'wb') as f:
        pickle.dump(result_list,f)

if __name__ == "__main__":
    args = argument_parse()

    video_file = args.filename
    pos_msec = args.pos_msec

    # For video input:
    cap = cv2.VideoCapture(cv2.samples.findFile(video_file))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_MSEC, pos_msec)
    n=0

    if args.aruco:
        aruco = ArucoTracker()

    result_list = []

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2,
        smooth_landmarks=False) as pose:

        # while cap.isOpened():
        for ii in trange(n_frames):
            success, image = cap.read()
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            if args.rotate != 0:
                if args.rotate == 90:
                    rot = cv2.ROTATE_90_CLOCKWISE
                elif args.rotate == 180:
                    rot = cv2.ROTATE_180
                elif args.rotate == 270:
                    rot = cv2.ROTATE_90_COUNTERCLOCKWISE
                else:
                    print('wrong rotation value!')
                    exit(1)

                image = cv2.rotate(image, rot)

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                results = pose.process(image)
            except Exception:
                logging.warning(traceback.format_exc())
                results = None
                output_results(results,time=time)
                continue

            if args.aruco:
                aruco.detect(image)
                try:
                    aruco.draw_on_image(image)
                except KeyError:
                    pass
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            if args.gui:
                cv2.imshow('MediaPipe Pose Detection', ensure_max_dim(image,height=800))
            if args.single:
                key = cv2.waitKey(0)
                if key & 0xFF == 27:
                    break   

            res = {'time': time/1000}
            if results is None or results.pose_landmarks is None:
                pass
            else:
                landmrks = {}
                for ii, lm in enumerate(results.pose_landmarks.landmark):
                    landmrks[ii] = {       
                        'x': lm.x, 
                        'y': lm.y, 
                        'z': lm.z, 
                        'visibility': lm.visibility
                    }
                res['landmarks'] = landmrks
            result_list.append(res)

            # try:
                # output_results(results, time=time)
            # except Exception:
                # logging.warning(traceback.format_exc())

            if args.gui:
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            
            if n > n_frames:
                break

            n+=1
        cap.release()
        outfile = os.path.splitext(args.filename)[0]+'_pose.pickle'
        pickle_results(result_list, outfile)