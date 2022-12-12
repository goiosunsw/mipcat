import cv2
import os
import json
import pickle
import argparse
import numpy as np
from tqdm import trange, tqdm

def corner_dist(x,y):
    sdm=((x[:,:,np.newaxis]-y[:,:,np.newaxis].transpose())**2).sum(axis=1)
    mdm=sdm.copy()
    minvals=np.zeros(mdm.shape[0])
    indexes = np.zeros((mdm.shape[0],2))
    for ii in range(mdm.shape[0]):
        minval=np.min(mdm)
        minvals[ii]=np.sqrt(minval)
        row,col = divmod(mdm.argmin(),mdm.shape[1])
        indexes[ii,0]=row
        indexes[ii,1]=col
        mdm[row,:]=np.max(mdm)
        mdm[:,col]=np.max(mdm)
    return minvals, indexes, sdm

def centers_from_bbox_json(markers):
    # cents=[{k:(v['bbox'][0]+v['bbox'][2]/2,v['bbox'][1]+v['bbox'][3]/2) 
    #         for k,v in mrk.items() if type(v) is dict} 
    #        for mrk in markers]
    # unique_ids = set([k for mrk in markers for k,v in mrk.items() if t])
    cents = []
    qual = []
    id_count = {} 
    for mrk in markers:
        c = {}
        q = {}
        for k,v in mrk.items():
            try:
                c[k] = v['bbox'][0]+v['bbox'][2]/2,v['bbox'][1]+v['bbox'][3]/2
            except (KeyError, TypeError):
                continue
            try:
                q[k] = v['template_val']
            except (KeyError, TypeError):
                pass
            try:
                if 'corners' in v:
                    id_count[k]+=1
            except KeyError:
                id_count[k]=1
        cents.append(c)
        qual.append(q)
    uids = list(id_count.keys())

    x = np.nan*np.ones((len(cents),len(uids),2))
    q = np.zeros((len(cents),len(uids)))
    for idx,cidx in enumerate(uids):
        for ii,c in enumerate(cents):
            try:
                x[ii,idx,:] = c[cidx]
            except KeyError:
                pass
            try:
                q[ii,idx] = qual[ii][cidx]
            except KeyError:
                pass
    return x, id_count, q

class ArucoTracker(object):
    """Track Aruco tags on a video
       This class adds similarity tracking whenever a previous
       Aruco tag cannot be found in the current frame.
    """
    def __init__(self, vidfile, from_time=0.0, to_time=None, progress=False, forget_time=1.0, 
                 crop=None):
        """Initialize tracker with a video file

        Args:
            vidfile (string): Path to the video file
            from_time (float, optional): Starting time for tracking. Defaults to 0.0.
            to_time ([type], optional): End time for tracking. Defaults to end of file.
            progress (bool, optional): Show progress bar. Defaults to False.
            forget_time (float, optional): Do not keep tracking after x seconds. Defaults to 5.0.
        """
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        arucoParams = cv2.aruco.DetectorParameters_create()
        arucoParams.adaptiveThreshConstant = 5
        arucoParams.adaptiveThreshWinSizeMin = 3
        arucoParams.adaptiveThreshWinSizeStep= 5
        arucoParams.minMarkerPerimeterRate = 0.03
        self.arucoParams = arucoParams
        self.template_method = cv2.TM_SQDIFF
        self.forget_time = forget_time
        self.progress = progress
        self.cur_time = 0.0

        self.unique_ids = set()
        self.markers = []
        self.centers = []
        self.bboxes = []
        self.last_templates = {}
        self.last_seen = {}
        self.cert_templates = {}
        self.last_bbox = {}
        self.cert_bbox = {}
        self.crop=crop
        
        self.set_source(vidfile, from_time=from_time, to_time=to_time)
        
    def set_source(self, vidfile, from_time=0.0, to_time=None):
        cap = cv2.VideoCapture(cv2.samples.findFile(vidfile))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rate = (cap.get(cv2.CAP_PROP_FPS))
        print(f"Video length: {length} ({length/rate} sec)")
        print(f"{rate} FPS")

        cap.set(cv2.CAP_PROP_POS_MSEC,from_time*1000)
        self.cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if to_time is not None:
            self.max_time = to_time
        else:
            self.max_time = length/rate
            
        self.min_time = from_time
        self.frame_msec = 1000/rate
        self.length = int((self.max_time - self.min_time)*rate)

        self.cap = cap
        
    def get_next_frame(self):
        ret, image = self.cap.read()
        self.cur_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        if self.crop:
            crop = self.crop
            try:
                image = image[crop[0]:crop[2],crop[1]:crop[3]]
            except TypeError:
                pass
        return ret, image
        
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
                self.last_seen[iid] = self.cur_time
            except IndexError:
                mrks[iid] = {}
                
        for iid,mrk in mrks.items():
            if 'corners' in mrk:
                bbox = cv2.boundingRect(mrk['corners'])
                mrk['bbox'] = list(bbox) 
                template = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
                self.cert_templates[iid] = template
                self.cert_bbox[iid] = bbox
                self.last_templates[iid] = template
                self.last_bbox[iid] = bbox
                
        if self.crop is not None:
            cropped_mrks = mrks.copy()
            for iid,mrk in mrks.items():
                for ii in [0,1]:
                    try:
                        mrk['bbox'][ii] += self.crop[1-ii]
                    except KeyError:
                        pass
                    try:
                        for crn in mrk['corners'][0]:
                            crn[ii] += self.crop[1-ii]
                    except KeyError:
                        pass
        
        self.markers.append(mrks)
                
    def template_track_remaining(self, image):
        markers = self.markers[-1]
        remaining_ids = [k for k,v in markers.items() if len(v)==0]

        for iid in remaining_ids:
            template = self.last_templates[iid]
            bbox = self.last_bbox[iid]
            pval,pos = self.template_track(image, template)
            bbox = [pos[0],pos[1],bbox[2],bbox[3]]
            template = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
            self.last_bbox[iid] = bbox
            self.last_templates[iid] = template
            if self.crop is not None:
                for ii in [0,1]:
                    bbox[ii] += self.crop[1-ii]   
            markers[iid] = {'bbox':bbox,'source':'template','template_val':pval}
        
    def template_track(self, image, template):
        # Apply template Matching
        res = cv2.matchTemplate(image,template,self.template_method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if self.template_method == cv2.TM_SQDIFF or self.template_method == cv2.TM_SQDIFF_NORMED:
            return min_val, min_loc
        else:
            return max_val, max_loc
        
    def flush_old_markers(self):
        ids = list(self.unique_ids)
        for iid in ids:
            if self.cur_time - self.last_seen[iid] > self.forget_time:
                print(f'Forgetting tracker {iid}')
                self.unique_ids.remove(iid)
        
    def detect(self,image):
        self.aruco_detect(image)
        self.template_track_remaining(image)
        self.flush_old_markers()
        
    def run(self):
        if self.progress:
            self.pbar = tqdm(total=self.length)
        for ii in range(self.length):
            ret, image = self.get_next_frame()
            if not ret:
                self.pbar.write(f'Error reading frame {ii}. Skipping')
                continue
            msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if msec>self.max_time*1000:
                break
            self.detect(image)
            self.markers[-1]['msec'] = msec
            if self.progress:
                nmarkers = len(self.markers[-1])-1
                ndet = len([k for k,v in self.markers[-1].items() 
                            if isinstance(v,dict) and 'corners' in v.keys()])
                self.pbar.update()
                self.pbar.set_postfix({'t':nmarkers,'d':ndet})
        if self.progress:
            self.pbar.close()
    
    def to_json(self, filename=None):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

            
        with open(os.path.splitext(vidfile)[0]+'_markers.json','w') as f:    
            #json.dump(rets,f,default=default_ser)
            
            if filename is not None:
                with open(filename,'w') as f:
                    json.dump(self.markers,f,cls=NumpyEncoder)
            else:
                return json.dumps(self.markers,cls=NumpyEncoder)

    def to_pickle(self, filename=None):
        if filename is None:
            filename = os.path.splitext(self.video_file)[0]+'_markers.pickle'
        with open(filename,'wb') as f:
            pickle.dump(self.markers, f)
        
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="movie filename", default="")
    parser.add_argument("-s", "--start_sec", type=float, default=0.0,
                        help="start time in seconds")
    parser.add_argument("-e", "--end_sec", type=float, default=0.0,
                        help="end time in seconds")
    parser.add_argument("-c", "--crop", default="", 
                        help="crop rectangle XL:YB:XR:YT (default = no crop)")
    parser.add_argument("-o", "--output", default="", 
                        help="output file (leave empty for same name as video)")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.end_sec <= 0.001:
        end_time = None
    else:
        end_time = args.end_sec
    
    vidfile = args.filename
    basename = os.path.splitext(vidfile)[0]
    if len(args.output)==0:
        output = basename+'_markers.json'
    else:
        output = args.output 

    if len(args.crop)>0:
        crop = [int(x) for x in args.crop.split(':')]
        print('cropping to ',crop)
    else:
        crop=None

    trk = ArucoTracker(vidfile, from_time=args.start_sec,to_time=end_time,progress=True, crop=crop)
    try:
        trk.run()
    except AttributeError:
        import traceback
        traceback.print_exc()
    finally:
        trk.to_json(output)

