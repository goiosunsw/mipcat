import os
import argparse
import json
import pickle
import numpy as np
import cv2 
import scipy.signal as sig
from tqdm import trange, tqdm


DEFAULT_HSV_RANGE = [(.9,.2),(0.6,1.0),(0.3,1.0)]

def hsv_mask(hsv, lower, upper):
    if lower[0] > upper[0]:
        ret = (((hsv[:,:,0]>=lower[0]) | (hsv[:,:,0]<=upper[0]))  & 
               ((hsv[:,:,1]>=lower[1]) & (hsv[:,:,1]<=upper[1]))  &
               ((hsv[:,:,2]>=lower[2]) & (hsv[:,:,2]<=upper[2])))
    else:
        ret = (((hsv[:,:,0]>=lower[0]) & (hsv[:,:,0]<=upper[0]))  & 
               ((hsv[:,:,1]>=lower[1]) & (hsv[:,:,1]<=upper[1]))  &
               ((hsv[:,:,2]>=lower[2]) & (hsv[:,:,2]<=upper[2])))
    return ret.astype('uint8')
        

def find_nearest_blob(mask, anchor_pt, minArea=20, mindist = 1e12):
    maskr=mask

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

class FrameProcessor(object):
    def __init__(self, min_area=20, hsv_range=DEFAULT_HSV_RANGE, 
                 anchor_point=(0,0), fg_opening_rad=40, fg_fill_rad=5):
        self.min_area = min_area
        self.hsv_low_range = [int(hsv_range[0][0]*179),
                              int(hsv_range[1][0]*255),
                              int(hsv_range[2][0]*255)]
        self.hsv_hi_range = [int(hsv_range[0][1]*179),
                              int(hsv_range[1][1]*255),
                              int(hsv_range[2][1]*255)]
        self.anchor_point = np.asarray(anchor_point)
        self.fg_opening_rad = fg_opening_rad
        self.fg_fill_rad = fg_fill_rad

    def set_video_source(self, video_file, pos=0):
        self.video_file = video_file
        cap = cv2.VideoCapture(cv2.samples.findFile(video_file))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video length:", length)
        self.cap = cap
        self.n_frames = length
        self.get_frame(pos)

    def get_frame(self, pos=None):
        if pos is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, self.img = self.cap.read()
        self.pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ret:
            print("Error reading frame number ", pos)
        return self.img

    def show_img(self, img, window="image"):
        cv2.imshow(window, ensure_max_dim(img,1200,800))

    def show_mask(self):
        ims = self.roi_mask.astype('uint8')*255
        self.show_img(ims, "image")

    def image_results(self):
        """
        Produce an image overlayed with detection results
        """
        ims = self.image.copy()
        h, w, *_ = ims.shape
        cv2.drawContours(ims,self.contours,0,[0,255,0],3)
        
        # draw reference rectangle
        pts=cv2.boxPoints(self.results['min_rect'])
        pts = pts.reshape((-1,1,2)).astype('int32')
        cv2.polylines(ims,[pts],True,(0,0,255))
        
        # print area
        cv2.putText(ims,str(self.results['area']),(0,h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        return ims #ensure_max_dim(ims, 1200, 800)
        
    def process(self, img):
        """
        Main detection function
        """
        self.image = img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = hsv_mask(hsv, self.hsv_low_range, self.hsv_hi_range)
        output = cv2.bitwise_and(img,img, mask=(1-mask))

        rad = self.fg_opening_rad

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(rad,rad))
        clmask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

        roi_mask = cv2.bitwise_and(mask, cv2.bitwise_not(clmask))
        self.roi_mask = roi_mask

        # find blob nearest to point
        try:
            masks,new_pt = find_nearest_blob(roi_mask, self.anchor_point, 
                                            minArea = self.min_area)
        except UnboundLocalError:
            pass

        contours, _ = cv2.findContours(masks.astype('uint8')*255, 
                                    cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        c=contours[0]
        minRect = cv2.minAreaRect(c)    
        pts=cv2.boxPoints(minRect)

        # get mask with filled holes
        rad = self.fg_fill_rad
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(rad,rad))
        maskr = cv2.morphologyEx(masks.astype('uint8')*255,cv2.MORPH_CLOSE,kernel)

        ims = img.copy()

        contours, _ = cv2.findContours(maskr, 
                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        c = np.concatenate(contours)
        minRect = cv2.minAreaRect(c)    
        pts = cv2.boxPoints(minRect)


        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(ims, (cX, cY), 7, (255, 255, 255), -1)

        #print(f'Centroid: {cX}, {cY}')
        #print(f'Area: {np.sum(maskr>0)}')
        #print(f'Rect cent: {minRect[0]}, w={minRect[1][0]}, h={minRect[1][1]}, angle={minRect[2]}')

        ret = {'cent': (cX, cY),
               'area': np.sum(maskr>0),
               'min_rect': minRect}
        self.results = ret
        return ret

    def onChange(self, trackbarValue):
        img = self.get_frame(trackbarValue)
        self.process(img)
        self.show_img(self.image_results(),window="image")
        pass

    def gui(self):
        wait_time = 33
        cv2.namedWindow('image')
        cv2.createTrackbar('frame no.', 'image', 0, self.n_frames, 
                          self.onChange)

        self.onChange(0)
        
        self.get_frame(0)
        while (self.cap.isOpened()) & (cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) > 0):
            k = cv2.waitKey(33)
            if k == 27:
                break
            if k == ord('m'):
                self.show_mask()

        cv2.destroyAllWindows() 


def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="movie filename")
    parser.add_argument("-n", "--frame-number", type=int, default=0,
                        help="Process frame number")
    parser.add_argument("-g", "--gui", action='store_true',
                        help="GUI (test mode)")

    return parser.parse_args()


def process_video(processor, progress=True):
    ret_array = []
    basename = os.path.splitext(processor.video_file)[0]

    if progress:
        pbar = tqdm(total=processor.n_frames)

    with open(basename, 'w', 1) as f:
        for ii in range(processor.n_frames):
            img = processor.get_frame()
            try:
                ret = processor.process(img)
            except Exception:
                ret = {}
            ret['time'] = processor.pos_msec
            ret_array.append(ret)
            if progress:
                pbar.update()
        if progress:
            pbar.close()
    
    with open(basename+'_bbox.pickle', 'wb') as f:
        pickle.dump(ret_array, f)


if __name__ == "__main__":
    args = argument_parse()
    pos = args.frame_number
    processor = FrameProcessor()
    processor.set_video_source(args.filename)
    if args.gui:
        processor.gui()
    else:
        process_video(processor)

