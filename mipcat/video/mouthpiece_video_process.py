import os
import argparse
import json
import pickle
import numpy as np
import cv2 as cv
import scipy.signal as sig


MIN_MATCH_COUNT = 10


class FrameProcessor(object):
    def __init__(self, template, template_mask=None,
                 alignment_mask=None):
        # Initiate SIFT detector
        self.read_template(template, template_mask, alignment_mask)
        self.sift = cv.SIFT_create(nOctaveLayers=5, contrastThreshold=0.02)
        # self.sift = cv.xfeatures2d.SURF_create(hessianThreshold=minHessian)
        kp1, des1 = self.sift.detectAndCompute(self.template,
                                               self.alignment_mask)
        self.template_keypoints = kp1
        self.template_descriptors = des1
        # self.matcher = cv.BFMatcher()

        self.matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        self.gui_template = False

    def read_template(self, template_file_name, template_mask_name,
                      alignment_mask_name):
        template = cv.imread(template_file_name)
        print("template shape:", template.shape)
        template_mask = cv.imread(template_mask_name)
        try:
            template_mask = cv.cvtColor(template_mask, cv.COLOR_BGR2GRAY)
        except cv.error:
            template_mask = np.ones_like(template)[:, :, -1]*255
            print('Mask not found, proceding with full image')
        ret, template_mask = cv.threshold(template_mask, 127, 255, 
                                          cv.THRESH_BINARY)
        alignment_mask = cv.imread(alignment_mask_name)
        try:
            alignment_mask = cv.cvtColor(alignment_mask, cv.COLOR_BGR2GRAY)
        except cv.error:
            alignment_mask = np.ones_like(template)[:, :, -1] * 255
            print('Alignment mask not found, proceding with full image')
        ret, alignment_mask = cv.threshold(alignment_mask, 127, 255,
                                           cv.THRESH_BINARY)
        self.template = template
        self.template_mask = template_mask
        self.alignment_mask = alignment_mask

    def find_template(self, img):
        kp1 = self.template_keypoints
        des1 = self.template_descriptors
        # find the keypoints and descriptors with SIFT
        kp2, des2 = self.sift.detectAndCompute(img, None)
        # BFMatcher with default params
        matches = self.matcher.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        # cv.drawMatchesKnn expects list of lists as matches.

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in 
                                  good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in
                                  good]).reshape(-1, 1, 2)
            #M, mask = cv.findHomography(src_pts, dst_pts, cv.USAC_ACCURATE,5.0)
            M, mask = cv.estimateAffine2D(src_pts, dst_pts, True)
            Mp = np.vstack([M, [[0, 0, 1]]])
            matchesMask = mask.ravel().tolist()
            h, w, d = self.template.shape
            pts = np.float32([[0, 0], [0, h-1], 
                              [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, Mp)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), 
                                                                MIN_MATCH_COUNT))
            matchesMask = None
            img2 = img
            dst = None
            dst_pts = []
            src_pts = np.float32([k.pt for k in kp1]).reshape(-1, 1, 2)

        self.template_box = dst
        self.matched_points = dst_pts
        self.src_pts = src_pts
        return dst

    def identify_scale(self):
        # Create HSV Image and threshold into a range.
        img_transf = cv.warpAffine(img, Mi, (template.shape[1],template.shape[0]))
        hsv = cv.cvtColor(img_transf, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)
        output = cv.bitwise_and(img_transf, img_transf, mask=mask)
        output = output[..., 2]
        ret, output = cv.threshold(output, minv, 255, cv.THRESH_BINARY)
        output = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel_diff)
        # output = sig.medfilt2d(output, 9)
        output = cv.bitwise_and(output, template_mask) 
        
    def set_video_source(self, video_file, pos=0):
        self.video_file = video_file
        cap = cv.VideoCapture(cv.samples.findFile(video_file))
        cap.set(cv.CAP_PROP_POS_FRAMES, pos)

        length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print("Video length:", length)
        self.cap = cap
        self.n_frames = length
        self.get_frame(pos)

    def get_frame(self, pos):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, pos)
        ret, self.img = self.cap.read()
        if not ret:
            print("Error reading frame number ", pos)
        return self.img

    def image_results(self):
        dst = self.template_box
        dst_pts = self.matched_points
        if dst is not None:
            # if np.any(dst < 0):
            #    return self.img
            img2 = cv.polylines(self.img.copy(), [np.int32(dst)], True, 255, 3, 
                                cv.LINE_AA)
            for pt in dst_pts[:, -1, :]:
                img2 = cv.circle(img2, pt.astype('i'), 2, [255, 0, 0], 1)
        else:
            img2 = self.img
        return img2
        
    def process(self, img):
        box = self.find_template(img)
        self.img = img
        return box

    def onChange(self, trackbarValue):
        img = self.get_frame(trackbarValue)
        self.process(img)
        cv.imshow("image", self.image_results())
        pass

    def toggle_template(self):
        if self.gui_template:
            cv.destroyWindow('template')
            self.gui_template = False
        else:
            temp2 = self.template.copy()
            src_pts = self.src_pts

            for pt in src_pts[:, -1, :]:
                # print(pt)
                temp2 = cv.circle(temp2, pt.astype('i'), 2, [255, 0, 0], 1)
            cv.imshow('template', temp2)
            self.gui_template = True

    def gui(self):
        wait_time = 33
        cv.namedWindow('image')
        cv.createTrackbar('frame no.', 'image', 0, self.n_frames, 
                          self.onChange)

        self.onChange(0)
        
        self.get_frame(0)
        self.toggle_template()
        k = cv.waitKey()
        while self.cap.isOpened():
           if k == 27:
                break

        cv.destroyAllWindows() 


def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="movie filename")
    parser.add_argument("-n", "--frame-number", type=int, default=0,
                        help="Process frame number")
    parser.add_argument("-m", "--min-match-count", type=int, default=10,
                        help="Minimum matches")
    parser.add_argument("-t", "--template", default="mouthpiece_scale.png",
                        help="template bitmap")
    parser.add_argument("-a", "--alignment-mask",
                        help="mask used for alignment")
    parser.add_argument("-k", "--measurement-mask",
                        help="mask used for measurement")
    parser.add_argument("-g", "--gui", action='store_true',
                        help="GUI (test mode)")

    return parser.parse_args()


def process_video(processor):
    box_array = []
    basename = os.path.splitext(processor.video_file)[0]
    with open(basename, 'w', 1) as f:
        for ii in range(processor.n_frames):
            img = processor.get_frame(ii)
            box = processor.process(img)
            if box is None:
                box = []
                
            try:
                jsbox = box.tolist()
            except AttributeError:
                jsbox = box

            dd = {'frame': ii,
                  'bbox': jsbox}
            f.write(json.dumps(dd, indent=2)+',\n')
    
    with open(basename+'_bbox.pickle', 'wb') as f:
        pickle.dump(box_array, f)


if __name__ == "__main__":
    args = argument_parse()
    if args.measurement_mask is not None:
        template_mask_filename = args.measurement_mask
    else:
        template_mask_filename = os.path.splitext(args.template)[0]+'_mask.png'
    if args.alignment_mask is not None:
        alignment_mask_filename = args.alignment_mask
    else:
        alignment_mask_filename = os.path.splitext(args.template)[0]+\
                '_align_mask.png'
    print(template_mask_filename)
    pos = args.frame_number
    MIN_MATCH_COUNT = args.min_match_count
    processor = FrameProcessor(args.template, 
                               template_mask=template_mask_filename,
                               alignment_mask=alignment_mask_filename)
    processor.set_video_source(args.filename)
    if args.gui:
        processor.gui()
    else:
        process_video(processor)

