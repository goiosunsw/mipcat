import os
import cv2
import sys
import argparse
import numpy as np

MAX_WIDTH  = 800
MAX_HEIGHT = 600

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="movie filename", default="")
parser.add_argument("-k", "--k-means", type=int, default=2,
                    help="number of means to compute")
parser.add_argument("-n", "--frame-number", type=int, default=0,
                    help="Process frame number")

args = parser.parse_args()
pos = args.frame_number

global img
img=None

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

def nothing(x):
    pass

def video_seek(trackbarValue):
    cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
    err,img = cap.read()
    if not err:
        print("Error reaing frame number ",trackbarValue)

    pass



useCamera=False

# Check if filename is passed
if (len(args.filename) <= 1) :
    print("'Usage: python hsvThresholder.py <ImageFilePath>' to ignore camera and use a local image.")
    useCamera = True

# Create a window
#cv2.namedWindow('image')
cv2.namedWindow('controls', flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)

# create trackbars for color change
cv2.createTrackbar('HMin','controls',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','controls',0,255,nothing)
cv2.createTrackbar('VMin','controls',0,255,nothing)
cv2.createTrackbar('HMax','controls',0,179,nothing)
cv2.createTrackbar('SMax','controls',0,255,nothing)
cv2.createTrackbar('VMax','controls',0,255,nothing)
cv2.createTrackbar('Process','controls',0,1,nothing)
cv2.createTrackbar('Mask','controls',0,1,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'controls', 179)
cv2.setTrackbarPos('SMax', 'controls', 255)
cv2.setTrackbarPos('VMax', 'controls', 255)
cv2.setTrackbarPos('Process', 'controls', 1)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# Output Image to display
if useCamera:
    cap = cv2.VideoCapture(0)
    # Wait longer to prevent freeze for videos.
    waitTime = 330
else:
    img = cv2.imread(args.filename)

    if img is None: 
        cap = cv2.VideoCapture(cv2.samples.findFile(args.filename))
        cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
        ret, img = cap.read()
        if not ret:
            print('Could not read movie')
            exit(1)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video length:",length)

        cv2.createTrackbar('Video_position','controls',0,length-1,video_seek)

    img = ensure_max_dim(img, width=MAX_WIDTH, height=MAX_HEIGHT)
    output = img
    waitTime = 33

last_frame = 0
while(1):

    this_frame = cv2.getTrackbarPos('Video_position','controls')
    if this_frame != last_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES,this_frame)
        err,img = cap.read()
        img = ensure_max_dim(img, width=MAX_WIDTH, height=MAX_HEIGHT)
        
        if not err:
            print("Error reaing frame number ",trackbarValue)
        output = img
        last_frame = this_frame


    if useCamera:
        # Capture frame-by-frame
        ret, img = cap.read()
        output = img

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','controls')
    sMin = cv2.getTrackbarPos('SMin','controls')
    vMin = cv2.getTrackbarPos('VMin','controls')

    hMax = cv2.getTrackbarPos('HMax','controls')
    sMax = cv2.getTrackbarPos('SMax','controls')
    vMax = cv2.getTrackbarPos('VMax','controls')
    
    process = cv2.getTrackbarPos('Process','controls')
    showmask = cv2.getTrackbarPos('Mask', 'controls')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    if process==0:
        cv2.imshow('image',img)
    else:
        if showmask==0:
            cv2.imshow('image',output)
        else:
            cv2.imshow('image',mask)

    # Wait longer to prevent freeze for videos.
    key = cv2.waitKey(waitTime) 
    if key & 0xFF == ord('w'):
        import json
        basename = os.path.splitext(args.filename)[0]
        data = {'hue':{'min':hMin,'max':hMax},
                'saturation':{'min':sMin,'max':sMax},
                'value':{'min':vMin,'max':vMax}}
        with open(basename+"_colormask.json", "w+") as f:
            f.write(json.dumps(data,indent=4))
    if key & 0xFF == ord('q'):
        break


# Release resources
if useCamera:
    cap.release()
cv2.destroyAllWindows()
