import sys
import collections
import face_alignment
import numpy as np
import json
import cv2
from skimage import io
from skimage import draw
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

# 2D-Plot
plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                  flip_input=False, device='cpu')


if len(sys.argv)>2:
    frame_no = int(sys.argv[2])
else:
    frame_no = 0
try:
    input = io.imread(sys.argv[1])
except ValueError:
    cap = cv2.VideoCapture(sys.argv[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, input = cap.read()
    print(cap.get(cv2.CAP_PROP_POS_MSEC))
    input = input[:,:,::-1]
    
input = cv2.rotate(input,cv2.ROTATE_90_COUNTERCLOCKWISE)

plt.imshow(input)
preds = fa.get_landmarks(input)

for pt in preds[0]:
    rr,cc=draw.disk((pt[0],pt[1]),2,shape=input.shape)
    input[rr,cc,0]=1


faces = []
for face in preds:
    #plt.scatter(face[:,0],face[:,1],2)
    facedict={}
    for item, pred_type in pred_types.items():
        plt.plot(face[pred_type.slice, 0],
                face[pred_type.slice, 1],
                color=pred_type.color, **plot_style)
        facedict[item] = face[pred_type.slice,:].tolist()
    faces.append(facedict)

def default_serializer(x):
    try:
        return float(x)
    except ValueError:
        return str(x)

print(json.dumps(faces,default=default_serializer,indent=4))

plt.show()
