import numpy as np
import cv2

def no_crop_rotate(image, angle, scale=1.0):
    height, width = image.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, scale)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rot= cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rot, np.array([bound_w,bound_h])/2

def calc_inner_rect(rotated_rect):
    (cx, cy), (w, h), angle_deg = rotated_rect
    
    # from https://stackoverflow.com/a/65763369
    angle = np.mod(angle_deg,180)/180*np.pi
    if angle>np.pi/2:
        angle=np.pi-angle
    w,h=h,w
    
    if h < w * np.sin(2*angle):
        new_w = h/2/np.cos(angle)
        new_h = h/2/np.sin(angle)
    elif w < h * np.sin(2*angle):
        new_w = w/2/np.sin(angle)
        new_h = w/2/np.cos(angle)
    else:
        new_w = (h*np.cos(angle)-w*np.sin(angle))/np.cos(2*angle)
        new_h = (w*np.cos(angle)-h*np.sin(angle))/np.cos(2*angle)

    return (int(cx-new_w/2), int(cy-new_h/2), int(new_w), int(new_h))

def rotate_image(image, angle_deg, scale=1.0):
    h, w = np.asarray(image.shape[:2])
    imrot, cent = no_crop_rotate(image, angle_deg, scale=scale)
    hr, wr = imrot.shape[:2]
    cent = (wr//2,hr//2)
    
    rot_rect = (cent, (w*scale,h*scale), angle_deg)
    inner_rect = calc_inner_rect(rot_rect)
    crop = inner_rect[0], inner_rect[1]
    return imrot[inner_rect[1]:inner_rect[1]+inner_rect[3],
                 inner_rect[0]:inner_rect[0]+inner_rect[2]], crop

    
def multiple_rotations(image, angle_step=90):
    rots = []
    for angle in range(0, 360, angle_step):
        rots.append(rotate_image(image, angle))
    return rots
    
    
def template_track(image, template, template_method=cv2.TM_CCORR_NORMED):
    # Apply template Matching
    res = cv2.matchTemplate(image, template, template_method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if template_method == cv2.TM_SQDIFF or template_method == cv2.TM_SQDIFF_NORMED:
        return min_val, min_loc
    else:
        return max_val, max_loc


def template_track_rot(image, template, rot=0.0, rot_tol=5.0, template_method=cv2.TM_CCORR_NORMED):
    crop = (0,0)
    if np.abs(rot)>rot_tol:
        template, ncrop = rotate_image(template, rot)
    val, pos =  template_track(image, template)
    return val, (pos[0]-crop[0], pos[1]-crop[1])


class MultiAngleTemplateTracker(object):
    def __init__(self, template=None, angle_step=30, size_fact=2, n_size=1):
        self.angle_pool = np.arange(angle_step-180,180,angle_step).astype('f')
        if n_size>1:
            n_size = int(np.floor(n_size/2)*2+1)
            self.size_pool = np.logspace(-np.log10(size_fact), np.log10(size_fact), n_size)
        else:
            self.size_pool = np.array([1.0])
        if template is not None:
            self.set_template(template)    

    def set_template(self, template):
        self.template = template
        self.rot_templates = []
        self.crops = []
        self.angles = []
        self.sizes = []
        for angle in self.angle_pool:
            for size in self.size_pool:
                templ, crop = rotate_image(self.template, angle, scale=size)
                self.rot_templates.append(templ)
                self.crops.append(crop)
                self.angles.append(angle)
                self.sizes.append(size)

    def match(self, img):
        vals = []
        cents = [] 
        for templ, crop in zip(self.rot_templates, self.crops):
            shp = templ.shape
            val, pos = template_track(img, templ)
            vals.append(val)
            cents.append([pos[0]+shp[1]/2,pos[1]+shp[0]/2])
            
        cent, angle, size, val = max(zip(cents, self.angles, self.sizes, vals), key=lambda x: x[3]) 
        self.found_index = vals.index(val)

        self.cent = cent
        self.size = size
        self.angle = -angle
        self.val = val

        return cent, angle
    
    @property
    def shape(self):
        shp = self.template.shape
        return np.array([shp[1], shp[0]])
    
    @property
    def rect(self):
        return (self.cent, self.shape*self.size, self.angle)

    @property
    def found_template(self):
        return self.rot_templates[self.found_index]
