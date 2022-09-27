import sys
import cv2
import tkinter as tk
from numpy import sqrt, ones
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
from clemotion.video.template_trackers import calc_inner_rect, no_crop_rotate
from template_trackers import rotate_image


def show_image(image, container):
    h, w = image.shape[:2]
    img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=img)
    container.imgtk = imgtk
    container.configure(image=imgtk, width=w, height=h)

def resize_prop(image, size):
    h, w = image.shape[:2]
    fact_w = w/size[0]
    fact_h = h/size[1]
    fact = max(fact_w, fact_h)
    return cv2.resize(image, (int(w/fact), int(h/fact)))


class CVImage(tk.Canvas):
    def __init__(self, parent, maxdim=None, *args, **kwargs):
        if maxdim is not None:
            super().__init__(width=maxdim[0], height=maxdim[1], *args, **kwargs)
            self.width = maxdim[0]
            self.height = maxdim[1]
        else:
            super().__init__(*args, **kwargs)
        self.auto_resize = False

    def set_cv_image(self, img):
        if self.auto_resize:
            h, w = img.shape[:2]
        else:
            h, w = img.shape[:2] 
            dimg = ones((self.height,self.width,3),dtype='uint8')*255
            off_w = (self.width-w)//2
            off_h = (self.height-h)//2
            dimg[off_h:off_h+h,off_w:off_w+w] = img
            img = dimg
            h, w = self.height, self.width
            
        data = f'P6 {w} {h} 255 '.encode() + img[..., ::-1].tobytes()
        tkimg = tk.PhotoImage(width=w, height=h, data=data, format='PPM')
        self.create_image(0, 0, image=tkimg, anchor=tk.NW)
        self.image = tkimg
        if self.auto_resize:
            self.config(width=w,height=h)
        

class App(tk.Tk):
    def __init__(self, image_file, max_geom=None):
        super().__init__()

        self.image = cv2.imread(image_file)
        if max_geom:
            self.image = resize_prop(self.image, max_geom)
        h, w = self.image.shape[:2]

        # configure the root window
        self.title('Rotate image')
        #self.geometry('300x50')

        # image display
        diag = int(sqrt(h**2+w**2))
        self.imageContainer = CVImage(self, maxdim=(diag,diag))
        self.imageContainer.pack(fill=tk.Y)
        self.imageContainer.set_cv_image(self.image)
        
        #imageFrame.grid(row=0, column=0, padx=10, pady=2)

        # adjustment slider
        self.slider = ttk.Scale(self, from_=0, to=360, length=400, orient=tk.HORIZONTAL, command= self.slider_change)
        #self.slider.bind("<ButtonRelease-1>",self.slider_change)
        self.slider.pack(side=tk.LEFT)

        #status bar
        self.statusbar = tk.Label(self, text="Angle:   0; crop: (  0,  0)", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    def slider_change(self, event):
        angle = self.slider.get()
        if False:
            rot_img, crop = rotate_image(self.image, angle)

            imrot, cent = no_crop_rotate(self.image, angle)
            rotated_rect = (cent, (self.image.shape[1], self.image.shape[0]), angle)

            rect = calc_inner_rect(rotated_rect)
            imrot = cv2.rectangle(imrot, rect[:2], (rect[0]+rect[2], rect[1]+rect[3]), [255,0,0])
        else:
            imrot, crop = rotate_image(self.image,angle) 
        self.statusbar.config(text=f"Angle: {angle:3.0f}; crop: ({crop[0]:3d},{crop[1]:3d})")
        self.imageContainer.set_cv_image(imrot)

if __name__ == "__main__":
    app = App(sys.argv[1], max_geom=(600,400))
    app.mainloop()