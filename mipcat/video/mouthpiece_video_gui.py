from inspect import trace
import os
import sys
import cv2
import tkinter as tk
import numpy as np
import json
import traceback
from numpy import sqrt, ones, array
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
from . import template_trackers
from .template_trackers import calc_inner_rect, no_crop_rotate
from .template_trackers import rotate_image
from .mouthpiece_tracker import MouthpieceTracker

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
    new_w = max(1,int(w/fact))
    new_h = max(1,int(h/fact))
    return cv2.resize(image, (new_w, new_h)), fact


class CVImage(tk.Canvas):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.bind("<Configure>",self._resize_image)
        self.width = self.winfo_reqwidth()
        self.height = self.winfo_reqheight()
        self.frame=None

    def set_cv_image(self, img):
        self.image = img
        # resize the canvas 
        #self.config(width=self.width, height=self.height)
        self.redraw(self.width, self.height)

    def _set_raw_image(self, img):
        h, w = img.shape[:2] 
        data = f'P6 {w} {h} 255 '.encode() + img[..., ::-1].tobytes()
        self.tkimg = tk.PhotoImage(width=w, height=h, data=data, format='PPM')
        if self.frame:
            self.itemconfig(self.frame, image=self.tkimg)
        else:
            self.frame = self.create_image(0, 0, image=self.tkimg, anchor=tk.NW)

    def _resize_image(self,event):
        self.width = event.width
        self.height = event.height
        # resize the canvas 
        #self.config(width=self.width, height=self.height)

        self.redraw(self.width, self.height)

    def redraw(self, width, height):
        if width<=0 or height<=0:
            print("too small")
            return
        size = (width, height)
        img_r, self.scale_fact = resize_prop(self.image, size)
        self._set_raw_image(img_r)

    def canvas_to_img_coords(self, coords):
        try:
            len(coords)
            return np.array(coords)*self.scale_fact
        except AttributeError:
            return coords * self.scale_fact
    
    def img_to_canvas_coords(self, coords):
        try:
            len(coords)
            return np.array(coords)/self.scale_fact
        except AttributeError:
            return coords / self.scale_fact


class CVVideo(tk.Frame, MouthpieceTracker):
    def __init__(self, parent, video_file=None, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        MouthpieceTracker.__init__(self,*args, **kwargs)
        self.do_process = False
        self.time = 0
        
        # display
        self.image_container = CVImage(self)
        #self.image_container = tk.Label(self, text="hello")
        self.image_container.pack(fill=tk.BOTH, expand=True)
        
        # scroller
        self.position = tk.Scale(self, command=self._pos_change, orient=tk.HORIZONTAL)
        self.position.pack(fill=tk.X,side=tk.BOTTOM)


        # rectangle selector
        self.rect = None
        self.start_pt = None
        self.end_pt = None
        self.dragging = False
        self.image_container.bind("<Button-1>", self.__update, '+')
        self.image_container.bind("<B1-Motion>", self.__update, '+')
        self.image_container.bind("<ButtonRelease-1>", self.__stop)

        if video_file:
            self.set_video(video_file)

    def set_video(self,video_file):
        conffile = os.path.splitext(video_file)[0]+'_conf.json'
        self.set_video_source(video_file=video_file)
        self.position.configure(to=self.max_time)
        try:
            self.read_config(conffile)
        except Exception:
            traceback.print_exc()
        self.show()

    def show(self):
        #self.get_frame()
        self.destroy_temp()
        if self.do_process:
            self.process(self.img)
            self.image_container.set_cv_image(self.image_results())
            parent = self.master.master
            parent.statusbar.config(text=f"Area = {self.area}")
        else:
            self.image_container.set_cv_image(self.img)
    
    def show_hsv(self):
        self.color_convert()
        
        mm = self.green_mask
        ims = self.img/np.max(self.img)/2 
        ims[:,:,0] += mm/np.max(mm)/2

        ims = (ims*255).astype('uint8')
        self.image_container.set_cv_image(ims)

        
    def _pos_change(self, event):
        self.time = self.position.get()
        self.get_frame(time=self.time)
        self.show()
        
    def __update(self, event):
        if self.start_pt is None:
            self.start_pt = (event.x, event.y)
            self.end_pt = (event.x,event.y)
            self._selection_rect = self.image_container.create_rectangle(*self.start_pt,
                                                                         *self.end_pt,
                                                                         outline="red")
        elif self.end_pt is not None:
            self.end_pt = (event.x, event.y)
            self.image_container.coords(self._selection_rect,
                                       *self.start_pt,
                                        *self.end_pt)

    def __stop(self, event):
        self.image_container.delete(self._selection_rect)
        self._selection_rect = None
        self.rect = (*self.start_pt, *self.end_pt)
        self.start_pt = None
        self.end_pt = None

        rf = self.image_container.canvas_to_img_coords(self.rect)
        r = [int(x) for x in rf]
        parent = self.master.master
        parent.statusbar.config(text=f"Rect selection: {r[0]},{r[1]} -> {r[2]},{r[3]}")
        rect = [min(r[0], r[2]), min(r[1], r[3]), abs(r[2]-r[0]), abs(r[3]-r[1])]
        self.set_template_from_time(rect, self.time)
        #self.image_container.set_cv_image(self.template)
        self.draw_rect(rect)

    def draw_rect(self, rect, color="red"):
        """Draw rectangle in image coords

        Args:
            rect (_type_): _description_
            color (_type_): _description_
        """
        crect = self.image_container.img_to_canvas_coords(rect)
        self.temp_rect = self.image_container.create_rectangle(crect[0], crect[1], crect[0]+crect[2], crect[1]+crect[3], outline=color, tags=["temp"])

    def destroy_temp(self):
        self.image_container.delete("temp")

    def set_template_from_time(self, rect, time=0):
        super().set_template_from_time(rect, time)
        self.master.master.show_template(self.template)



class HiLoController(tk.Frame):
    def __init__(self, parent, text="", text_width=None,
                 slider_width=200, min=0, max=255, 
                 command=None, *args, **kwargs):
        super().__init__(parent,  *args, **kwargs)
    
        self.label = tk.Label(self, text=text, width=text_width)
        self.label.pack(side=tk.LEFT)

        # sliders
        self.slider_container = tk.Frame(self)
        self.slider_container.pack()
        
        self.low = tk.Scale(self.slider_container,
                                orient=tk.HORIZONTAL, 
                                length=slider_width,
                                from_=min, to=max, 
                                command=command)

        self.high = tk.Scale(self.slider_container,
                                orient=tk.HORIZONTAL, 
                                length=slider_width,
                                from_=min, to=max, 
                                command=command)
        self.high.pack()

        self.low.pack()

        
class HSVControls(tk.Frame):
    def __init__(self, parent, text_width=10, slider_width=300,
                 command=None, *args, **kwargs):
        super().__init__(parent, bg="#ffa0a0", *args, **kwargs)
        
        # hue
        self.hue = HiLoController(self, text="Hue", 
                                 text_width=text_width,
                                 slider_width=slider_width, 
                                 command=command,
                                 max=179)
        self.hue.pack()

        # saturation
        self.saturation = HiLoController(self, text="Saturation", 
                                        text_width=text_width, 
                                        slider_width=slider_width, 
                                        command=command,
                                        max=255)
        self.saturation.pack()

        # value
        self.value = HiLoController(self, text="Value",
                                    text_width=text_width,
                                    slider_width=slider_width, 
                                    command=command,
                                    max=255)
        self.value.pack()
        
    def get_vals(self):
        hvals = (self.hue.low.get(), self.hue.high.get())
        svals = (self.saturation.low.get(), self.saturation.high.get())
        vvals = (self.value.low.get(), self.value.high.get())
        return([hvals, svals, vvals])
    
    def from_processor(self, fp):
        self.hue.low.set(fp.lower_color[0])
        self.hue.high.set(fp.upper_color[0])
        self.saturation.low.set(fp.lower_color[1])
        self.saturation.high.set(fp.upper_color[1])
        self.value.low.set(fp.lower_color[2])
        self.value.high.set(fp.upper_color[2])
        
    
class PushButtons(tk.Frame):
    def __init__(self, parent, command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

         
        self.template_display = tk.Canvas(self, width=50,height=50)
        self.template_display.pack()
        

        self.process_var = tk.BooleanVar(value=False)
        self.process_check = tk.Checkbutton(self,text="Process", 
                                            variable = self.process_var,
                                            command=lambda: self.on_change("process"))
        self.process_check.pack()
        
        self.row1 = tk.Frame(self)
        self.row1.pack()

        self.write_button = tk.Button(self.row1, text="Write",
                                      command=lambda: self.on_change("write"))
        self.write_button.pack(side=tk.LEFT)

        self.print_button = tk.Button(self.row1, text="Print",
                                      command=lambda: self.on_change("print"))
        self.print_button.pack()

        self.row2 = tk.Frame(self)
        self.row2.pack()
        self.back_button = tk.Button(self.row2, text="Back to Ref.",
                                      command=lambda: self.on_change("back"))
        self.back_button.pack(side=tk.LEFT)
        
        self.find_button = tk.Button(self.row2, text="Find ref",
                                      command=lambda: self.on_change("find"))
        self.find_button.pack()

        self.frame = None
        self.command = command

    def on_change(self, element):
        arg = None
        if element == "process":
            arg = self.process_var.get()
        self.command(element,arg)

    def show_image(self, img):
        
        h, w = img.shape[:2] 
        data = f'P6 {w} {h} 255 '.encode() + img[..., ::-1].tobytes()
        self.tkimg = tk.PhotoImage(width=w, height=h, data=data, format='PPM')
        if self.frame:
            self.template_display.itemconfig(self.frame, image=self.tkimg)
        else:
            self.frame = self.template_display.create_image(0, 0, image=self.tkimg, anchor=tk.NW)



class App(tk.Tk):
    def __init__(self, video_file):
        super().__init__()

        self.video_file = video_file
        
        # configure the root window
        self.title('Mouthpiece configurator')
        self.geometry('800x400')

        # widget container
        self.work_area = tk.Frame(self)
        self.work_area.pack( fill=tk.BOTH, expand=True)

        # video display
        self.video_container = CVVideo(self.work_area) 
        self.video_container.pack(expand=True, fill=tk.BOTH, 
                                  side=tk.LEFT)
        
        ### controls
        self.controller_frame = tk.Frame(self.work_area, width=400)
        self.controller_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        self.hsv_controls = HSVControls(self.controller_frame, 
                                        command=self.hsv_change)
        self.hsv_controls.pack()
        self.hsv_controls.from_processor(self.video_container)

        self.push_buttons = PushButtons(self.controller_frame, 
                                        command=self.button_dispatcher)
        self.push_buttons.pack(side=tk.BOTTOM)
        
        #status bar
        self.statusbar = tk.Label(self, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.video_container.set_video(video_file)
        self.hsv_controls.from_processor(self.video_container)

    def hsv_change(self, event):
        hsv_vals = self.hsv_controls.get_vals()
        self.video_container.lower_color = array([x[0] for x in hsv_vals])
        self.video_container.upper_color = array([x[1] for x in hsv_vals])
        self.video_container.show_hsv()

    def button_dispatcher(self, element, arg):
        if element == "process":
            print(arg)
            self.video_container.do_process = arg
            self.video_container.show()
        elif element == "print":
            js = self.video_container.config_to_json()
            for k, v in js.items():
                print(k, v, type(v))
            

        elif element == "write":
            js = self.video_container.config_to_json()
            outfile = os.path.splitext(self.video_file)[0]+'_conf.json'
            with open(outfile,'w') as f:
                json.dump(js, f)
                self.send_status('Config written to {}'.format(outfile))

        elif element == "find":
            templ_fn = '/'.join(os.path.abspath(template_trackers.__file__).split('\\')[:-3])+'/resources/template_25.png'
            template = cv2.imread(templ_fn)
            h, w = template.shape[:2]
            self.send_status("Finding template...")
            trk = template_trackers.MultiAngleTemplateTracker(template, angle_step=15, size_fact=1.3, n_size=5)
            cent, angle = trk.match(self.video_container.img)
            cent = [int(c) for c in cent]
            trect = [int(cent[0]-w/2), int(cent[1]-h/2), w, h]
            self.video_container.set_template_from_time(trect, self.video_container.time)
            self.video_container.draw_rect(trect)

            self.send_status(f"Found template at ({cent[0]}, {cent[1]})")

    def send_status(self, text):
        self.statusbar.config(text=text)

    def show_template(self, cv_image):
        self.push_buttons.show_image(cv_image)


if __name__ == "__main__":
    app = App(sys.argv[1])
    app.mainloop()