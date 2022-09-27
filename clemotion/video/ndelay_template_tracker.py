import cv2

class NDelayTemplateTracker(object):
    def __init__(self, delays=[1]):
        self.templates = []
        