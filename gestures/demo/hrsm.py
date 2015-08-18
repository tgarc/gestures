import cv2
import numpy as np
import h5py

from gestures.gesture_classifier import dollar
from gestures.fused_segmentation import SkinMotionSegmenter
from gestures.hand_detection import ConvexityHandDetector
from gestures.tracking import MeanShiftTracker
from gestures.config import scale, samplesize
from abc import ABCMeta, abstractmethod


# global constants
WAIT_PERIOD = 5
VAL_PERIOD = 1

MAXPORTION = 2
MINSKINPORTION = 8
MINWAYPTS = 10
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
chans = [1,2]
ranges = [0, 256, 0, 256]
nbins = [16,16]


class StateMachineBase(object):
    __metaclass__ = ABCMeta

    def __init__(self,init_state):
        self._state = init_state

    @property
    def state(self):
        return self._state.__name__

    def tick(self,*args):
        newstate = self._state(*args) or self._state
        self._state = newstate
    
        
class HandGestureRecognizer(StateMachineBase):

    def __init__(self,prev,curr,templates,callback=None,match_threshold=0.8):
        """
        State machine for hand gesture recognition.

        Callback is called with the parameters:

        query : array_like
            The set of (x,y) points that represent the hand path
        template : array_like
            The best matching template gesture
        score : float
            The matching score (in range [0,1] with 1 being perfect match)
        theta : float
            The rotation of the gesture that gave the best match
        clsid : int
            The ID of the template class that matched best

        Parameters
        ----------
        imshape : tuple
            The dimensions tuple of the image stream
        templates : string, h5py.File
            h5py.File instance or path to the h5 dataset of template gestures to use
        callback : function
            Callback function called when a gesture match is made
        """
        super(self.__class__,self).__init__(self.Wait)

        self.callback = callback
        self.match_threshold = match_threshold

        if isinstance(templates,basestring):
            self.template_ds = h5py.File(templates,'r')
        else:
            self.template_ds = templates

        self.segmenter = SkinMotionSegmenter(prev,curr,scale=0.25)
        self.detector = ConvexityHandDetector()
        self.tracker = MeanShiftTracker()

        self.imshape = prev.shape
        self.counter = 0
        self.waypts = []

    def Wait(self,img):
        self.counter = (self.counter+1) % WAIT_PERIOD
        if self.counter == 0:
            return self.Search

    def Search(self,img):
        mask = self.segmenter(img)
        if mask.size and self.detector(mask):
            return self.Validate

    def Validate(self,img):
        mask = self.segmenter(img)

        if not self.detector(mask):
            self.counter = 0
            return self.Search

        self.counter = (self.counter+1) % VAL_PERIOD

        if self.counter == 0:
            self.tracker.init(img,self.segmenter.bbox)
            self.waypts = [self.segmenter.com]

            return self.Track
            
    def Track(self,img):
        mask = self.segmenter(img)

        if np.sum(mask) > 100:
            # probably needs some error checking here
            x,y,w,h = self.tracker.track(img)
            self.waypts.append((x+w//2,y+h//2))
        else:
            if len(self.waypts) > MINWAYPTS and self.callback is not None:
                # Find best gesture match
                x,y = zip(*[(self.imshape[1]-x,self.imshape[0]-y) for x,y in self.waypts])
                matches = dollar.query(x,y,scale,samplesize,self.template_ds)
                score,theta,clsid = matches[0]

                if score > self.match_threshold:
                    ds = self.template_ds[clsid][0]
                    self.callback((x,y),(ds['x'],ds['y']),score,theta,clsid)
            self.waypts = []
            return self.Wait
