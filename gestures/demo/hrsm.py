import cv2
import numpy as np
import h5py

from gestures.gesture_classification import dollar
from gestures.segmentation import SkinMotionSegmenter
from gestures.hand_detection import ConvexityHandDetector
from gestures.tracking import CrCbMeanShiftTracker
from gestures.config import scale, samplesize
from gestures.core.common import findBBoxCoM_contour,findBBoxCoM
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
        self.tracker = CrCbMeanShiftTracker()

        self.imshape = prev.shape
        self.counter = 0
        self.waypts = []

    @property
    def backprojection(self):
        if self.state == 'Track' and self.tracker.backprojection is not None:
            return self.tracker.backprojection
        return self.segmenter.backprojection

    def Wait(self,img):
        self.counter = (self.counter+1) % WAIT_PERIOD
        if self.counter == 0:
            return self.Search

    def Search(self,img):
        mask = self.segmenter.segment(img)
        if self.detector(mask):
            # return self.Validate
            bbox,com = findBBoxCoM_contour(self.detector.contour)

            # use the hand contour to draw out a more precise mask of the crcb
            # image
            mask = np.zeros_like(mask)
            cv2.drawContours(mask.view(np.uint8),[self.detector.contour],0,1,thickness=-1)
            colorimg = self.segmenter.coseg.converted_image * mask.reshape(mask.shape+(1,))

            self.tracker.init(colorimg,bbox)
            self.waypts = [com]
            self.tracker.backprojection = mask # this is a dirty little trick >:)

            return self.Track


    def Validate(self,img):
        mask = self.segmenter.segment(img)

        if not self.detector(mask):
            self.counter = 0
            return self.Search

        self.counter = (self.counter+1) % VAL_PERIOD

        if self.counter == 0:
            bbox,com = findBBoxCoM_contour(self.detector.contour)
            self.tracker.init(self.segmenter.coseg.converted_image,bbox)
            self.waypts = [com]

            return self.Track
            
    def Track(self,img):
        mask = self.segmenter.segment(img)

        # probably needs some error checking here in case tracking fails
        if np.sum(mask) > 500:
            x,y,w,h = self.tracker.track(self.segmenter.coseg.converted_image,mask=mask)
            bbox,(xc,yc) = findBBoxCoM(self.tracker.backprojection)
            self.waypts.append((int(xc),int(yc)))
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
