import cv2
import numpy as np
import common as cmn
import h5py
from gesture_classifier import dollar
from config import scale, samplesize
from abc import ABCMeta, abstractmethod


# global constants
WAIT_PERIOD = 5
VAL_PERIOD = 3

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
    
        
def estimateHand(skin,move_bbox):
    # Estimate hand centroid as the centroid of skin colored pixels inside
    # the bbox of detected movement
    hand_bbox,(xcom,ycom) = cmn.findBBoxCoM(skin,move_bbox)

    # Use the hand centroid estimate as our initial estimate for
    # tracking Estimate the hand's bounding box by taking the minimum
    # vertical length to where the skin ends. If the hand is vertical,
    # this should correspond to the length from the palm to tip of
    # fingers
    y = min(hand_bbox[1],move_bbox[1])
    h = 2*(ycom-y)
    x = min(hand_bbox[0],move_bbox[0])
    w = max(move_bbox[0]+move_bbox[2],hand_bbox[0]+hand_bbox[2])-x

    return (x,y,w,h),(xcom,ycom)


class HandGestureRecognizer(StateMachineBase):

    def __init__(self,imshape,templates,callback=None,match_threshold=0.8):
        """
        State machine for hand gesture recognition.

        imshape - the dimensions tuple of the image stream
        templates - the dataset of template gestures
        callback - callback function called when a gesture match is made

        Callback is called with the parameters:

        query - the set of (x,y) points that represent the hand path
        template - the best matching template gesture
        score - the matching score (in range [0,1] with 1 being perfect match)
        theta - the rotation of the gesture that gave the best match
        clsid - the ID of the template class that matched best
        """
        super(self.__class__,self).__init__(self.Wait)

        self.callback = callback
        self.match_threshold = match_threshold
        if isinstance(templates,basestring):
            self.template_ds = h5py.File(templates,'r')
        else:
            self.template_ds = templates

        self.imshape = imshape
        self.blobthresh = (min(imshape)//MINSKINPORTION)**2
        self.maxarea = imshape[0]*imshape[1]//MAXPORTION
        self.hist = np.zeros(nbins,dtype=np.float)
        self.backproject = np.zeros(imshape,dtype=np.float)
        self.track_bbox = 0,0,imshape[1],imshape[0]
        self.counter = 0
        self.waypts = []

    def Wait(self,motion,skin,move_bbox,img_crcb):
        self.counter = (self.counter+1) % WAIT_PERIOD
        if self.counter == 0:
            return self.Search

    def Search(self,motion,skin,move_bbox,img_crcb):
        x,y,w,h = move_bbox
        skinarea = np.sum(skin[y:y+h,x:x+w])

        if not skinarea: return None

        (x,y,w,h),(xcom,ycom) = estimateHand(skin,move_bbox)
        ecc = w/float(h)

        if (skinarea > self.blobthresh and w*h < self.maxarea
            and 0.4 <= ecc and ecc <= 1.1):
            self.track_bbox = x,y,w,h
            self.counter = 1

            return self.Validate

    def Validate(self,motion,skin,move_bbox,img_crcb):
        try:
            (x,y,w,h),(xcom,ycom) = estimateHand(skin,move_bbox)
        except ValueError:
            return None

        lastarea = self.track_bbox[2]*self.track_bbox[3]
        if abs(w*h-lastarea)/lastarea < 0.1:
            self.track_bbox = x,y,w,h
            self.counter = (self.counter+1) % VAL_PERIOD
        else:
            self.counter = 0
            return self.Search

        if self.counter == 0:
            # Use the skin bbox color histogram to initiate tracking
            crcb_roi = img_crcb[y:y+h,x:x+w]
            skin_roi = skin[y:y+h,x:x+w]

            # Normalize to 1 to get the sample PDF
            self.hist = cv2.calcHist([crcb_roi], chans, skin_roi.view(np.uint8), nbins, ranges)
            cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)

            self.waypts = [(xcom,ycom)]
            return self.Track
            
    def Track(self,motion,skin,move_bbox,img_crcb):
        x,y,w,h = move_bbox
        if np.sum(skin[y:y+h,x:x+w]) > self.blobthresh:
            bkproject = cv2.calcBackProject([img_crcb],chans,self.hist,ranges,1)
            bkproject &= motion
            bkproject[y+self.track_bbox[-1]:,] = 0
            self.backproject = bkproject

            # notice we're using the track_bbox from last iteration
            # for the intitial estimate
            niter, self.track_bbox = cv2.meanShift(bkproject,self.track_bbox,term_crit)
            x,y,w,h = self.track_bbox

            try:
                xcom,ycom = cmn.findBBoxCoM(skin,self.track_bbox)[1]
            except ValueError:
                xcom,ycom = x+w//2,y+h//2
            self.waypts.append((xcom,ycom))
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
