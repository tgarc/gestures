import numpy as np
import cv2
from gestures.core.common import findBBoxCoM
from gestures.core.processor import Processor

class MotionSegmenter(Processor):
    '''
    A frame differencing based motion segmenter

    Parameters
    ----------
    alpha : array_like
        Sensitivity parameter. Higher values makes the background adapt quicker
        to changes.
    T0 : float
        Absolute lower threshold for motion. Anything less than this will always
        be ignored.
    '''
    def __init__(self,prev,curr,model_params={}):
        super(self.__class__, self).__init__(self.segment,**model_params)

        assert(prev.ndim == 2 and curr.ndim == 2)

        self.T = np.ones_like(prev)*self.T0

        self.background = np.zeros_like(prev)

        self._buff = (prev,curr)
        self.bbox = None
        self.com = None

    def segment(self,img):
        '''
        Parameters
        ----------
        img : array_like
            Grayscale image
        '''
        self.bbox = self.com = None

        prv,cur,nxt = self._buff + (img,)

        T = self.T
        bkgnd = self.background
        moving = (cv2.absdiff(prv,nxt) > T) & (cv2.absdiff(cur,nxt) > T)

        # if any motion was found, attempt to fill in the objects detected
        # TODO: needs to work with multiple independent objects
        if moving.any():
            self.bbox, self.com = findBBoxCoM(moving)
            x,y,w,h = self.bbox
            motionfill = cv2.absdiff(nxt[y:y+h,x:x+w],bkgnd[y:y+h,x:x+w]) > self.T0
            moving[y:y+h,x:x+w] |= motionfill

        # TODO replace boolean indexing with boolean multiply where possible
        # Updating threshold depends on current background model
        # so always update this before updating background
        T[~moving] = self.alpha*T[~moving] \
                          + (1-self.alpha)*5*cv2.absdiff(nxt,bkgnd)[~moving]
        # T[moving] = T[moving]
        T[T<self.T0] = self.T0

        bkgnd[~moving] = self.alpha*bkgnd[~moving] + (1-self.alpha)*nxt[~moving]
        bkgnd[moving] = nxt[moving]

        self._buff = (cur,nxt)
        
        return moving
