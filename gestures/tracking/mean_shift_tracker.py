from gestures.core.processor import Processor
import numpy as np
import cv2


class CrCbMeanShiftTracker(Processor):
    '''
    A wrapper for the OpenCV implementation of mean shift tracking. See opencv
    docs for more information abouts parameters.

    Parameters
    ----------
    chans : array_like
    nbins : array_like
    term_criteria : tuple
    ranges : array_like
    '''
    def __init__(self,model_params={}):
        super(self.__class__, self).__init__(self.track,**model_params)

        self.bbox = None
        self.hist = np.zeros(self.nbins,dtype=int)
        self.backprojection = None

    def track(self,img,mask=None):
        self.backprojection = cv2.calcBackProject([img],self.chans,self.hist,self.ranges,1)
        if mask is not None: 
            self.backprojection *= mask # this is way faster than x[~mask] = 0

        niter, self.bbox = cv2.meanShift(self.backprojection,self.bbox,self.term_criteria)

        return self.bbox

    def init(self,img,bbox,mask=None):
        self.bbox = tuple(bbox)
        x,y,w,h = self.bbox
        roi = img[y:y+h,x:x+w]

        if mask is not None:
            mask = mask[y:y+h,x:x+w]

        self.hist = cv2.calcHist([roi], self.chans, mask, self.nbins, self.ranges)
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
