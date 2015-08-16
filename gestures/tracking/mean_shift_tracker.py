import numpy as np
import cv2


class MeanShiftTracker(object):
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
    nbins = [16,16]
    chans = [1,2]
    term_criteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    ranges = [0, 256, 0, 256]

    def __init__(self,**params):
        for k in set(('nbins','chans','term_criteria','ranges')).intersection(params.keys()):
            setattr(self,k,params[k])
        self.bbox = None
        self.hist = np.zeros(nbins,dtype=int)
        self.backproject = None

    @property
    def params(self):
        return {'nbins':self.nbins
                ,'chans':self.chans
                ,'term_criteria':self.term_criteria
                ,'ranges':self.ranges}

    def __call__(self,*args,**kwargs):
        return self.track(*args,**kwargs)

    def track(self,img,mask=None):
        self.backproject = cv2.calcBackProject([img],chans,self.hist,self.ranges,1)
        if mask is not None: 
            self.backproject &= mask

        niter, self.bbox = cv2.meanShift(self.backproject,self.bbox,self.term_criteria)

        return self.bbox

    def init(self,img,bbox,mask=None):
        self.bbox = tuple(bbox)
        x,y,w,h = self.bbox
        roi = img[y:y+h,x:x+w]

        if mask is not None:
            mask = mask[y:y+h,x:x+w]

        self.hist = cv2.calcHist([roi], self.chans, mask, self.nbins, self.ranges)
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
