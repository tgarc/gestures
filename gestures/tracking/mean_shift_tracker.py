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
        self.hist = np.zeros(self.nbins,dtype=np.float32)
        self.backprojection = None

    def track(self,img,mask=None):
        self.backprojection = cv2.calcBackProject([img],self.chans,self.hist,self.ranges,1/self.hist.max())

        if mask is not None: 
            self.backprojection *= mask

        niter, self.bbox = cv2.meanShift(self.backprojection,self.bbox,self.term_criteria)

        return self.bbox

    def init(self,img,bbox,mask=None,update=False):
        self.bbox = tuple(bbox)
        x,y,w,h = self.bbox
        roi = img[y:y+h,x:x+w]

        if mask is not None:
            mask = mask[y:y+h,x:x+w].view(np.uint8)

        # bug in opencv doesn't allow assignment to a preallocated histogram in
        # the general case; this is a workaround for that
        if update:
            self.hist += cv2.calcHist([roi], self.chans, mask, self.nbins, self.ranges)
        else:
            self.hist = cv2.calcHist([roi], self.chans, mask, self.nbins, self.ranges)
