import numpy as np
import cv2
from gestures.core.common import findBBoxCoM

class MotionSegmenter(object):
    '''
    A frame differencing based motion segmenter

    Parameters
    ----------
    alpha : array_like
        Covariance matrix
    T0 : float
        Absolute lower threshold for motion. Anything less than this will be
        ignored.
    '''

    alpha=0.5
    T0=10

    def __init__(self,prev,curr,**params):
        assert(prev.ndim == 2 and curr.ndim == 2)

        for k in set(('alpha','T0')).intersection(params.keys()):
            setattr(self,k,params[k])

        self.T = np.ones_like(prev)*self.T0

        self.background = prev.copy()
        self._buff = (prev,curr)
        self.backproject = None
        self.bbox = None
        self.com = None
        self.mkernel = np.ones((3,3),dtype=np.uint8)

    @property
    def params(self):
        return {'alpha':self.alpha,'T0':self.T0}

    def __call__(self,*args,**kwargs):
        return self.segment(*args,**kwargs)

    def segment(self,img,fill=False):
        prv,cur,nxt = self._buff + (img,)

        T = self.T
        bkgnd = self.background
        moving = (cv2.absdiff(prv,nxt) > T) & (cv2.absdiff(cur,nxt) > T)

        cv2.morphologyEx(moving.view(np.uint8),cv2.MORPH_CLOSE,self.mkernel,dst=moving.view(np.uint8))

        # if any motion was found, attempt to fill in the objects detected
        # TODO: needs to work with multiple independent objects
        try:
            self.bbox, self.com = findBBoxCoM(moving)
        except ValueError:
            self.bbox = None
            self.com = None
        else:
            if fill:
                x,y,w,h = self.bbox
                motionfill = (cv2.absdiff(cur,bkgnd) > 0.1*T)[y:y+h,x:x+w].view(np.uint8)
                cv2.morphologyEx(motionfill,cv2.MORPH_CLOSE,self.mkernel,dst=motionfill)
                moving[y:y+h,x:x+w] |= motionfill

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
