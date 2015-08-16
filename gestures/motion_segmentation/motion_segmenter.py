import numpy as np
import cv2

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

    @property
    def params(self):
        return {'alpha':self.alpha,'T0':self.T0}

    def __call__(self,*args,**kwargs):
        return self.segment(*args,**kwargs)

    def segment(self,img):
        prv,cur,nxt = self._buff + (img,)

        T = self.T
        bkgnd = self.background
        moving = (cv2.absdiff(prv,nxt) > T) & (cv2.absdiff(cur,nxt) > T)

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
