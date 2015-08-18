import numpy as np
import cv2
from skin_segmenter import GaussianSkinSegmenter
from motion_segmenter import MotionSegmenter
from gestures.core.common import findBBoxCoM
from gestures.core.processor import Processor

class SkinMotionSegmenter(Processor):
    '''
    Hand detector basecd on finding convexity defects of blobs in a binary image

    Parameters
    ----------
    '''
    _params = {}

    def __init__(self,prev,curr,scale=1,skin_params={},motion_params={}):
        super(self.__class__, self).__init__(self.segment)

        self.coseg = GaussianSkinSegmenter(scale=scale,model_params=skin_params)
        self.moseg = MotionSegmenter(prev,curr,model_params=motion_params)

        self.bbox = None
        self.com = None
        self.backproject = None
        self.skin = self.motion = np.zeros_like(prev)

    def segment(self,img):
        self.bbox = self.com = None

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.motion = self.moseg.segment(gray)
        self.skin = self.coseg.segment(img)

        if self.moseg.bbox is None:
            return self.motion

        x,y,w,h = self.moseg.bbox
        sprobImg = cv2.normalize(self.coseg.backproject,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
        self.backproject = 0.5*sprobImg + 0.5*self.motion

        fused = np.zeros_like(gray,dtype=bool)
        fused[y:y+h,x:x+w] = self.skin[y:y+h,x:x+w] # | motion[y:y+h,x:x+w]
        if fused.any():
            self.bbox,self.com = findBBoxCoM(fused)

        return fused
