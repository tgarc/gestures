import numpy as np
import cv2
from gestures.skin_segmentation import GaussianSkinSegmenter
from gestures.motion_segmentation import MotionSegmenter
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
        self.backproject = None
        self.bbox = None
        self.com = None

    def segment(self,img,fill=False):
        self.bbox = self.com = self.backproject = None
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ycc = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)

        motion = self.moseg.segment(gray,fill=fill)
        if self.moseg.bbox is None:
            return np.array([])
        x,y,w,h = self.moseg.bbox

        skin = self.coseg.segment(ycc)
        sprobImg = cv2.normalize(self.coseg.backproject,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)

        self.backproject = np.zeros_like(gray,dtype=float)
        self.backproject[y:y+h,x:x+w] = 0.5*sprobImg[y:y+h,x:x+w] + 0.5*motion[y:y+h,x:x+w]

        fused = np.zeros_like(gray,dtype=bool)
        fused[y:y+h,x:x+w] = skin[y:y+h,x:x+w] # | motion[y:y+h,x:x+w]

        if fused.any():
            self.bbox,self.com = findBBoxCoM(fused)
        
        return fused
