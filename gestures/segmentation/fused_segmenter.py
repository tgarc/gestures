import numpy as np
import cv2
from skin_segmenter import GaussianSkinSegmenter
from motion_segmenter import MotionSegmenter
from gestures.core.common import findBBoxCoM
from gestures.core.processor import Processor

class SkinMotionSegmenter(Processor):
    '''
    Blends skin and motion segmentation together to get more accurate
    segmentation

    Parameters
    ----------
    alpha : float
        Skin-to-motion blending parameter; i.e., fusion = w*skin + (1-w)*motion
    '''

    def __init__(self,prev,curr,fusion_params={},skin_params={},motion_params={}):
        super(self.__class__, self).__init__(self.segment,**fusion_params)

        self.coseg = GaussianSkinSegmenter(model_params=skin_params)
        self.moseg = MotionSegmenter(prev,curr,model_params=motion_params)

        self.bbox = None
        self.com = None
        self.backprojection = None
        self.skin = None
        self.motion = None

    def segment(self,img):
        self.bbox = self.com = None

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.motion = self.moseg.segment(gray)
        self.skin = self.coseg.segment(img)

        if self.moseg.bbox is None:
            return self.motion

        x,y,w,h = self.moseg.bbox
        sprobImg = cv2.normalize(self.coseg.backprojection[y:y+h,x:x+w],alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
        self.backprojection = np.zeros_like(gray,dtype=float)
        self.backprojection[y:y+h,x:x+w] = self.alpha*sprobImg + (1-self.alpha)*self.motion[y:y+h,x:x+w]

        fused = np.zeros_like(gray,dtype=bool)
        fused[y:y+h,x:x+w] = self.skin[y:y+h,x:x+w] | self.motion[y:y+h,x:x+w]
        if fused.any():
            self.bbox,self.com = findBBoxCoM(fused)

        return fused
