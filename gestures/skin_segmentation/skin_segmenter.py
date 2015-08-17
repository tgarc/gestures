from gestures.core.processor import Processor
import numpy as np
import cv2
from scipy.stats import multivariate_normal as mvn


class GaussianSkinSegmenter(Processor):
    '''
    A Gaussian YCrCb skin color model

    Parameters
    ----------
    cov : array_like
        covariance matrix
    mu : array_like
        mean vector
    threshold : float
        probability threshold to use for segmentation
    '''

    _params = {'cov' : np.array([[ 113.55502511,  -73.84680762],
                                 [ -73.84680762,   75.83236121]])
               ,'mu' : np.array([ 155.20978977,  104.60955366])
               ,'threshold' : 0.0010506537825898023}

    def __init__(self,scale=1,model_params={}):
        super(self.__class__, self).__init__(self.segment,**model_params)

        self.model = mvn(mean=self.mu,cov=self.cov)
        self.backproject = None
        self.threshold = scale*self.threshold

    def segment(self,ycc):
        self.backproject = self.model.pdf(ycc[...,1:]).reshape(ycc.shape[:2])

        mask = self.backproject > self.threshold
        cv2.medianBlur(mask.view(np.uint8),3,dst=mask.view(np.uint8))
        return mask
