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

    def __init__(self,model_params={}):
        super(self.__class__, self).__init__(self.segment,**model_params)

        self.model = mvn(mean=self.mu,cov=self.cov)
        self.backprojection = None
        self.converted_image = None

    def segment(self,img):
        '''
        Parameters
        ----------
        img : array_like
            RGB color image in BGR channel order
        '''
        self.converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        self.backprojection = self.model.pdf(self.converted_image[...,1:]).reshape(img.shape[:2])

        return self.backprojection > self.threshold
