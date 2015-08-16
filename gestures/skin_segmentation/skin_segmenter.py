import numpy as np
import cv2

from scipy.stats import multivariate_normal as mvn

class GaussianSkinSegmenter(object):
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
    cov = np.array([[ 113.55502511,  -73.84680762],
                    [ -73.84680762,   75.83236121]])

    mu = np.array([ 155.20978977,  104.60955366])
    threshold = 0.0010506537825898023

    def __init__(self,**params):
        for k in set(('mu','cov','threshold')).intersection(params.keys()):
            setattr(self,k,params[k])
        self.model = mvn(mean=self.mu,cov=self.cov)

    @property
    def params(self):
        return {'mu':self.mu,'cov':self.cov,'threshold':self.threshold}

    def __call__(self,*args,**kwargs):
        return self.segment(*args,**kwargs)

    def segment(self,ycc):
        return self.model.pdf(ycc[...,1:]).reshape(ycc.shape[:2]) > self.threshold
