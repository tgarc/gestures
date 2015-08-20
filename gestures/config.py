import numpy as np
import cv2
import pkg_resources


def get(*keys):
    return reduce(dict.__getitem__,keys,options)
    
model_parameters = {

    'MotionSegmenter' : {
        'alpha' : 0.5,
        'T0' : 10
    },

    'CrCbMeanShiftTracker' : {
        'nbins' : [16,16],
        'chans' : [1,2],
        'term_criteria' : ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ),
        'ranges' : [0, 256, 0, 256],
    },

    'GaussianSkinSegmenter' : {
        'cov' : np.array([[ 113.55502511,  -73.84680762],
                          [ -73.84680762,   75.83236121]]),
        'mu' : np.array([ 155.20978977,  104.60955366]),
        'threshold' : 0.0010506537825898023 * 0.5,
    },
    
    'ConvexityHandDetector' : {
        'angle_threshold' : ((np.pi/12),(np.pi/4)),
        'depth_threshold' : 0.5,
    },

    'SkinMotionSegmenter' : {
        'alpha' : 0.5,
    },

    'dollar' : {
        'scale' : 250,
        'samplesize' : 32,
    },
    
}

options = {
'model_parameters': model_parameters, 
'gesture_templates': pkg_resources.resource_filename('gestures'
                                                     ,'data/templates.hdf5'),
}
