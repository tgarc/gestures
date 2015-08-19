from gestures.core.processor import Processor
import numpy as np
import cv2


class ConvexityHandDetector(Processor):
    '''
    Hand detector basecd on finding convexity defects of blobs in a binary image

    Parameters
    ----------
    angle_threshold : tuple
        Lower and upper threshold for convexity angle (angle between fingers)
    depth_threshold : float
        Lower convexity defect depth threshold (length of fingers) relative to
        largest defect depth.
    '''
    def __init__(self,model_params={}):
        super(self.__class__, self).__init__(self.detect,**model_params)

        self.contour = np.array([])
        self.hull = np.array([])
        self.dpoints = np.array([])

    def detect(self,bimg,mask=None):
        '''
        Attempts to find a hand-like object in a binary image, returning a
        binary decision. Also sets the 'contour','hull',and 'dpoints' attributes
        for the class instance. If a definite match is not found, these
        attributes will be set to the closest match available (or just empty
        arrays).

        Parameters
        ----------
        bimg : array_like
            Binary image created from some sort of thresholding processl

        Returns
        -------
        detected : bool
            Returns a binary decision of whether a definite hand match was found.
        '''
        detected = False
        self.hull = self.dpoints = self.contour = np.array([])

        contours,hierarchy = cv2.findContours(bimg.view(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return detected

        cntlens = np.array(map(len,contours))
        contours = np.array(contours)
        max_dpoints = []
        for cnt in contours[cntlens > 0.5*np.max(cntlens)]:
            hull_idx = cv2.convexHull(cnt,returnPoints=False)
            if len(hull_idx) <= 3: 
                continue

            defects = cv2.convexityDefects(cnt,hull_idx)
            if defects is None: 
                continue

            magmask = defects[:,0,3] > self.depth_threshold*np.max(defects[:,0,3])
            defects = defects[magmask,0,:]

            # TODO vectorize
            dpoints = []
            for (startpt,endpt,dpt,d) in defects:
                pt = cnt[dpt,0,:]

                a,b,c = cnt[[dpt,startpt,endpt],0,:]
                angle = np.abs(np.arccos(np.dot(b-a,c-a)/(np.linalg.norm(b-a)*np.linalg.norm(c-a))))

                # kpinfo = "(%.2f)" % (angle*180/np.pi)
                # cv2.putText(dispimg,kpinfo,tuple(pt),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0))

                if self.angle_threshold[0] <= angle < self.angle_threshold[1]:
                    dpoints.append(pt)

            if len(dpoints) > len(max_dpoints):
                max_dpoints = dpoints
                self.contour = cnt
                self.dpoints = dpoints
                self.hull = hull_idx

            if len(dpoints) == 3: # this is a definite positive detect
                detected = True
                break

        # make some conversions for user convenience
        if len(max_dpoints) > 0: 
            self.dpoints = np.array(self.dpoints)
            # convert hull point indices to the actual points for convenience
            self.hull = np.take(self.contour,self.hull.ravel(),axis=0).reshape(-1,1,2)

        return detected
