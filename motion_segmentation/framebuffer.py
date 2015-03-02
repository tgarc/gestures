import cv2
import numpy as np
VERBOSE = 0

class FrameBuffer(object):
    def __init__(self,src,stop=None,start=None):
        if start is not None: start, stop = stop, start
        self.cap = cv2.VideoCapture(src)
        self.is_video = isinstance(src,basestring)
        self.name = str(src)
        
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.dim = (h,w)

        self.start = start
        self.stop = stop
        if self.is_video:
            if self.start is None:
                self.start = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if self.stop is None:
                self.stop = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)+self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        else:
            self.start = 0
            self.stop = -1
        self.idx = self.start-1

        if VERBOSE:
            print "Opened %r (name='%s',range=(%s,%s),dim=%s)" % (self.cap,self.name,self.start,self.stop,self.dim)

    def read(self):
        self.idx += 1
        if self.is_video and self.idx == self.stop:
            return False, np.array([])

        return self.cap.read()

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()
