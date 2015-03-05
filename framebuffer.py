import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
VERBOSE = 0


class FrameBufferBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

class VideoBuffer(FrameBufferBase):
    def __init__(self,src,start=None,stop=None):
        self.start = start
        self.stop = stop

        self.__buff = cv2.VideoCapture(src)

        if isinstance(src,basestring):
            if self.start is None:
                self.start = int(self.__buff.get(cv2.CAP_PROP_POS_FRAMES))
            if self.stop is None:
                self.stop = int(self.__buff.get(cv2.CAP_PROP_FRAME_COUNT)+self.__buff.get(cv2.CAP_PROP_POS_FRAMES))
            self.__buff.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        else:
            self.start = 0
            self.stop = -1

        self._idx = self.start-1

    def read(self):
        self._idx += 1
        if self._idx == self.stop:
            return np.array([])

        valid, img = self.__buff.read()
        if valid:
            return img
        return np.array([])

    def close(self):
        self.__buff.release()


class ImageBuffer(FrameBufferBase):
    def __init__(self,src,start=None,stop=None):
        self.start = start
        self.stop = stop

        self.__buff = (cv2.imread(fn,**kwargs) for fn in list(src)[start:stop])

    def read(self):
        try:
            return self.__buff.next()
        except StopIteration:
            return np.array([])

    def close(self):
        None

        
class FrameBuffer(object):
    def __init__(self,src,stop=None,start=None):
        """
        Class for handling video files/streams generically through opencv's video captures class
        """
        if start is not None: start, stop = stop, start

        if hasattr(src,'__iter__'):
            buff = ImageBuffer
        else:
            buff = VideoBuffer
        self.__cap = buff(src,start,stop)

        if VERBOSE:
            print "Opened %r (range=(%s,%s))" % (self.__cap,self.start,self.stop)
        
    def __getattr__(self,attr):
        return getattr(self.__cap, attr)

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

