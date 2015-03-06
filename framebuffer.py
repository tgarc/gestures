import cv2
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
VERBOSE = 0


class FrameBufferBase(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def type(self):
        raise NotImplementedError

    @abstractmethod
    def read(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod        
    def reset(self):
        raise NotImplementedError        

    def __iter__(self):
        return self

    def next(self):
        img = self.read()
        if img.size: return img
        self.reset()
        raise StopIteration


class VideoBuffer(FrameBufferBase):
    def __init__(self,src,start=None,stop=None,**kwargs):
        self.start = start
        self.stop = stop

        self.__buff = cv2.VideoCapture(src,**kwargs)

        if isinstance(src,basestring):
            if self.start is None:
                self.start = int(self.__buff.get(cv2.CAP_PROP_POS_FRAMES))
            if self.stop is None:
                self.stop = int(self.__buff.get(cv2.CAP_PROP_FRAME_COUNT)
                                +self.__buff.get(cv2.CAP_PROP_POS_FRAMES))
            self.__buff.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        else:
            self.start = 0
            self.stop = -1

        self._idx = self.start

    def reset(self):
        self._idx = self.start
        self.__buff.set(cv2.CAP_PROP_POS_FRAMES, self.start)

    # allow user to indirectly access opencv VideoCapture object attributes
    def __getattr__(self,attr):
        return getattr(self.__buff, attr)

    def read(self):
        if self._idx == self.stop:
            return np.array([])

        valid, img = self.__buff.read()
        if valid and img is not None:
            self._idx += 1
            return img
        return np.array([])

    def close(self):
        self.__buff.release()


class ImageBuffer(FrameBufferBase):
    def __init__(self,src,start=None,stop=None,**kwargs):
        self.start = start
        self.stop = stop
        self.kwargs = dict(**kwargs)

        self.__buff = list(src)
        if self.stop is None: self.stop = len(self.__buff)
        if self.start is None: self.start = 0
        self._idx = self.start

    def reset(self):
        self._idx = self.start

    def read(self):
        if self._idx == self.stop:
            return np.array([])

        img = cv2.imread(self.__buff[self._idx],**self.kwargs)
        self._idx += 1
        return img
            
    def close(self):
        None

        
class FrameBuffer(object):
    def __init__(self,src,stop=None,start=None,**kwargs):
        """
        Class for handling images, video files, and cameras generically through
        opencv
        """
        if start is not None: start, stop = stop, start

        if hasattr(src,'__iter__'):
            buff = ImageBuffer
            self.source = 'image'
        elif isinstance(src,basestring):
            buff = VideoBuffer
            self.source = 'video'
        else:
            buff = VideoBuffer
            self.source = 'camera'            

        self.__cap = buff(src,start,stop,**kwargs)

        if VERBOSE:
            print "Opened %r (range=(%s,%s))" % (self.__cap,self.start,self.stop)

    def __iter__(self):
        return self

    def next(self):
        return self.__cap.next()
            
    def __getattr__(self,attr):
        return getattr(self.__cap, attr)

    def __repr__(self):
        return self.__cap.__repr__()

    def __str__(self):
        return self.__cap.__str__()
        
    def __enter__(self):
        return self

    def __exit__(self):
        self.close()
