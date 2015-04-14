import cv2
import numpy as np
from abc import ABCMeta, abstractmethod


class FrameBufferBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self):
        """
        Returns the next image in the feed.

        If end of feed is reached, an empty numpy array is returned.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def seek(self,offset):
        """
        If possible with the current stream, move to new frame index.
        """
        raise NotImplementedError

    def __iter__(self):
        return self

    def next(self):
        img = self.read()
        if img.size: return img
        raise StopIteration


class VideoBuffer(FrameBufferBase):
    """
    Wrapper for the opencv video capture interface
    """
    def __init__(self,src,start=None,stop=None,**kwargs):
        self.start = start
        self.stop = stop
        self._idx = self.start

        self.__buff = cv2.VideoCapture(src,**kwargs)

        if isinstance(src,basestring):
            if self.start is None:
                self.start = int(self.__buff.get(cv2.CAP_PROP_POS_FRAMES))
            if self.stop is None:
                self.stop = int(self.__buff.get(cv2.CAP_PROP_FRAME_COUNT)
                                +self.__buff.get(cv2.CAP_PROP_POS_FRAMES))
            self.__buff.set(cv2.CAP_PROP_POS_FRAMES, self.start)
            self.live = False
        else:
            self.start = 0
            self.stop = np.inf # '-1' is reserved
            self.live = True

        self._shape = self.__buff.read()[1].shape
        self.seek()

    def seek(self,frame_index=None):
        if not self.live:
            self._idx = frame_index if frame_index is not None else self.start
            self.__buff.set(cv2.CAP_PROP_POS_FRAMES, self.start)

    def read(self):
        if self._idx == self.stop:
            return np.array([],dtype=np.void)

        valid, img = self.__buff.read()
        if valid:
            self._idx += 1
            return img

        return np.array([],dtype=np.void)

    def __repr__(self):
        return "%r (range=%r,shape=%r)" % (self.__buff,(self.start,self.stop),self._shape)

    def close(self):
        self.__buff.release()


class ImageBuffer(FrameBufferBase):
    """
    Wrapper that uses the opencv imread function to implement a stream of images
    """
    def __init__(self,src,start=None,stop=None,**kwargs):
        self.start = start
        self.stop = stop
        self.kwargs = dict(**kwargs)

        self.__buff = list(src)
        if self.stop is None: self.stop = len(self.__buff)
        if self.start is None: self.start = 0
        self._idx = self.start

    def seek(self,frame_index=None):
        self._idx = frame_index if frame_index is not None else self.start

    def read(self):
        if self._idx == self.stop:
            return np.array([],dtype=np.void)

        img = cv2.imread(self.__buff[self._idx],**self.kwargs)
        self._idx += 1
        return img

    def close(self):
        self.__buff = []

        
class FrameBuffer(object):
    def __init__(self,src,stop=None,start=None,**kwargs):
        """
        Class for handling images, video files, and cameras generically through
        opencv
        """
        if start is not None: start, stop = stop, start

        if hasattr(src,'__iter__'):
            buff = ImageBuffer
        elif isinstance(src,basestring):
            buff = VideoBuffer
        else:
            buff = VideoBuffer
        self.__cap = buff(src,start,stop,**kwargs)
        self.source = self.__cap
        self._closed = False
        self.read = self.__cap.read

    def __iter__(self):
        return self.__cap

    # allow user to indirectly access underlying capture object's attributes
    def __getattr__(self,attr):
        return getattr(self.__cap, attr)

    def __repr__(self):
        return self.__cap.__repr__()

    def __str__(self):
        return self.__cap.__str__()
        
    def __enter__(self):
        return self

    def __exit__(self):
        self.__cap.close()

    def close(self):
        if not self._closed: self.__cap.close()
        self._closed = True
        self.source = None
