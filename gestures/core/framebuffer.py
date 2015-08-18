import cv2
import numpy as np
from abc import ABCMeta, abstractmethod

try:
    from cv2 import (CAP_PROP_POS_FRAMES,CAP_PROP_FRAME_COUNT)
except ImportError: # v < 3.0
    from cv2.cv import (CV_CAP_PROP_POS_FRAMES as CAP_PROP_POS_FRAMES
                        , CV_CAP_PROP_FRAME_COUNT as CAP_PROP_FRAME_COUNT)

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

        self.__buff = cv2.VideoCapture(src,**kwargs)

        if isinstance(src,basestring):
            if self.start is None:
                self.start = int(self.__buff.get(CAP_PROP_POS_FRAMES))
            if self.stop is None:
                self.stop = int(self.__buff.get(CAP_PROP_FRAME_COUNT)
                                +self.__buff.get(CAP_PROP_POS_FRAMES))
            self.__buff.set(CAP_PROP_POS_FRAMES, self.start)
            self.live = False
        else:
            self.start = 0
            self.stop = np.inf # '-1' is reserved
            self.live = True
        self._idx = self.start

        self._shape = self.__buff.read()[1].shape
        self.seek()

    def seek(self,frame_index=None):
        if not self.live:
            self._idx = frame_index if frame_index is not None else self.start
            self.__buff.set(CAP_PROP_POS_FRAMES, self.start)

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
    """
    FrameBuffer(src[,start], stop, **kwargs)

    Class for handling images, video files, and cameras generically through
    opencv

    Parameters
    ----------
    stop : int or None
        Starting frame index (not supported when using an iterable source)
    start : int or None
        Stopping frame index (not supported when using an iterable source)
    src : string, array_like
        Either a single path to a video file or an iterable of paths to
        image files
    """

    def __init__(self,src,stop=None,start=None,**kwargs):
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

    @classmethod
    def from_argv(cls,stop=None,start=None,**kwargs):
        '''
        Convenience method for parsing command line args

        Usage:

        for video
        <framebuffer-program>.py <device_num> or <file_name>

        for images
        <framebuffer-program>.py <widlcard_expression>...
        '''
        from glob import glob
        from itertools import chain,imap
        from sys import argv

        if len(argv)>2:
            src = chain.from_iterable(imap(glob,argv[1:]))
        elif len(argv)>1:
            try:
                src = int(argv[1])
            except ValueError:
                src = argv[1]
        else:
            src = -1

        return cls(src,stop=stop,start=start,**kwargs)

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
