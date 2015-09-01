import cv2
import numpy as np
from itertools import imap,chain
from glob import glob
from sys import argv


try:
    from cv2 import (CAP_PROP_POS_FRAMES,CAP_PROP_FRAME_COUNT)
except ImportError: # v < 3.0
    from cv2.cv import (CV_CAP_PROP_POS_FRAMES as CAP_PROP_POS_FRAMES
                        , CV_CAP_PROP_FRAME_COUNT as CAP_PROP_FRAME_COUNT)


# video/image factory class
class FrameBuffer(object):
    def __new__(cls,src,stop=None,start=None,**kwargs):
        if start is not None: start, stop = stop, start

        if hasattr(src,'__iter__'):
            buff = ImageBuffer
        elif isinstance(src,basestring):
            buff = VideoBuffer
        else:
            buff = VideoBuffer

        return super(FrameBuffer,cls).__new__(buff,src,stop=stop,start=start,**kwargs)

    @classmethod
    def from_argv(cls,stop=None,start=None,**kwargs):
        '''
        Convenience method for parsing command line args

        Usage:

        for video
        <framebuffer-program>.py <device_num> OR <file_name>

        for images
        <framebuffer-program>.py <widlcard_expression> [<widlcard_expression> ...]
        '''
        if len(argv)>1:
            try:
                src = int(argv[1])
            except ValueError:
                src = list(chain.from_iterable(imap(glob,argv[1:])))
                if len(src) == 1:
                    src = src[0]
        else:
            src = -1

        return cls(src,stop=stop,start=start,**kwargs)

    def _read(self):
        """
        Returns the next image in the feed.

        If end of feed is reached, an empty numpy array is returned.
        """
        raise NotImplementedError

    def read(self):
        if self._closed:
            raise ValueError("I/O operation on closed frame buffer")

        try:
            return self._read()
        except StopIteration:
            return np.array([],dtype=np.void)            

    def close(self):
        raise NotImplementedError

    def seek(self,offset):
        """
        If possible with the current stream, move to new frame index.
        """
        raise NotImplementedError

    def __iter__(self):
        return self

    def next(self):
        frm = self.read()
        if frm.size == 0: 
            raise StopIteration
        return frm


class VideoBuffer(FrameBuffer):
    """
    Wrapper for the opencv video capture interface
    """

    def __init__(self,src,start=None,stop=None,**kwargs):
        self.start = start
        self.stop = stop
        self._closed = False

        self._reader = cv2.VideoCapture(src)

        if isinstance(src,basestring):
            if self.start is None:
                self.start = int(self._reader.get(CAP_PROP_POS_FRAMES))
            if self.stop is None:
                self.stop = int(self._reader.get(CAP_PROP_FRAME_COUNT)
                                +self._reader.get(CAP_PROP_POS_FRAMES))
            self._reader.set(CAP_PROP_POS_FRAMES, self.start)
            self.live = False
        else:
            self.start = 0
            self.stop = np.inf # '-1' is reserved
            self.live = True
        self._idx = self.start

        self.set = self._reader.set
        self.get = self._reader.get

        self._shape = self._reader.read()[1].shape
        self.seek()

    def seek(self,frame_index=None):
        if self.live: 
            return

        self._idx = frame_index
        if frame_index is None:
            self._idx = self.start

        self._reader.set(CAP_PROP_POS_FRAMES, self.start)

    def _read(self):
        if self._idx == self.stop:
            raise StopIteration

        valid, img = self._reader.read()
        if not valid:
            raise IOError("Unable to read from frame buffer")

        self._idx += 1
        return img

    def __repr__(self):
        return "%r (range=%r,shape=%r)" % (self._reader,(self.start,self.stop),self._shape)

    def close(self):
        self._reader.release()
        self._closed = True


class ImageBuffer(FrameBuffer):
    """
    Wrapper that uses the opencv imread function to implement a stream of images
    """

    def __init__(self,src,start=None,stop=None,**kwargs):
        self.kwargs = kwargs

        self._buffer = src if hasattr(src,'__getslice__') else list(src)

        if stop is not None: 
            self._buffer = self._buffer[:stop]
        if start is not None:
            self.offset = start
            self._buffer = self._buffer[start:]
        else:
            self.offset = 0

        self.seek()
        self._closed = False
        
    def seek(self,frame_index=None):
        if frame_index is not None:
            offset = frame_index - self.offset
        else:
            offset = 0
        self._reader = imap(lambda x: cv2.imread(x,**self.kwargs), iter(self._buffer[offset:]))

    def _read(self):
        return next(self._reader)

    def close(self):
        self._buffer = []
        self._reader = None
        self._closed = True
