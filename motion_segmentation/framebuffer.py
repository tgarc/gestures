import cv2

class FrameBuffer(object):
    def __init__(self,src,start=None,stop=None):
        self.cap = cv2.VideoCapture(src)
        self.is_video = not isinstance(src,basestring)
        self.name = str(src)

        self.start = start
        self.stop = stop

        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)        
        self.dim = (h,w)

        if self.is_video:
            if self.start is None:
                self.start = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if self.stop is None:
                self.stop = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)

    def read(self):
        valid, img = self.cap.read()
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.stop:
            return False, img
        return valid, img

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()
