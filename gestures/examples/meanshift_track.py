#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gestures.utils.framebuffer import FrameBuffer
from gestures.utils.gui import VideoApp
from gestures.tracking import CrCbMeanShiftTracker
import cv2
from itertools import imap

get_imdisp = lambda ax: ax.get_images()[0]
blur = lambda x: cv2.blur(x,(7,7),borderType=cv2.BORDER_REFLECT,dst=x)


class MeanShiftApp(VideoApp):
    def __init__(self,callback,**kwargs):
        fig, axes = plt.subplots(1,2)
        self.axes = dict(zip(['raw','backprojection'],axes.ravel()))
        for k,ax in self.axes.items(): ax.set_title(k)

        VideoApp.__init__(self,fig,**kwargs)

        img = self._cap.read()
        self.axes['raw'].imshow(img)
        self.axes['backprojection'].imshow(img[...,0],cmap=mpl.cm.get_cmap('gray'))
        fig.tight_layout()

        cid = fig.canvas.mpl_connect('button_press_event', self.on_press)
        cid = fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.SELECTING = False
        self.bbox = None
        self._bbox = None
        self.callback = callback
        self.show = self._fig.show

    def on_press(self,event):
        self.SELECTING = True
        try:
            self._bbox = [int(event.xdata),int(event.ydata),0,0]
        except:
            self.SELECTING = False

    def on_release(self,event):
        try:
            self._bbox[2:] = int(event.xdata) - self._bbox[0], int(event.ydata) - self._bbox[1]
            assert(self._bbox[2] > 1 and self._bbox[3] > 1)
        except:
            self._bbox = None            
        else:
            self.bbox = tuple(self._bbox)
            self.callback(self.bbox)
        self.SELECTING = False


mstrk = CrCbMeanShiftTracker()
app = MeanShiftApp(lambda bbox: mstrk.init(curr,bbox))
app.show()
for curr in imap(blur,app.cap):
    artists = []
    if not app.SELECTING and app.bbox is not None:
        bbox = mstrk.track(curr)

    artists.append((get_imdisp(app.axes['raw']), curr[:,:,::-1]))
    if mstrk.backprojection is not None:
        artists.append((get_imdisp(app.axes['backprojection']), mstrk.backprojection*255))

    app.update_artists(*artists)
