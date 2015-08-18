#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gestures.core.framebuffer import FrameBuffer
from gestures.tracking import MeanShiftTracker
import cv2


class App(object):
    def __init__(self,img,callback):
        fig, axes = plt.subplots(1,2)
        axes = dict(zip(['raw','backproject'],axes.ravel()))

        self.fig = fig
        self.axes = axes
        for k,ax in self.axes.items(): ax.set_title(k)

        axes['raw'].imshow(img)
        axes['backproject'].imshow(img[...,0],cmap=mpl.cm.get_cmap('gray'))
        fig.tight_layout()

        cid = fig.canvas.mpl_connect('button_press_event', self.on_press)
        cid = fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.SELECTING = False
        self.bbox = None
        self._bbox = None
        self.callback = callback

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

    def draw(self):
        for ax in self.axes.values(): self.fig.canvas.blit(ax.bbox)
    
    def close(self):
        plt.close(self.fig)


cap = FrameBuffer.from_argv()
try:
    curr = cap.read()
    mstrk = MeanShiftTracker()
    app = App(curr, lambda bbox: mstrk.init(curr,bbox))

    get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

    for curr in cap:
        cv2.blur(curr,(7,7),dst=curr,borderType=cv2.BORDER_REFLECT)
        if not app.SELECTING and app.bbox is not None:
            bbox = mstrk.track(curr)

        get_imdisp(app.axes['raw']).set_data(curr[:,:,::-1])
        if mstrk.backproject is not None:
            get_imdisp(app.axes['backproject']).set_data(mstrk.backproject)
        app.draw()

        plt.pause(1e-6)
except KeyboardInterrupt:
    pass
finally:
    cap.close()

