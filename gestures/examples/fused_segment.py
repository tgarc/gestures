#!/usr/bin/python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gestures.core.framebuffer import FrameBuffer
from gestures.core.common import findBBoxCoM
from gestures.segmentation import SkinMotionSegmenter
from itertools import imap

class App(object):
    def __init__(self,img):
        fig, axes = plt.subplots(2,2)
        axes = dict(zip(['raw','moving','skin','fused'],axes.ravel()))

        self.fig = fig
        self.axes = axes
        for k,ax in self.axes.items(): ax.set_title(k)

        axes['raw'].imshow(img)
        axes['moving'].imshow(img,cmap=mpl.cm.get_cmap('gray'))
        axes['skin'].imshow(img,cmap=mpl.cm.get_cmap('gray'))
        axes['fused'].imshow(img,cmap=mpl.cm.get_cmap('gray'))
        fig.tight_layout()

    def draw(self):
        for ax in self.axes.values(): self.fig.canvas.blit(ax.bbox)

    def close(self):
        plt.close(self.fig)


get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

cap = FrameBuffer.from_argv()
try:
    blur = lambda x: cv2.blur(x,(9,9),borderType=cv2.BORDER_REFLECT)

    prev = blur(cap.read())
    curr = blur(cap.read())
    prevg = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    currg = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
    smseg = SkinMotionSegmenter(prevg,currg,scale=0.5)

    curr = blur(cap.read())
    app = App(curr)
    for curr in imap(blur,cap):
        mask = smseg(curr)
        dispimg = curr.copy()

        if smseg.bbox is not None:
            x,y,w,h = smseg.bbox
            cv2.circle(dispimg,tuple(smseg.com),5,color=(0,255,0),thickness=-1)
            cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)

        skinbackproject = cv2.normalize(smseg.coseg.backprojection,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
        fusebackproject = cv2.normalize(smseg.backprojection,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)

        get_imdisp(app.axes['raw']).set_data(dispimg[:,:,::-1])
        get_imdisp(app.axes['moving']).set_data(smseg.motion*255)
        get_imdisp(app.axes['skin']).set_data(skinbackproject.astype(np.uint8))
        get_imdisp(app.axes['fused']).set_data(fusebackproject.astype(np.uint8))
        app.draw()

        plt.pause(1e-6)
except KeyboardInterrupt:
    pass
finally:
    cap.close()
