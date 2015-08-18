#!/usr/bin/python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gestures.core.framebuffer import FrameBuffer
from gestures.core.common import findBBoxCoM
from gestures.hand_detection import ConvexityHandDetector
from gestures.segmentation import GaussianSkinSegmenter
from itertools import imap


class App(object):
    def __init__(self,img):
        fig, axes = plt.subplots(1,2)
        axes = dict(zip(['raw','skin'],axes.ravel()))

        self.fig = fig
        self.axes = axes
        for k,ax in self.axes.items(): ax.set_title(k)

        axes['raw'].imshow(img)
        axes['skin'].imshow(img,cmap=mpl.cm.get_cmap('gray'))
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

    hdetect = ConvexityHandDetector()
    smseg = GaussianSkinSegmenter(scale=0.25)

    curr = blur(cap.read())
    app = App(curr)
    for curr in imap(blur,cap):
        dispimg = curr.copy()
        mask = smseg(curr)

        detected = hdetect(mask)
        if detected:
            hull_pts = hdetect.hull
            com = (np.sum(hull_pts[:,0,:],axis=0,dtype=float)/len(hull_pts)).astype(int)
            cv2.circle(dispimg,tuple(com),5,color=(0,255,0),thickness=-1)

        color = (255,0,0) if detected else (0,0,255)
        cv2.drawContours(dispimg,[hdetect.hull],0,color,3)
        for pt in hdetect.dpoints:
            cv2.circle(dispimg,tuple(pt),7,color=color,thickness=2)
        cv2.drawContours(dispimg,[hdetect.contour],0,(0,255,0),2)

        get_imdisp(app.axes['raw']).set_data(dispimg[:,:,::-1])
        get_imdisp(app.axes['skin']).set_data(mask*255)
        app.draw()

        plt.pause(1e-6)
except KeyboardInterrupt:
    pass
finally:
    cap.close()
