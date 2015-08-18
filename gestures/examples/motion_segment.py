#!/usr/bin/python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gestures.core.framebuffer import FrameBuffer
from gestures.segmentation import MotionSegmenter
from itertools import imap

fig, axes = plt.subplots(2,2)
axes = dict(zip(['raw','bkgnd','thresh','moving'],axes.ravel()))

for k,ax in axes.items(): ax.set_title(k)

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

cap = FrameBuffer.from_argv()
try:
    blur = lambda x: cv2.blur(x,(7,7),borderType=cv2.BORDER_REFLECT)

    prev = blur(cap.read())
    curr = blur(cap.read())
    prevg = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    currg = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
    moseg = MotionSegmenter(prevg,currg,dict(alpha=0.1))

    axes['raw'].imshow(curr)
    axes['bkgnd'].imshow(currg,cmap=mpl.cm.get_cmap('gray'))
    axes['moving'].imshow(currg,cmap=mpl.cm.get_cmap('gray'))
    axes['thresh'].imshow(currg,cmap=mpl.cm.get_cmap('gray'))
    fig.tight_layout()

    for curr in imap(blur,cap):
        moving = moseg(cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY))
        dispimg = curr

        # estimate centroid and bounding box
        if moseg.bbox is not None:
            x,y,w,h = moseg.bbox
            cv2.circle(dispimg,(x,y),5,color=(0,255,0),thickness=-1)
            cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)

        get_imdisp(axes['raw']).set_data(dispimg[:,:,::-1])
        get_imdisp(axes['bkgnd']).set_data(moseg.background)
        get_imdisp(axes['moving']).set_data(moving*255)
        get_imdisp(axes['thresh']).set_data(moseg.T)
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        plt.pause(1e-6)
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.close()
