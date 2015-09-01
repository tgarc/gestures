#!/usr/bin/python
import cv2
import numpy as np
from gestures.utils.framebuffer import FrameBuffer
from gestures.segmentation import MotionSegmenter
from gestures.utils.gui import VideoApp

import matplotlib as mpl
import matplotlib.pyplot as plt

get_imdisp = lambda ax: ax.get_images()[0]
blur = lambda x: cv2.blur(x,(7,7),borderType=cv2.BORDER_REFLECT)

cap = FrameBuffer.from_argv()
prev = blur(cap.read())
prevg = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
currg = cv2.cvtColor(blur(cap.read()),cv2.COLOR_BGR2GRAY)

fig, axes = plt.subplots(2,2)
axes = dict(zip(['raw','bkgnd','thresh','moving'],axes.ravel()))
for k,ax in axes.items(): ax.set_title(k)
axes['raw'].imshow(prev)
axes['bkgnd'].imshow(prevg,cmap=mpl.cm.get_cmap('gray'))
axes['moving'].imshow(prevg,cmap=mpl.cm.get_cmap('gray'))
axes['thresh'].imshow(prevg,cmap=mpl.cm.get_cmap('gray'))
fig.tight_layout()

app = VideoApp(fig,cap=cap)
moseg = MotionSegmenter(prevg,currg,dict(alpha=0.1))

fig.show()
while app:
    curr = blur(cap.read())
    moving = moseg(cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY))
    dispimg = curr

    # estimate centroid and bounding box
    if moseg.bbox is not None:
        x,y,w,h = moseg.bbox
        cv2.circle(dispimg,(x,y),5,color=(0,255,0),thickness=-1)
        cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)
    
    app.update_artists((get_imdisp(axes['raw']),dispimg[...,::-1].copy())
                       ,(get_imdisp(axes['bkgnd']),moseg.background.copy())
                       ,(get_imdisp(axes['moving']),moving*255)
                       ,(get_imdisp(axes['thresh']),moseg.T.copy()))
