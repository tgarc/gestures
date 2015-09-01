#!/usr/bin/python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gestures.utils.framebuffer import FrameBuffer
from gestures.core.common import findBBoxCoM
from gestures.segmentation import SkinMotionSegmenter
from gestures.utils.gui import VideoApp


get_imdisp = lambda ax: ax.get_images()[0]
blur = lambda x: cv2.blur(x,(9,9),borderType=cv2.BORDER_REFLECT)


fig, axes = plt.subplots(2,2)
axes = dict(zip(['raw','moving','skin','fused'],axes.ravel()))
for k,ax in axes.items(): ax.set_title(k)

cap = FrameBuffer.from_argv()
curr = cap.read()
axes['raw'].imshow(curr)
axes['moving'].imshow(curr,cmap=mpl.cm.get_cmap('gray'))
axes['skin'].imshow(curr,cmap=mpl.cm.get_cmap('gray'))
axes['fused'].imshow(curr,cmap=mpl.cm.get_cmap('gray'))
fig.tight_layout()

prev = blur(curr)
curr = blur(cap.read())
prevg = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
currg = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
smseg = SkinMotionSegmenter(prevg,currg)

app = VideoApp(fig,cap)
fig.show()
while app:
    curr = blur(cap.read())
    mask = smseg(curr)
    dispimg = curr.copy()

    if smseg.bbox is not None:
        x,y,w,h = smseg.bbox
        cv2.circle(dispimg,tuple(smseg.com),5,color=(0,255,0),thickness=-1)
        cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)

    app.update_artists((get_imdisp(axes['raw']), dispimg[:,:,::-1])
                       ,(get_imdisp(axes['moving']), smseg.motion*255)
                       ,(get_imdisp(axes['skin']), smseg.skin*255)
                       ,(get_imdisp(axes['fused']), smseg.backprojection*255))
