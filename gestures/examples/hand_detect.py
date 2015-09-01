#!/usr/bin/python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gestures.utils.framebuffer import FrameBuffer
from gestures.utils.gui import VideoApp
from gestures.hand_detection import ConvexityHandDetector
from gestures.segmentation import GaussianSkinSegmenter


blur = lambda x: cv2.blur(x,(9,9),borderType=cv2.BORDER_REFLECT)
get_imdisp = lambda ax: ax.get_images()[0]


fig, axes = plt.subplots(1,2)
axes = dict(zip(['raw','skin'],axes.ravel()))
for k,ax in axes.items(): ax.set_title(k)

cap = FrameBuffer.from_argv()
curr = cap.read()
axes['raw'].imshow(curr)
axes['skin'].imshow(curr,cmap=mpl.cm.get_cmap('gray'))
fig.tight_layout()

hdetect = ConvexityHandDetector()
smseg = GaussianSkinSegmenter()
app = VideoApp(fig,cap)

fig.show()
while app:
    curr = blur(cap.read())
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

    app.update_artists((get_imdisp(axes['raw']), dispimg[:,:,::-1])
                       ,(get_imdisp(axes['skin']), mask*255))
