#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gestures.core.framebuffer import FrameBuffer,ImageBuffer
from gestures.segmentation import GaussianSkinSegmenter
from gestures.utils.gui import VideoApp
import cv2
from itertools import imap

get_imdisp = lambda ax: ax.get_images()[0]
blur = lambda x: cv2.blur(x,(9,9),borderType=cv2.BORDER_REFLECT)


fig, axes = plt.subplots(1,2)
axes = dict(zip(['raw','skin'],axes.ravel()))
for k,ax in axes.items(): ax.set_title(k)

cap = FrameBuffer.from_argv()
curr = cap.read()
axes['raw'].imshow(curr)
axes['skin'].imshow(curr)
fig.tight_layout()

app = VideoApp(fig,cap)
coseg = GaussianSkinSegmenter()
fig.show()
for curr in imap(blur,cap):
    skin = coseg(curr)

    dispimg = curr.copy()
    dispimg *= skin[...,None]

    app.update_artists((get_imdisp(axes['raw']), curr[:,:,::-1])
                       ,(get_imdisp(axes['skin']), dispimg[:,:,::-1]))
