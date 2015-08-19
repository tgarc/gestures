#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gestures.core.framebuffer import FrameBuffer,ImageBuffer
from gestures.segmentation import GaussianSkinSegmenter
import cv2
from itertools import imap


fig = plt.figure()
axes = {}
axes['raw'] = fig.add_subplot(211)
axes['skin'] = fig.add_subplot(212)

for k,ax in axes.items(): ax.set_title(k)

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]
blur = lambda x: cv2.blur(x,(9,9),borderType=cv2.BORDER_REFLECT)

cap = FrameBuffer.from_argv()
try:
    curr = cap.read()
    axes['raw'].imshow(curr)
    axes['skin'].imshow(curr)
    fig.tight_layout()

    pause = 1 if isinstance(cap.source,ImageBuffer) else 1e-6
    coseg = GaussianSkinSegmenter()
    for curr in imap(blur,cap):
        skin = coseg(curr)

        dispimg = curr.copy()
        dispimg *= skin[...,None]

        get_imdisp(axes['raw']).set_data(curr[:,:,::-1])
        get_imdisp(axes['skin']).set_data(dispimg[:,:,::-1])
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        plt.pause(pause)
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.close()
