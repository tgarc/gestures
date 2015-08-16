#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import gestures.framebuffer as fb
from skin_segmenter import GaussianSkinSegmenter
import cv2

# Do some command line parsing
# ------------------------------
from glob import glob
import sys
if len(sys.argv)>1:
    try:
        args = [int(sys.argv[1])]
    except ValueError:
        args = glob(sys.argv[1])
    if len(args) == 1: args = args[0]
else:
    args = -1
# ------------------------------

fig = plt.figure()
axes = {}
axes['raw'] = fig.add_subplot(211)
axes['skin'] = fig.add_subplot(212)

for k,ax in axes.items(): ax.set_title(k)

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

cap = fb.FrameBuffer(args, *map(int,sys.argv[2:]))
try:
    curr = cap.read()
    axes['raw'].imshow(curr)
    axes['skin'].imshow(curr)
    fig.tight_layout()

    pause = 1 if isinstance(cap.source,fb.ImageBuffer) else 1e-6
    coseg = GaussianSkinSegmenter()
    for curr in cap:
        smoothed = cv2.blur(curr,(7,7),borderType=cv2.BORDER_REFLECT)
        skin = coseg(cv2.cvtColor(smoothed,cv2.COLOR_BGR2YCR_CB))

        dispimg = curr.copy()
        dispimg[~skin,:] = 0

        get_imdisp(axes['raw']).set_data(curr[:,:,::-1])
        get_imdisp(axes['skin']).set_data(dispimg[:,:,::-1])
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        plt.pause(pause)
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.close()
