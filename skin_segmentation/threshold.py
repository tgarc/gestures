import sys,os
sys.path.insert(0, os.path.abspath('../'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import framebuffer as fb
import cv2

# Do some command line parsing
# ------------------------------
from glob import glob
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
fig.tight_layout()

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

cap = fb.FrameBuffer(args, *map(int,sys.argv[2:]))
try:
    curr = cap.read()
    axes['raw'].imshow(curr)
    axes['skin'].imshow(curr)

    for curr in cap:
        cimg = cv2.cvtColor(curr,cv2.COLOR_BGR2YCR_CB)
        y,cr,cb = cimg[:,:,0], cimg[:,:,1], cimg[:,:,2]
        mask = (60 <= cb)&(cb <= 90)
        mask &= (165 <= cr)&(cr <= 195)

        dispimg = curr.copy()
        dispimg[~mask,:] = 0

        get_imdisp(axes['raw']).set_data(curr[:,:,::-1])
        get_imdisp(axes['skin']).set_data(dispimg[:,:,::-1])
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        plt.pause(1 if isinstance(cap.source,fb.ImageBuffer) else 1e-6)

except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.close()
