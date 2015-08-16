#!/usr/bin/python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gestures.framebuffer import FrameBuffer
from gestures.motion_segmentation import MotionSegmenter
import sys


fig, axes = plt.subplots(2,2)
axes = dict(zip(['raw','bkgnd','thresh','moving'],axes.ravel()))

for k,ax in axes.items(): ax.set_title(k)

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

cap = FrameBuffer(sys.argv[1] if len(sys.argv)>1 else -1, *map(int,sys.argv[2:]))
try:
    blur = lambda x: cv2.blur(x,(7,7),borderType=cv2.BORDER_REFLECT)

    prev = blur(cap.read())
    curr = blur(cap.read())
    prevg = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    currg = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
    moseg = MotionSegmenter(prevg,currg,alpha=0.5,T0=10)

    axes['raw'].imshow(curr)
    axes['bkgnd'].imshow(currg,cmap=mpl.cm.get_cmap('gray'))
    axes['moving'].imshow(currg,cmap=mpl.cm.get_cmap('gray'))
    axes['thresh'].imshow(currg,cmap=mpl.cm.get_cmap('gray'))
    fig.tight_layout()

    rownums = np.arange(currg.shape[0],dtype=int).reshape(-1,1)
    colnums = np.arange(currg.shape[1],dtype=int).reshape(1,-1)

    curr = blur(cap.read())
    while plt.pause(1e-6) is None and curr.size:
        moving = moseg(cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY))

        # estimate centroid and bounding box
        area = np.sum(moving)
        dispimg = curr.copy()
        if area > 10:
            mov_cols = moving*colnums
            mov_rows = moving*rownums
            x = np.sum(mov_cols) / area
            y = np.sum(mov_rows) / area
            x0,x1 = np.min(mov_cols[moving]), np.max(mov_cols[moving])
            y0,y1 = np.min(mov_rows[moving]), np.max(mov_rows[moving])

            cv2.circle(dispimg,(x,y),5,color=(0,255,0),thickness=-1)
            cv2.rectangle(dispimg,(x0,y0),(x1,y1),color=(0,204,255),thickness=2)

        get_imdisp(axes['raw']).set_data(dispimg[:,:,::-1])
        get_imdisp(axes['bkgnd']).set_data(moseg.background)
        get_imdisp(axes['moving']).set_data(moving*255)
        get_imdisp(axes['thresh']).set_data(moseg.T)
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        curr = blur(cap.read())
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.close()
