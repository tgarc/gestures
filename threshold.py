import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import framebuffer as fb
from common import *


alpha = 0.25
T0 = 30
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
chans = [1,2]
ranges = [0, 256, 0, 256]
hSize = [256,256]

fig = plt.figure()
axes = {}
axes['raw'] = plt.subplot2grid((3,2), (0, 0), colspan=2)
axes['bkgnd'] = plt.subplot2grid((3,2), (1, 0))
axes['thresh'] = plt.subplot2grid((3,2), (1, 1))
axes['moving'] = plt.subplot2grid((3,2), (2, 0))
axes['tracking'] = plt.subplot2grid((3,2), (2, 1))

for k,ax in axes.items():
    ax.set_title(k)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

cap = fb.FrameBuffer(sys.argv[1] if len(sys.argv)>1 else -1, *map(int,sys.argv[2:]))
try:
    prev = cap.read()
    curr = cap.read()

    prev_g = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
    T = np.ones_like(curr_g)*T0
    bkgnd = curr_g.copy()

    axes['raw'].imshow(curr)
    axes['tracking'].imshow(curr)
    axes['bkgnd'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    axes['thresh'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    axes['moving'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    fig.tight_layout()

    next = cap.read()
    tracking=False
    rownums = np.arange(curr_g.shape[0],dtype=int).reshape(-1,1)
    colnums = np.arange(curr_g.shape[1],dtype=int).reshape(1,-1)
    track_bbox = 0,0,curr_g.shape[0],curr_g.shape[1]
    while plt.pause(1e-6) is None and next.size:
        next_g = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)
        next_crcb = cv2.cvtColor(next,cv2.COLOR_BGR2YCR_CB)
        dispimg = next.copy()
        
        moving = (cv2.absdiff(prev_g,next_g) > T) & (cv2.absdiff(curr_g,next_g) > T)
        cv2.medianBlur(moving.view(np.uint8),3,dst=moving.view(np.uint8))

        area = np.sum(moving)
        x,y,w,h = track_bbox
        print (x,y,w,h)
        if tracking and area > 100:
            bkproject = cv2.calcBackProject([next_crcb],chans,hist,ranges,1)
            ret, track_bbox = cv2.meanShift(bkproject,track_bbox,term_crit)
            cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)
        elif tracking:
            tracking = False
            track_bbox = 0,0,curr_g.shape[0],curr_g.shape[1]
        elif area > 200:
            # Estimate initial location by centroid bbox of moving pixels
            mov_cols = moving*colnums
            mov_rows = moving*rownums
            x = np.sum(mov_cols) / area
            y = np.sum(mov_rows) / area
            x0,x1 = np.min(mov_cols[moving]), np.max(mov_cols[moving])+1
            y0,y1 = np.min(mov_rows[moving]), np.max(mov_rows[moving])+1

            roi = next_crcb[y0:y1,x0:x1]
            cr,cb = roi[:,:,1], roi[:,:,2]
            skin = (77 <= cb)&(cb <= 127)
            skin &= (133 <= cr)&(cr <= 173)
            if np.sum(skin[y0:y1,x0:x1]) > 50:
                tracking = True
                skin = moving[y0:y1,x0:x1]
                hist = cv2.calcHist([roi], chans, skin.view(np.uint8), hSize, ranges)
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                bkproject = cv2.calcBackProject([next_crcb],chans,hist,ranges,1)
                ret, track_bbox = cv2.meanShift(bkproject,(x0,y0,x1-x0,y1-y0),term_crit)
                x,y,w,h = track_bbox
                cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)
        else:
            skin = np.zeros_like(moving)
            cr,cb = next_crcb[:,:,1], next_crcb[:,:,2]
            skin = (77 <= cb)&(cb <= 127)
            skin &= (133 <= cr)&(cr <= 173)
            skin &= moving

        # draw
        get_imdisp(axes['raw']).set_data(dispimg[:,:,::-1])
        get_imdisp(axes['bkgnd']).set_data(bkgnd)
        get_imdisp(axes['thresh']).set_data(T)
        get_imdisp(axes['moving']).set_data(moving*255)
        get_imdisp(axes['tracking']).set_data(next[y:y+h,x:x+h,::-1])
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        # Updating threshold depends on current background model
        # so always update this before updating background
        T[~moving] = alpha*T[~moving] + (1-alpha)*5*cv2.absdiff(next_g,bkgnd)[~moving]
        T[T<T0] = T0
        # T[moving] = T[moving]
        bkgnd[~moving] = alpha*bkgnd[~moving] + (1-alpha)*next_g[~moving]
        bkgnd[moving] = next_g[moving]

        prev, prev_g = curr, curr_g
        curr, curr_g = next, next_g
        next = cap.read()
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.close()
