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
hSize = [32,32]

fig = plt.figure()
axes = {}
axes['raw'] = plt.subplot2grid((3,2), (0, 0), colspan=2)
axes['bkgnd'] = plt.subplot2grid((3,2), (1, 0))
axes['thresh'] = plt.subplot2grid((3,2), (1, 1))
axes['moving'] = plt.subplot2grid((3,2), (2, 0))
axes['skin'] = plt.subplot2grid((3,2), (2, 1))

for k,ax in axes.items():
    ax.set_title(k)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

cap = fb.FrameBuffer(sys.argv[1] if len(sys.argv)>1 else -1, *map(int,sys.argv[2:]))
try:
    imgq = [cap.read()]*3
    imgq_g = [cv2.cvtColor(imgq[0],cv2.COLOR_BGR2GRAY)]*3
    imshape = imgq_g[0].shape
    waypts = []
    blobthresh_hi = 100
    blobthresh_lo = 50

    T = np.ones_like(imgq_g[0])*T0
    bkgnd = imgq_g[0].copy()

    axes['raw'].imshow(bkgnd)
    axes['skin'].imshow(bkgnd)
    axes['bkgnd'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    axes['thresh'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    axes['moving'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    fig.tight_layout()

    tracking=False
    rownums = np.arange(imshape[0],dtype=int).reshape(-1,1)
    colnums = np.arange(imshape[1],dtype=int).reshape(1,-1)
    track_bbox = 0,0,imshape[0],imshape[1]
    while plt.pause(1e-6) is None and imgq[-1].size:
        imgq_g[-1] = cv2.cvtColor(imgq[-1],cv2.COLOR_BGR2GRAY)
        dispimg = imgq[-1].copy()
        
        moving = (cv2.absdiff(imgq_g[0],imgq_g[-1]) > T) & (cv2.absdiff(imgq_g[1],imgq_g[-1]) > T)
        cv2.medianBlur(moving.view(np.uint8),3,dst=moving.view(np.uint8))

        area = np.sum(moving)
        x,y,w,h = track_bbox

        img_crcb = cv2.cvtColor(imgq[-1],cv2.COLOR_BGR2YCR_CB)
        cr,cb = img_crcb[:,:,1], img_crcb[:,:,2]
        skin = (77 <= cb)&(cb <= 127)
        skin &= (133 <= cr)&(cr <= 173)
        # skin = (60 <= cb)&(cb <= 90)
        # skin &= (165 <= cr)&(cr <= 195)

        if tracking and area > blobthresh_lo:
            bkproject = cv2.calcBackProject([img_crcb],chans,hist,ranges,1)
            bkproject &= moving
            ret, track_bbox = cv2.meanShift(bkproject,track_bbox,term_crit)
            cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)
            waypts.append((x+w/2,y+h/2))

            print (x,y,w,h)
            for p0,p1 in zip(waypts[:-1],waypts[1:]):
                cv2.circle(dispimg, p0, 5, (255,0,0),thickness=-1)
                cv2.line(dispimg, inttuple(p0), inttuple(p1), (255,0,0), 1)
        elif tracking:
            tracking = False
            waypts = []
            track_bbox = 0,0,imshape[0],imshape[1]
        elif area > blobthresh_hi:
            # Estimate initial location by centroid bbox of moving pixels
            mov_cols = moving*colnums
            mov_rows = moving*rownums
            x0,x1 = np.min(mov_cols[moving]), np.max(mov_cols[moving])+1
            y0,y1 = np.min(mov_rows[moving]), np.max(mov_rows[moving])+1

            roi = img_crcb[y0:y1,x0:x1]
            skin_roi = skin[y0:y1,x0:x1]
            if np.sum(skin[y0:y1,x0:x1]) > (roi.size//8):
                tracking = True
                skin_roi = moving[y0:y1,x0:x1]
                hist = cv2.calcHist([roi], chans, skin_roi.view(np.uint8), hSize, ranges)
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                bkproject = cv2.calcBackProject([img_crcb],chans,hist,ranges,1)
                ret, track_bbox = cv2.meanShift(bkproject,(x0,y0,x1-x0,y1-y0),term_crit)
                x,y,w,h = track_bbox
                waypts.append((x+w/2,y+h/2))
                cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)
                print (x,y,w,h)
        # draw
        get_imdisp(axes['raw']).set_data(dispimg[:,:,::-1])
        get_imdisp(axes['bkgnd']).set_data(bkgnd)
        get_imdisp(axes['thresh']).set_data(T)
        get_imdisp(axes['moving']).set_data(moving*255)
        
        skinimg = imgq[-1].copy()
        skinimg[~skin] = 0
        get_imdisp(axes['skin']).set_data(skinimg[:,:,::-1])
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        # Updating threshold depends on current background model
        # so always update this before updating background
        T[~moving] = alpha*T[~moving] + (1-alpha)*5*cv2.absdiff(imgq_g[-1],bkgnd)[~moving]
        T[T<T0] = T0
        # T[moving] = T[moving]
        bkgnd[~moving] = alpha*bkgnd[~moving] + (1-alpha)*imgq_g[-1][~moving]
        bkgnd[moving] = imgq_g[-1][moving]

        # shift buffer left        
        imgq[:-1] = imgq[1:] 
        imgq_g[:-1] = imgq_g[1:]
        imgq[-1] = cap.read()
        
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.close()
