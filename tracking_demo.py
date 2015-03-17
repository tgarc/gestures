import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import framebuffer as fb
import time
from common import *


alpha = 0.5
T0 = 30
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
chans = [1,2]
ranges = [0, 256, 0, 256]
hSize = [32,32]

fig = plt.figure(dpi=100)
axes = {}
figshape = (1,2)
axes['raw'] = plt.subplot2grid(figshape, (0, 0))
axes['draw'] = plt.subplot2grid(figshape, (0, 1))

for k,ax in axes.items():
    ax.set_title(k)
    if k == 'draw': continue
    ax.set_xticklabels([])
    ax.set_yticklabels([])

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

cap = fb.FrameBuffer(sys.argv[1] if len(sys.argv)>1 else -1, *map(int,sys.argv[2:]))
try:
    imgq = [cap.read()]*3
    imgq_g = [cv2.cvtColor(imgq[0],cv2.COLOR_BGR2GRAY)]*3
    imshape = imgq_g[0].shape
    waypts = []
    blobthresh_hi = imgq_g[0].size//10
    blobthresh_lo = 100

    T = np.ones_like(imgq_g[0])*T0
    bkgnd = imgq_g[0].copy()

    axes['raw'].imshow(bkgnd)
    axes['draw'].plot((),(),'-o',color='b')
    axes['draw'].set_ylim(0,imshape[0])
    axes['draw'].set_xlim(0,imshape[1])  
    fig.set_size_inches((figshape[0]*imshape[0]/100., figshape[1]*imshape[1]/100.))
    fig.tight_layout()
    fig.show()
    fig.canvas.draw()
    blankcanvas = fig.canvas.copy_from_bbox(axes['draw'].bbox)

    tracking=False
    t_loop=time.time()
    rownums = np.arange(imshape[0],dtype=int)
    colnums = np.arange(imshape[1],dtype=int)
    track_bbox = 0,0,imshape[0],imshape[1]
    while imgq[-1].size:
        imgq_g[-1] = cv2.cvtColor(imgq[-1],cv2.COLOR_BGR2GRAY)
        dispimg = imgq[-1].copy()
        
        moving = (cv2.absdiff(imgq_g[0],imgq_g[-1]) > T) & (cv2.absdiff(imgq_g[1],imgq_g[-1]) > T)
        cv2.medianBlur(moving.view(np.uint8),3,dst=moving.view(np.uint8))

        img_crcb = cv2.cvtColor(imgq[-1],cv2.COLOR_BGR2YCR_CB)
        cr,cb = img_crcb[:,:,1], img_crcb[:,:,2]
        skin = (77 <= cb)&(cb <= 127)
        skin &= (133 <= cr)&(cr <= 173)
        # skin = (60 <= cb)&(cb <= 90)
        # skin &= (165 <= cr)&(cr <= 195)

        movearea = np.sum(moving)
        if movearea:
            # Calculate bbox of moving pixels
            mov_cols = moving*colnums.reshape(1,-1)
            mov_rows = moving*rownums.reshape(-1,1)
            x0,x1 = np.min(mov_cols[moving]), np.max(mov_cols[moving])+1
            y0,y1 = np.min(mov_rows[moving]), np.max(mov_rows[moving])+1
            movearea = (x1-x0)*(y1-y0)

        if tracking and movearea > blobthresh_lo and np.sum(moving)>120:
            bkproject = cv2.calcBackProject([img_crcb],chans,hist,ranges,1)
            movereg = np.zeros_like(moving)
            movereg[y0:y1,x0:x1] = True
            bkproject &= movereg

            ret, track_bbox = cv2.meanShift(bkproject,track_bbox,term_crit)
            x,y,w,h = track_bbox
            cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)
            waypts.append((x+w/2,y+h/2))

            print "Skin Tracking:",x,y,w,h
        elif tracking:
            tracking = False
            waypts = []
            track_bbox = 0,0,imshape[0],imshape[1]
            axes['draw'].lines[0].remove()
        elif movearea > blobthresh_hi:
            cv2.rectangle(dispimg,(x0,y0),(x1,y1),color=(0,255,0),thickness=2)
            print "Moving:", (x0+x1)//2, (y0+y1)//2, x1-x0, y1-y0
            
            crcb_roi = img_crcb[y0:y1,x0:x1]
            skin_roi = skin[y0:y1,x0:x1]
            if np.sum(skin_roi) > (crcb_roi.size//10):
                tracking = True

                # Estimate hand centroid as the centroid of skin colored pixels
                # inside the bbox of detected movement
                skin_cols = skin_roi*colnums[x0:x1].reshape(1,-1)
                skin_rows = skin_roi*rownums[y0:y1].reshape(-1,1)
                x0,x1 = np.min(skin_cols[skin_roi]), np.max(skin_cols[skin_roi])+1
                y0,y1 = np.min(skin_rows[skin_roi]), np.max(skin_rows[skin_roi])+1
                
                # Use the skin bbox/centroid to initiate tracking
                hist = cv2.calcHist([crcb_roi], chans, skin_roi.view(np.uint8), hSize, ranges)
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                bkproject = cv2.calcBackProject([img_crcb],chans,hist,ranges,1)

                movereg = np.zeros_like(moving)
                movereg[y0:y1,x0:x1] = True
                bkproject &= movereg
                
                ret, track_bbox = cv2.meanShift(bkproject,(x0,y0,x1-x0,y1-y0),term_crit)
                x,y,w,h = track_bbox
                waypts.append((x+w/2,y+h/2))
                cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)
                print "Skin Tracking:",x,y,w,h
                line = plt.Line2D((x,),(y,),marker='o',color='b')
                axes['draw'].add_line(line)

        # draw
        get_imdisp(axes['raw']).set_data(dispimg[:,:,::-1])
        axes['raw'].draw_artist(get_imdisp(axes['raw']))
        if waypts:
            fig.canvas.restore_region(blankcanvas)
            axes['draw'].draw_artist(axes['draw'].lines[0])
            axes['draw'].lines[0].set_data(zip(*waypts))
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        # Updating threshold depends on current background model
        # so always update this before updating background
        T[~moving] = alpha*T[~moving] + (1-alpha)*5*cv2.absdiff(imgq_g[-1],bkgnd)[~moving]
        # T[moving] = T[moving]
        T[T<T0] = T0

        bkgnd[~moving] = alpha*bkgnd[~moving] + (1-alpha)*imgq_g[-1][~moving]
        bkgnd[moving] = imgq_g[-1][moving]

        # shift buffer left        
        imgq[:-1] = imgq[1:] 
        imgq_g[:-1] = imgq_g[1:]
        t = time.time()-t_loop
        while t < 0.0667: t = time.time()-t_loop
        print 1/t
        imgq[-1] = cap.read()
        t_loop = time.time()

except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.close()
