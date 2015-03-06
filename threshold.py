import sys,os
sys.path.insert(0, os.path.abspath('../'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import framebuffer as fb


alpha = 0.5
T0 = 20

fig = plt.figure()
axes = {}
axes['raw'] = plt.subplot2grid((3,2), (0, 0), colspan=2)
axes['bkgnd'] = plt.subplot2grid((3,2), (1, 0))
axes['thresh'] = plt.subplot2grid((3,2), (1, 1))
axes['moving'] = plt.subplot2grid((3,2), (2, 0))
axes['moving&skin'] = plt.subplot2grid((3,2), (2, 1))

for k,ax in axes.items():
    ax.set_title(k)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

fb.VERBOSE = 1


cap = fb.FrameBuffer(sys.argv[1] if len(sys.argv)>1 else -1, *map(int,sys.argv[2:]))
try:
    prev = cap.read()
    curr = cap.read()

    prevg = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    currg = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
    T = np.ones_like(currg)*T0
    bkgnd = currg.copy()

    axes['raw'].imshow(curr)
    axes['bkgnd'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    axes['thresh'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    axes['moving'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    axes['moving&skin'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    fig.tight_layout()

    next = cap.read()
    krn = np.ones((3,3),dtype=np.uint8)
    rownums = np.arange(currg.shape[0],dtype=int).reshape(-1,1)
    colnums = np.arange(currg.shape[1],dtype=int).reshape(1,-1)
    while plt.pause(1e-6) is None and next.size:
        nextg = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)

        moving = (cv2.absdiff(prevg,nextg) > T) & (cv2.absdiff(currg,nextg) > T)
        cv2.medianBlur(moving.view(np.uint8),3,dst=moving.view(np.uint8))

        # estimate centroid and bounding box
        area = np.sum(moving)
        dispimg = next.copy()
        mask = moving.copy()
        if area > 100:
            mov_cols = mask*colnums
            mov_rows = mask*rownums
            x = np.sum(mov_cols) / area
            y = np.sum(mov_rows) / area
            x0,x1 = np.min(mov_cols[mask]), np.max(mov_cols[mask])
            y0,y1 = np.min(mov_rows[mask]), np.max(mov_rows[mask])

            cimg = cv2.cvtColor(curr[y0:y1+1,x0:x1+1,:],cv2.COLOR_BGR2YCR_CB)
            cr,cb = cimg[:,:,1], cimg[:,:,2]
            skinmask = (60 <= cb)&(cb <= 90)
            skinmask &= (165 <= cr)&(cr <= 195)
            # skinmask = (77 <= cb)&(cb <= 127)
            # skinmask &= (133 <= cr)&(cr <= 173)
            mask[y0:y1+1,x0:x1+1] = skinmask

            x0,x1 = np.min(mov_cols[mask]), np.max(mov_cols[mask])
            y0,y1 = np.min(mov_rows[mask]), np.max(mov_rows[mask])
            cv2.circle(dispimg,(x,y),5,color=(0,255,0),thickness=-1)
            cv2.rectangle(dispimg,(x0,y0),(x1,y1),color=(0,204,255),thickness=2)

        # draw
        get_imdisp(axes['raw']).set_data(dispimg[:,:,::-1])
        get_imdisp(axes['bkgnd']).set_data(bkgnd)
        get_imdisp(axes['thresh']).set_data(T)
        get_imdisp(axes['moving']).set_data(moving*255)
        get_imdisp(axes['moving&skin']).set_data(mask*255)
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        # Updating threshold depends on current background model
        # so always update this before updating background
        T[~moving] = alpha*T[~moving] + (1-alpha)*5*cv2.absdiff(nextg,bkgnd)[~moving]
        T[T<T0] = T0
        # T[moving] = T[moving]
        bkgnd[~moving] = alpha*bkgnd[~moving] + (1-alpha)*nextg[~moving]
        bkgnd[moving] = nextg[moving]

        prev, prevg = curr, currg
        curr, currg = next, nextg
        next = cap.read()
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.close()
