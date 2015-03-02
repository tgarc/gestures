import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


fig = plt.figure()
axes = {}
axes['raw'] = fig.add_subplot(411)
axes['bkgnd'] = fig.add_subplot(412)
axes['thresh'] = fig.add_subplot(413)
axes['moving'] = fig.add_subplot(414)

for k,ax in axes.items(): ax.set_title(k)
fig.tight_layout()

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

alpha = 0.75
T0 = 35
try:
    cap = cv2.VideoCapture(-1)
    prev = cap.read()[1]
    curr = cap.read()[1]

    prevg = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    currg = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
    T = np.ones_like(currg)*T0
    bkgnd = currg.copy()

    axes['raw'].imshow(curr)
    axes['bkgnd'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    axes['moving'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))
    axes['thresh'].imshow(bkgnd,cmap=mpl.cm.get_cmap('gray'))

    while plt.pause(1e-6) is None:
        next = cap.read()[1]
        nextg = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)

        moving = (cv2.absdiff(prevg,nextg) > T) & (cv2.absdiff(currg,nextg) > T)

        # cimg = cv2.cvtColor(curr,cv2.COLOR_BGR2YCR_CB)
        # y,cb,cr = cimg[:,:,0], cimg[:,:,1], cimg[:,:,2]
        # y[(y <= 10) | (y >= 20)]    = 0
        # cb[(cb <= 10) | (cb >= 20)] = 0
        # cr[(cr <= 10) | (cr < 20)]  = 0
        # cv2.cvtColor(cimg,cv2.COLOR_YCR_CB2RGB,dst=cimg)

        get_imdisp(axes['raw']).set_data(next[:,:,::-1])
        get_imdisp(axes['bkgnd']).set_data(bkgnd)
        get_imdisp(axes['moving']).set_data(moving*255)
        get_imdisp(axes['thresh']).set_data(T)
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        # Updating threshold depends on current background model
        # so always update this before updating background
        T[~moving] = alpha*T[~moving] + (1-alpha)*5*cv2.absdiff(nextg,bkgnd)[~moving]
        # T[moving] = T[moving]
        bkgnd[~moving] = alpha*bkgnd[~moving] + (1-alpha)*nextg[~moving]
        bkgnd[moving] = nextg[moving]

        prev, prevg = curr, currg
        curr, currg = next, nextg
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.release()
