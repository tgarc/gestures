import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


fig = plt.figure()
axes = {}
axes['raw'] = fig.add_subplot(211)
axes['transform'] = fig.add_subplot(212)
get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

try:
    cap = cv2.VideoCapture(-1)
    prev = cap.read()[1]
    curr = cap.read()[1]

    prevg = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    currg = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)

    axes['raw'].imshow(curr)
    axes['transform'].imshow(currg,cmap=mpl.cm.get_cmap('gray'))

    T = 35
    while plt.pause(1e-6) is None:
        next = cap.read()[1]
        nextg = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)

        timg = (cv2.absdiff(prevg,nextg) > T) & (cv2.absdiff(currg,nextg) > T)
        # cimg = cv2.cvtColor(curr,cv2.COLOR_BGR2YCR_CB)
        # y,cb,cr = cimg[:,:,0], cimg[:,:,1], cimg[:,:,2]
        # y[(y <= 10) | (y >= 20)]    = 0
        # cb[(cb <= 10) | (cb >= 20)] = 0
        # cr[(cr <= 10) | (cr < 20)]  = 0
        # cv2.cvtColor(cimg,cv2.COLOR_YCR_CB2RGB,dst=cimg)

        get_imdisp(axes['transform']).set_data(timg*255)
        get_imdisp(axes['raw']).set_data(next[:,:,::-1])
        for ax in axes.values(): fig.canvas.blit(ax.bbox)

        prev, prevg = curr, currg
        curr, currg = next, nextg
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.release()
