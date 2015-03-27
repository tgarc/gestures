import sys
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import framebuffer as fb
from gesture_classifier import dollar
import time
from itertools import cycle
from common import *
import threading


mpl.use("TkAgg")

STATELEN = 5
STATECYCLE = cycle(('wait','search','track'))

alpha = 0.5
T0 = 30

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
chans = [1,2]
ranges = [0, 256, 0, 256]
nbins = [16,16]

fig = plt.figure(dpi=100)
axes = {}
figshape = (2,2)
axes['raw'] = plt.subplot2grid(figshape, (0, 0),rowspan=2)
axes['draw'] = plt.subplot2grid(figshape, (0, 1))
axes['match'] = plt.subplot2grid(figshape, (1, 1))

axes['raw'].set_title('raw')
axes['raw'].set_xticklabels([])
axes['raw'].set_yticklabels([])
axes['match'].set_ylim(-125,125)
axes['match'].set_xlim(-125,125)
axes['match'].grid(which='both')

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]

templates_fh = h5py.File('gesture_classifier/libras_templates.hdf5','r')
cap = fb.FrameBuffer(sys.argv[1] if len(sys.argv)>1 else -1, *map(int,sys.argv[2:]))
try:
    imgq = [cap.read()]*3
    imgq_g = [cv2.cvtColor(imgq[0],cv2.COLOR_BGR2GRAY)]*3
    imshape = imgq_g[0].shape
    waypts = []
    MAXLEN = min(imshape)//2
    blobthresh_hi = (min(imshape)//4)**2
    blobthresh_lo = 3*blobthresh_hi//4
    T_move = 150#np.pi*min(imshape)//2

    T = np.ones_like(imgq_g[0])*T0
    bkgnd = imgq_g[0].copy()

    axes['raw'].imshow(bkgnd)
    axes['draw'].set_ylim(0,imshape[0])
    axes['draw'].set_xlim(0,imshape[1])

    fig.tight_layout()
    bg_cache = {ax:fig.canvas.copy_from_bbox(ax.bbox) for ax in axes.values()}

    draw_state = 0
    def onclick(event):
        global draw_state
        draw_state = (draw_state+1)%4
        if draw_state == 0:
            axes['raw'].title.set_text('raw')
        elif draw_state == 1:
            axes['raw'].title.set_text('skin')
        elif draw_state == 2:
            axes['raw'].title.set_text('motion')            
        elif draw_state == 3:
            axes['raw'].title.set_text('backproject')
        fig.canvas.draw()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    timer = fig.canvas.new_timer(interval=50)
    figlock= threading.Lock()
    def update():
        figlock.acquire() # wait for data to be ready
        for ax in fig.axes: fig.canvas.blit(ax.bbox)
        figlock.release()
    timer.add_callback(update)
    timer.start()
    fig.show()

    state=STATECYCLE.next()
    statecnt = 0
    def CountState():
        global state, statecnt
        statecnt += 1
        if statecnt == STATELEN:
            statecnt = 0
            state = STATECYCLE.next()

    krn = np.ones((3,3),dtype=np.uint8)
    bkproject = np.zeros_like(bkgnd)
    while imgq[-1].size:
        imgq_g[-1] = cv2.cvtColor(imgq[-1],cv2.COLOR_BGR2GRAY)
        dispimg = imgq[-1].copy()

        # motion detection
        moving = (cv2.absdiff(imgq_g[0],imgq_g[-1]) > T) & (cv2.absdiff(imgq_g[1],imgq_g[-1]) > T)
        movesum = movearea = np.sum(moving)
        if movearea:
            movesum = movearea
            move_roi = findBBoxCoM(moving)[0]
            x0,y0,x1,y1 = move_roi
            movearea = (x1-x0)*(y1-y0)

            print "Moving:", (x0+x1)//2, (y0+y1)//2, x1-x0, y1-y0
            cv2.rectangle(dispimg,(x0,y0),(x1,y1),color=(0,255,0),thickness=2)

        # skin segementation
        img_crcb = cv2.cvtColor(imgq[-1],cv2.COLOR_BGR2YCR_CB)
        cr,cb = img_crcb[:,:,1], img_crcb[:,:,2]
        skin = (77 <= cb)&(cb <= 127)
        skin &= (133 <= cr)&(cr <= 173)
        # skin = (60 <= cb)&(cb <= 90)
        # skin &= (165 <= cr)&(cr <= 195)
        cv2.erode(skin.view(np.uint8),krn,dst=skin.view(np.uint8))
        cv2.dilate(skin.view(np.uint8),krn,dst=skin.view(np.uint8),iterations=2)
        cv2.erode(skin.view(np.uint8),krn,dst=skin.view(np.uint8))

        # set up the image to display
        if draw_state == 0:
            dispimg = dispimg
        elif draw_state == 1:
            dispimg[~skin] = 0
        elif draw_state == 2:
            dispimg[moving] = 255
            dispimg[~moving] = 0
        elif draw_state == 3:
            dispimg = cv2.cvtColor(bkproject*255,cv2.COLOR_GRAY2BGR)
            
        if state == 'wait':
            if statecnt == 0:
                if len(waypts) > 10:
                    # Find best gesture match
                    x,y = zip(*waypts)
                    matches = dollar.query(x,y,250,64,templates_fh)
                    score,theta,clsid = matches[0]

                    ds = templates_fh[clsid][0]
                    x,y = dollar.preprocess(x,y,250,64)

                    # clean up last match
                    figlock.acquire()
                    del axes['match'].lines[:]
                    fig.canvas.restore_region(bg_cache[axes['match']])

                    # Show preprocessed gesture and closest matching template
                    axes['match'].add_line(plt.Line2D(ds['x'],ds['y'],marker='x',color='g'))
                    axes['match'].add_line(plt.Line2D(x,y,marker='o',color='b'))

                    for l in axes['match'].lines: axes['match'].draw_artist(l)

                    axes['match'].set_title("%s (score: %.2f)" % (clsid,score))

                    fig.canvas.draw() # only way I know how to update text regions
                    figlock.release()

                    print "Class: %s (%.2f)" % (clsid,score)
                    print "Npoints:", len(waypts)
                # remove this gesture from the drawing board
                waypts = []
            CountState()
        elif state == 'track':
            x0,y0,x1,y1 = move_roi
            if statecnt == 0:
                CountState() # Make initial hand centroid estimate

                # Estimate hand centroid as the centroid of skin colored pixels
                # inside the bbox of detected movement
                crcb_roi = img_crcb[y0:y1,x0:x1]
                skin_roi = skin[y0:y1,x0:x1]
                (x0,y0,x1,y1),(xcom,ycom) = findBBoxCoM(skin,(x0,y0,x1,y1))
                waypts.append((xcom,ycom))

                # Use the hand centroid estimate as our initial estimate for
                # tracking
                # Estimate the hand's bounding box by taking the minimum
                # vertical length to where the skin ends. If the hand is
                # vertical, this should correspond to the length from the palm
                # to tip of fingers
                h = min(2*min((y1-ycom,ycom-y0)),MAXLEN)
                w = min(x1-x0,MAXLEN)
                track_bbox = xcom-w//2,ycom-h//2,w,h                

                # Use the skin bbox/centroid to initiate tracking
                hist = cv2.calcHist([crcb_roi], chans, skin_roi.view(np.uint8), nbins, ranges)
                # Normalize to 1 to get the sample PDF
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

                print "Skin Tracking:",xcom,ycom,w,h
                cv2.rectangle(dispimg,(x0,y0),(x1,y1),color=(0,204,255),thickness=2)
                cv2.circle(dispimg,(xcom,ycom),5,(0,255,0),thickness=-1)

                # Add this gesture to the drawing board
                figlock.acquire()
                del axes['draw'].lines[:]
                fig.canvas.restore_region(bg_cache[axes['draw']])
                axes['draw'].add_line(plt.Line2D((),(),marker='o',color='b'))
                fig.canvas.draw()
                figlock.release()
            elif movearea > blobthresh_lo and movesum > T_move:
                bkproject = cv2.calcBackProject([img_crcb],chans,hist,ranges,1)
                movereg = np.zeros_like(moving)
                movereg[y0:y1,x0:x1] = True
                bkproject &= movereg

                # notice we're using the track_bbox from last iteration
                # for the intitial estimate
                niter, track_bbox = cv2.meanShift(bkproject,track_bbox,term_crit)
                x,y,w,h = track_bbox
                x0,y0,x1,y1 = x,y,x+w,y+h

                # xcom,ycom = findBBoxCoM(skin,(x0,y0,x1,y1))[1]
                xcom = (x0+x1)/2
                ycom = (y0+y1)/2

                cv2.rectangle(dispimg,(x0,y0),(x1,y1),color=(0,204,255),thickness=2)
                cv2.circle(dispimg,(xcom,ycom),5,(0,255,0),thickness=-1)
                waypts.append((xcom,ycom))

                print "Skin Tracking:",x0,y0,x1,y1
            else:
                CountState() # Tracking failed this frame
        elif state == 'search':
            if movearea > blobthresh_hi:
                # Gesture candidate detected
                x0,y0,x1,y1 = move_roi

                # check that proportion of skin to are of movement bbox is high
                # enough
                if np.sum(skin[y0:y1,x0:x1]) > ((y1-y0)*(x1-x0)//16):
                    CountState()    # Increase trust for gesture candidate

        # wait for drawing to finish
        figlock.acquire()

        # Draw image
        get_imdisp(axes['raw']).set_data(dispimg[:,:,::-1])
        axes['raw'].draw_artist(get_imdisp(axes['raw']))

        # Draw gesture to the drawing board
        if waypts:
            fig.canvas.restore_region(bg_cache[axes['draw']])
            axes['draw'].lines[0].set_data(zip(*waypts))
            axes['draw'].draw_artist(axes['draw'].lines[0])

        # Finished updating data
        figlock.release()

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
        imgq[-1] = cap.read()
        fig.canvas.get_tk_widget().update()

except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    cap.close()
    templates_fh.close()
