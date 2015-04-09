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
import common as cmn
from config import alpha, T0, scale, samplesize
from gui import DemoGUI
import signal

mpl.use("TkAgg")

STATELEN = 3
STATECYCLE = cycle(('wait','search','track'))

krn = np.ones((3,3),dtype=np.uint8)
krn2 = np.ones((5,5),dtype=np.uint8)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
chans = [1,2]
ranges = [0, 256, 0, 256]
nbins = [16,16]

templates_fh = h5py.File('data/templates.hdf5','r')
cap = fb.FrameBuffer(sys.argv[1] if len(sys.argv)>1 else -1, *map(int,sys.argv[2:]))
imgq = [cap.read()]*3
imgq_g = [cv2.cvtColor(imgq[0],cv2.COLOR_BGR2GRAY)]*3
imshape = imgq_g[0].shape

MAXAREA = imgq_g[0].size//4
blobthresh = (min(imshape)//8)**2

state = STATECYCLE.next()
statecnt = 0
def CountState(inc=1):
    global state, statecnt
    statecnt += inc
    if statecnt == STATELEN:
        statecnt = 0
        state = STATECYCLE.next()

waypts = []
bkgnd = imgq_g[0].copy()
bkproject = np.zeros_like(bkgnd)
T = np.ones_like(imgq_g[0])*T0
artists = []
gui = DemoGUI(scale,imgq[0].shape,interval=10)
gui.show()

def cleanup(signal,frame):
    gui.close()
    cap.close()
    templates_fh.close()
    sys.exit(0)
signal.signal(signal.SIGINT,cleanup)


while imgq[-1].size:
    redraw = False
    imgq_g[-1] = cv2.cvtColor(imgq[-1],cv2.COLOR_BGR2GRAY)
    dispimg = imgq[-1].copy()

    # skin segementation
    img_crcb = cv2.cvtColor(imgq[-1],cv2.COLOR_BGR2YCR_CB)
    cr,cb = img_crcb[:,:,1], img_crcb[:,:,2]
    skin = (77 <= cb)&(cb <= 127)
    skin &= (133 <= cr)&(cr <= 173)
    # skin = (60 <= cb)&(cb <= 90)
    # skin &= (165 <= cr)&(cr <= 195)
    cv2.erode(skin.view(np.uint8),krn2,dst=skin.view(np.uint8))
    cv2.dilate(skin.view(np.uint8),krn2,dst=skin.view(np.uint8),iterations=2)
    cv2.erode(skin.view(np.uint8),krn2,dst=skin.view(np.uint8))

    # motion detection
    moving = (cv2.absdiff(imgq_g[0],imgq_g[-1]) > T) & (cv2.absdiff(imgq_g[1],imgq_g[-1]) > T)
    cv2.erode(moving.view(np.uint8),krn,dst=moving.view(np.uint8))
    cv2.dilate(moving.view(np.uint8),krn2,dst=moving.view(np.uint8),iterations=2)
    cv2.erode(moving.view(np.uint8),krn2,dst=moving.view(np.uint8))    
    if np.sum(moving):
        move_roi, move_com = cmn.findBBoxCoM(moving)
        x0,y0,x1,y1 = move_roi

        # fill in the area inside the boundaries of the motion mask
        backg = (cv2.absdiff(imgq_g[-1],bkgnd) > T)[y0:y1,x0:x1]
        cv2.erode(backg.view(np.uint8),krn,dst=backg.view(np.uint8),iterations=1)
        cv2.dilate(backg.view(np.uint8),krn,dst=backg.view(np.uint8),iterations=1)
        moving[y0:y1,x0:x1] |= backg
    else:
        move_roi = 0,0,0,0
    motion = moving.copy()

    if state == 'wait':
        CountState()
    elif state == 'track':
        x0,y0,x1,y1 = move_roi
        if np.sum(skin[y0:y1,x0:x1]) > blobthresh:
            bkproject = cv2.calcBackProject([img_crcb],chans,hist,ranges,1)
            bkproject[y0+track_bbox[-1]:,] = 0
            bkproject &= motion

            # notice we're using the track_bbox from last iteration
            # for the intitial estimate
            niter, track_bbox = cv2.meanShift(bkproject,track_bbox,term_crit)
            x,y,w,h = track_bbox
            x0,y0,x1,y1 = x,y,x+w,y+h

            try:
                xcom,ycom = cmn.findBBoxCoM(skin,(x0,y0,x1,y1))[1]
            except ValueError:
                xcom,ycom = x0+w//2,y0+h//2
            waypts.append((xcom,ycom))
        else:
            CountState() # Tracking failed this frame
            if statecnt == 0:
                if len(waypts) > 10:
                    # Find best gesture match
                    x,y = zip(*waypts)
                    matches = dollar.query(x,y,scale,samplesize,templates_fh)
                    score,theta,clsid = matches[0]

                    ds = templates_fh[clsid][0]
                    x,y = dollar.preprocess(x,y,scale,samplesize)

                    # Show preprocessed gesture and closest matching template
                    artists.append((gui.lines['template'],(ds['x'],ds['y'])))
                    artists.append((gui.lines['query'],(x,y)))
                    gui.axes['match'].set_title("%s (N=%d, score: %.2f)" % (clsid,len(waypts),score))
                    redraw = True

                    print "Class: %s (%.2f)" % (clsid,score)
                    print "Npoints:", len(waypts)
                # remove this gesture from the drawing board
                waypts = []
    elif state == 'search':
        x0,y0,x1,y1 = move_roi
        skinarea = np.sum(skin[y0:y1,x0:x1])

        # Estimate hand centroid as the centroid of skin colored pixels inside
        # the bbox of detected movement
        if skinarea:
            hand_bbox,(xcom,ycom) = cmn.findBBoxCoM(skin,(x0,y0,x1,y1))        

            # Use the hand centroid estimate as our initial estimate for
            # tracking Estimate the hand's bounding box by taking the minimum
            # vertical length to where the skin ends. If the hand is vertical,
            # this should correspond to the length from the palm to tip of
            # fingers
            y0,y1 = hand_bbox[1],hand_bbox[-1]
            # x0,y0,x1,y1 = hand_bbox
            h = 2*min((y1-ycom,ycom-y0))
            w = x1-x0
            ecc = w/float(h)
            skinbox = h*w
        else:
            ecc = 0
            skinbox = 0

        if skinarea > blobthresh and skinbox < MAXAREA and 0.4 <= ecc and ecc <= 1.1:
            # Gesture candidate detected. Check that proportion of skin to
            # area of movement bbox is high enough
            CountState(STATELEN)
            if statecnt == 0:
                # Use the skin bbox/centroid to initiate tracking
                crcb_roi = img_crcb[y0:y1,x0:x1]
                skin_roi = skin[y0:y1,x0:x1]
                hist = cv2.calcHist([crcb_roi], chans, skin_roi.view(np.uint8), nbins, ranges)
                # Normalize to 1 to get the sample PDF
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                waypts = [(xcom,ycom)]
                track_bbox = x0,y0,x1-x0,y1-y0
                redraw = True
        else:
            statecnt = 0

    # Update the display image
    if gui.draw_state == 0:
        dispimg = dispimg
    elif gui.draw_state == 1:
        dispimg[~skin] = 0
    elif gui.draw_state == 2:
        dispimg[motion] = 255
        dispimg[~motion] = 0
    elif gui.draw_state == 3:
        dispimg = cv2.cvtColor(bkproject*255,cv2.COLOR_GRAY2BGR)
    elif gui.draw_state == 4:
        dispimg = cv2.cvtColor(bkgnd,cv2.COLOR_GRAY2BGR)

    # add skin and motion bounding box annotations
    if state == 'track':
        print "Tracking:",track_bbox
        x,y,w,h = track_bbox
        cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)
        cv2.circle(dispimg,waypts[-1],5,(0,255,0),thickness=-1)
    if np.sum(motion):
        x0,y0,x1,y1 = move_roi
        move_bbox = x0,y0,x1-x0,y1-y0
        print "Moving:", move_bbox
        cv2.rectangle(dispimg,(x0,y0),(x1,y1),color=(0,255,0),thickness=2)

    # Update the figure
    artists.append((gui.imdisp,dispimg[:,:,::-1].copy()))
    if waypts:
        artists.append((gui.lines['draw'],zip(*list(waypts))))
    gui.update_artists(list(artists),redraw=redraw)

    # Updating threshold depends on current background model
    # so always update this before updating background
    T[~moving] = alpha*T[~moving] + (1-alpha)*5*cv2.absdiff(imgq_g[-1],bkgnd)[~moving]
    # T[moving] = T[moving]
    T[T<T0] = T0

    bkgnd[~moving] = alpha*bkgnd[~moving] + (1-alpha)*imgq_g[-1][~moving]
    # bkgnd[moving] = bkgnd[moving]

    # shift buffer left        
    imgq[:-1] = imgq[1:] 
    imgq_g[:-1] = imgq_g[1:]
    imgq[-1] = cap.read()
    artists = [] # empty the artists bucket
    redraw = False
