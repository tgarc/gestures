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

MAXLEN = min(imshape)//2
blobthresh_hi = (min(imshape)//8)**2
blobthresh_lo = 3*blobthresh_hi//4
T_move = 0#np.pi*min(imshape)//2

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
    cv2.erode(skin.view(np.uint8),krn,dst=skin.view(np.uint8))
    cv2.dilate(skin.view(np.uint8),krn,dst=skin.view(np.uint8),iterations=2)
    cv2.erode(skin.view(np.uint8),krn,dst=skin.view(np.uint8))

    # motion detection
    moving = (cv2.absdiff(imgq_g[0],imgq_g[-1]) > T) & (cv2.absdiff(imgq_g[1],imgq_g[-1]) > T)
    cv2.erode(moving.view(np.uint8),krn2,dst=moving.view(np.uint8))
    cv2.dilate(moving.view(np.uint8),krn2,dst=moving.view(np.uint8))    
    if np.sum(moving):
        move_roi, move_com = cmn.findBBoxCoM(moving)
        x0,y0,x1,y1 = move_roi
        print "Moving:", (x0+x1)//2, (y0+y1)//2, x1-x0, y1-y0
    else:
        move_roi = 0,0,0,0

    # set up the image to display
    if gui.draw_state == 0:
        dispimg = dispimg
    elif gui.draw_state == 1:
        dispimg[~skin] = 0
    elif gui.draw_state == 2:
        dispimg[moving] = 255
        dispimg[~moving] = 0
    elif gui.draw_state == 3:
        dispimg = cv2.cvtColor(bkproject*255,cv2.COLOR_GRAY2BGR)
    if np.sum(moving):
        cv2.rectangle(dispimg,(x0,y0),(x1,y1),color=(0,255,0),thickness=2)

    if state == 'wait':
        CountState()
    elif state == 'track':
        x0,y0,x1,y1 = move_roi
        if np.sum(skin[y0:y1,x0:x1]) > blobthresh_lo:
            bkproject = cv2.calcBackProject([img_crcb],chans,hist,ranges,1)
            movereg = np.zeros_like(moving)
            movereg[y0:y1,x0:x1] = True
            bkproject &= movereg

            # notice we're using the track_bbox from last iteration
            # for the intitial estimate
            niter, track_bbox = cv2.meanShift(bkproject,track_bbox,term_crit)
            x,y,w,h = track_bbox
            x0,y0,x1,y1 = x,y,x+w,y+h

            xcom,ycom = cmn.findBBoxCoM(skin,(x0,y0,x1,y1))[1]
            # xcom = (x0+x1)/2
            # ycom = (y0+y1)/2

            cv2.rectangle(dispimg,(x0,y0),(x1,y1),color=(0,204,255),thickness=2)
            cv2.circle(dispimg,(xcom,ycom),5,(0,255,0),thickness=-1)
            waypts.append((xcom,ycom))

            print "Skin Tracking:",x0,y0,x1,y1
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
        if np.sum(skin[y0:y1,x0:x1]) > blobthresh_hi:
            # Gesture candidate detected. Check that proportion of skin to
            # area of movement bbox is high enough
            CountState()    # Increase trust for gesture candidate
            if statecnt == 0:
                # Estimate hand centroid as the centroid of skin colored pixels
                # inside the bbox of detected movement
                hand_bbox,(xcom,ycom) = cmn.findBBoxCoM(skin,(x0,y0,x1,y1))
                waypts = [(xcom,ycom)]

                # Use the hand centroid estimate as our initial estimate for
                # tracking
                # Estimate the hand's bounding box by taking the minimum
                # vertical length to where the skin ends. If the hand is
                # vertical, this should correspond to the length from the palm
                # to tip of fingers
                x0_hand, x1_hand = hand_bbox[0], hand_bbox[2]
                h = min(2*min((y1-ycom,ycom-y0)),MAXLEN)
                w = min(x1-x0,MAXLEN)
                track_bbox = xcom-w//2,ycom-h//2,w,h                

                # Use the skin bbox/centroid to initiate tracking
                x0,y0,x1,y1 = hand_bbox
                crcb_roi = img_crcb[y0:y1,x0:x1]
                skin_roi = skin[y0:y1,x0:x1]
                hist = cv2.calcHist([crcb_roi], chans, skin_roi.view(np.uint8), nbins, ranges)
                # Normalize to 1 to get the sample PDF
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

                print "Skin Tracking:",xcom,ycom,w,h
                cv2.rectangle(dispimg,(x0,y0),(x1,y1),color=(0,204,255),thickness=2)
                cv2.circle(dispimg,(xcom,ycom),5,(0,255,0),thickness=-1)
                redraw = True
        else:
            statecnt = 0
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
    bkgnd[moving] = imgq_g[-1][moving]

    # shift buffer left        
    imgq[:-1] = imgq[1:] 
    imgq_g[:-1] = imgq_g[1:]
    imgq[-1] = cap.read()
    artists = [] # empty the artists bucket
