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
states = ('wait','search','track')
STATECYCLE = cycle(states)

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


while not cap.closed:
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
    motion = (cv2.absdiff(imgq_g[0],imgq_g[-1]) > T) & (cv2.absdiff(imgq_g[1],imgq_g[-1]) > T)
    cv2.erode(motion.view(np.uint8),krn,dst=motion.view(np.uint8))
    cv2.dilate(motion.view(np.uint8),krn2,dst=motion.view(np.uint8),iterations=2)
    cv2.erode(motion.view(np.uint8),krn2,dst=motion.view(np.uint8))    
    if np.sum(motion):
        move_bbox, move_com = cmn.findBBoxCoM(motion)
        x,y,w,h = move_bbox

        # fill in the area inside the boundaries of the motion mask
        backg = (cv2.absdiff(imgq_g[-1],bkgnd) > T)[y:y+h,x:x+w]
        cv2.erode(backg.view(np.uint8),krn,dst=backg.view(np.uint8),iterations=1)
        cv2.dilate(backg.view(np.uint8),krn,dst=backg.view(np.uint8),iterations=1)
        motion[y:y+h,x:x+w] |= backg
    else:
        move_bbox = 0,0,0,0

    if state == 'wait':
        CountState()
    elif state == 'search':
        x,y,w,h = move_bbox
        skinarea = np.sum(skin[y:y+h,x:x+w])

        # Estimate hand centroid as the centroid of skin colored pixels inside
        # the bbox of detected movement
        if skinarea:
            hand_bbox,(xcom,ycom) = cmn.findBBoxCoM(skin,move_bbox)

            # Use the hand centroid estimate as our initial estimate for
            # tracking Estimate the hand's bounding box by taking the minimum
            # vertical length to where the skin ends. If the hand is vertical,
            # this should correspond to the length from the palm to tip of
            # fingers
            y = min(hand_bbox[1],move_bbox[1])
            h = 2*(ycom-y)
            x = min(hand_bbox[0],move_bbox[0])
            w = max(move_bbox[0]+move_bbox[2],hand_bbox[0]+hand_bbox[2])-x
            ecc = w/float(h)
            skinbox = h*w
            track_bbox = x,y,w,h
        else:
            ecc = 0
            skinbox = 0

        if skinarea > blobthresh and skinbox < MAXAREA and 0.4 <= ecc and ecc <= 1.1:
            # Gesture candidate detected. Move to validation state
            CountState()

            # Use the skin bbox/centroid to initiate tracking
            x,y,w,h = track_bbox
            crcb_roi = img_crcb[y:y+h,x:x+w]
            skin_roi = skin[y:y+h,x:x+w]
            hist = cv2.calcHist([crcb_roi], chans, skin_roi.view(np.uint8), nbins, ranges)

            # Normalize to 1 to get the sample PDF
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            waypts = [(xcom,ycom)]
            redraw = True
    elif state == 'validate':
        None
    elif state == 'track':
        x,y,w,h = move_bbox
        if np.sum(skin[y:y+h,x:x+w]) > blobthresh:
            bkproject = cv2.calcBackProject([img_crcb],chans,hist,ranges,1)
            bkproject &= motion
            bkproject[y+track_bbox[-1]:,] = 0

            # notice we're using the track_bbox from last iteration
            # for the intitial estimate
            niter, track_bbox = cv2.meanShift(bkproject,track_bbox,term_crit)
            x,y,w,h = track_bbox

            try:
                xcom,ycom = cmn.findBBoxCoM(skin,track_bbox)[1]
            except ValueError:
                xcom,ycom = x+w//2,y+h//2
            waypts.append((xcom,ycom))
        else:
            CountState(STATELEN) # Tracking failed this frame
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
        x,y,w,h = move_bbox
        print "Motion:", move_bbox
        cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)

    # Update the figure
    artists.append((gui.imdisp,dispimg[:,:,::-1].copy()))
    if waypts:
        artists.append((gui.lines['draw'],zip(*list(waypts))))
    gui.update_artists(list(artists),redraw=redraw)

    # Updating threshold depends on current background model
    # so always update this before updating background
    T[~motion] = alpha*T[~motion] + (1-alpha)*5*cv2.absdiff(imgq_g[-1],bkgnd)[~motion]
    # T[motion] = T[motion]
    T[T<T0] = T0

    bkgnd[~motion] = alpha*bkgnd[~motion] + (1-alpha)*imgq_g[-1][~motion]
    # bkgnd[motion] = bkgnd[motion]

    # shift buffer left        
    imgq[:-1] = imgq[1:] 
    imgq_g[:-1] = imgq_g[1:]
    imgq[-1] = cap.read()
    artists = [] # empty the artists bucket
    redraw = False
