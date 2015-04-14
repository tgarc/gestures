import sys
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import framebuffer as fb
from gesture_classifier import dollar
import time
import common as cmn
from config import alpha, T0, scale, samplesize
from gui import DemoGUI
import signal
from hrsm import HandGestureRecognizer

mpl.use("TkAgg")

krn = np.ones((3,3),dtype=np.uint8)
krn2 = np.ones((5,5),dtype=np.uint8)

cap = fb.FrameBuffer(sys.argv[1] if len(sys.argv)>1 else -1, *map(int,sys.argv[2:]))
print cap
    
imgq = [cap.read()]*3
imgq_g = [cv2.cvtColor(imgq[0],cv2.COLOR_BGR2GRAY)]*3
imshape = imgq_g[0].shape
bkgnd = imgq_g[0].copy()
T = np.ones_like(imgq_g[0])*T0

redraw = False
artists = []
gui = DemoGUI(scale,imgq[0].shape,interval=1)
gui.show()

def cleanup(signal,frame):
    gui.close()
    cap.close()
    sys.exit(0)
signal.signal(signal.SIGINT,cleanup)

# Show preprocessed gesture and closest matching template
def gesture_match(query,template,score,theta,clsid):
    x,y = query
    n = len(x)
    x,y = dollar.preprocess(x,y,scale,samplesize)

    if score > 0.8:
        query = dollar.rotate(x,y,theta)
        artists.append((gui.lines['template'],template))
        title = "%s (N=%d, score: %.2f)" % (clsid,n,score)
    else:
        query = x,y
        title = "No match (scored too low)"
    artists.append((gui.lines['query'],query))
    gui.axes['match'].set_title(title)
    global redraw
    redraw = True

    print "Class: %s (%.2f)" % (clsid,score)
    print "Npoints:", n

hrsm = HandGestureRecognizer(imshape,'data/templates.hdf5',gesture_match,0)
for imgq[-1] in cap:
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
    cv2.dilate(skin.view(np.uint8),krn,dst=skin.view(np.uint8))

    # motion detection
    motion = (cv2.absdiff(imgq_g[0],imgq_g[-1]) > T) & (cv2.absdiff(imgq_g[1],imgq_g[-1]) > T)
    cv2.erode(motion.view(np.uint8),krn,dst=motion.view(np.uint8))
    cv2.dilate(motion.view(np.uint8),krn2,dst=motion.view(np.uint8))
    if np.sum(motion):
        move_bbox, move_com = cmn.findBBoxCoM(motion)
        x,y,w,h = move_bbox

        # fill in the area inside the boundaries of the motion mask
        backg = (cv2.absdiff(imgq_g[-1],bkgnd) > T)[y:y+h,x:x+w]
        cv2.erode(backg.view(np.uint8),krn,dst=backg.view(np.uint8),iterations=1)
        cv2.dilate(backg.view(np.uint8),krn2,dst=backg.view(np.uint8),iterations=1)
        motion[y:y+h,x:x+w] |= backg
    else:
        move_bbox = 0,0,0,0

    # Trigger state machine processing
    hrsm.tick(motion,skin,move_bbox,img_crcb)

    # Update the display image
    if gui.draw_state == 0:
        dispimg = dispimg
    elif gui.draw_state == 1:
        dispimg[~skin] = 0
    elif gui.draw_state == 2:
        dispimg[motion] = 255
        dispimg[~motion] = 0
    elif gui.draw_state == 3:
        dispimg = cv2.cvtColor(hrsm.backproject*255,cv2.COLOR_GRAY2BGR)
    elif gui.draw_state == 4:
        dispimg = cv2.cvtColor(bkgnd,cv2.COLOR_GRAY2BGR)

    # add skin and motion bounding box annotations
    if hrsm.state == 'Track':
        if len(hrsm.waypts) == 1: redraw = True
        artists.append((gui.lines['draw'],zip(*list(hrsm.waypts))))

        print "Tracking:",hrsm.track_bbox
        x,y,w,h = hrsm.track_bbox
        cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)
        cv2.circle(dispimg,hrsm.waypts[-1],5,(0,255,0),thickness=-1)

    if np.sum(motion):
        print "Motion:", move_bbox
        x,y,w,h = move_bbox
        cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    artists.append((gui.imdisp,dispimg[:,:,::-1].copy()))

    # Update the figure
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
    artists = [] # empty the artists bucket
    redraw = False
    
