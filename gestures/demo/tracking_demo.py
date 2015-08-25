#!/usr/bin/python
import cv2
import numpy as np

from gestures.demo.hrsm import HandGestureRecognizer
from gestures.demo.gui import DemoGUI
from gestures.gesture_classification import dollar
from gestures.core.framebuffer import FrameBuffer
from itertools import imap

from gestures import config
params = config.get('model_parameters','dollar')
scale, samplesize = params['scale'], params['samplesize']

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

cap = FrameBuffer.from_argv()
print cap
    
blur = lambda x: cv2.blur(x,(7,7),borderType=cv2.BORDER_REFLECT,dst=x)
prev = blur(cap.read())
curr = blur(cap.read())
prevg = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
currg = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)

gui = DemoGUI(scale,curr.shape,interval=1)
redraw = False
artists = []
gui.show()

hrsm = HandGestureRecognizer(prevg,currg,config.get('gesture_templates'),gesture_match,0)

for curr in imap(blur,cap):
    dispimg = curr.copy()

    laststate = hrsm.state

    # Trigger state machine processing
    hrsm.tick(curr)

    motion = hrsm.segmenter.motion
    skin = hrsm.segmenter.skin

    # Update the display image
    if gui.draw_state == 0:
        dispimg = dispimg
    elif gui.draw_state == 1:
        dispimg *= skin[...,None] # simply a way to broadcast multiply on three channels
    elif gui.draw_state == 2:
        dispimg *= motion[...,None]
    elif gui.draw_state == 3:
        dispimg = cv2.cvtColor((hrsm.backprojection*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    elif gui.draw_state == 4:
        dispimg = cv2.cvtColor(hrsm.segmenter.moseg.background,cv2.COLOR_GRAY2BGR)

    if hrsm.state == 'Track':
        if len(hrsm.waypts) == 1: redraw = True
        artists.append((gui.lines['draw'],zip(*list(hrsm.waypts))))
        cv2.circle(dispimg,hrsm.waypts[-1],5,(0,255,0),thickness=-1)
    elif hrsm.state == 'Search':
        color = (255,0,0) # if detected else (0,0,255)
        cv2.drawContours(dispimg,[hrsm.detector.hull],0,color,3)
        for pt in hrsm.detector.dpoints:
            cv2.circle(dispimg,tuple(pt),7,color=color,thickness=2)
        cv2.drawContours(dispimg,[hrsm.detector.contour],0,(0,255,0),2)

    if hrsm.state == 'Track':
        x,y,w,h = hrsm.tracker.bbox
        cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,204,255),thickness=2)
    elif hrsm.segmenter.bbox is not None:
        x,y,w,h = hrsm.segmenter.bbox
        cv2.rectangle(dispimg,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)

    # Update the figure
    artists.append((gui.imdisp,dispimg[:,:,::-1].copy()))
    gui.update_artists(list(artists),redraw=redraw)

    artists = [] # empty the artists bucket
    redraw = False
    
