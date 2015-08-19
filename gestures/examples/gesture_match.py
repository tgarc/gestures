#!/usr/bin/python
import h5py
import numpy as np
import matplotlib.pyplot as plt
from gestures.gesture_classification import dollar
from gestures.config import model_parameters as params
from gestures import config
import sys

templates_fh = h5py.File(config.get('gesture_templates_file'),'r')
libras_fh = h5py.File(sys.argv[1],'r')

print """
dollar classifier demo
======================
Use directional keys to navigate matches
"""

NSAMPLES = sum(len(ds) for ds in libras_fh.itervalues())
try:
    CNT   = 0
    scale = params['dollar']['scale']
    N     = params['dollar']['samplesize']

    fig = plt.figure()
    axes = {}
    axes['query'] = plt.subplot2grid((3,3), (1, 0))
    axes['transform'] = plt.subplot2grid((3,3), (1, 1))
    axes['match_0'] = plt.subplot2grid((3,3), (0, 2))
    axes['match_1'] = plt.subplot2grid((3,3), (1, 2))
    axes['match_2'] = plt.subplot2grid((3,3), (2, 2))
    fig.tight_layout()

    for ax in axes.values():
        ax.grid(which='both')
        ax.minorticks_on()

    def update():
        x,y,c = databuff[CNT]

        scores, thetas, matches = zip(*dollar.query(x,y,scale,N,templates_fh))
        print
        for m,r,s in zip(matches,thetas,scores):
            print "'%s': %f (%f)" % (m,s,r*180/np.pi)

        for ax in axes.values(): ax.cla()

        axes['query'].plot(x,y,'-o')
        axes['query'].set_title("Test sample from class '%s'" % c)

        x,y = dollar.preprocess(x,y,scale,N)
        axes['transform'].plot(x,y,'-o')
        axes['transform'].set_title("Transformed query")

        for i in range(3):
            ds = templates_fh[matches[i]][0]
            x_r,y_r = dollar.rotate(x,y,thetas[i])
            axes['match_'+str(i)].plot(ds['x'],ds['y'],'-o',color='b')
            axes['match_'+str(i)].plot(x_r,y_r,'-o',color='g')
            axes['match_'+str(i)].set_title("'%s' (%.2f)" % (matches[i],scores[i]))
        
        fig.canvas.draw()

    def onkey(event):
        global CNT
        if event.key in ('n','right',' '): CNT = min(CNT+1,NSAMPLES)
        elif event.key in ('backspace','left'): CNT = max(0,CNT-1)
        else: return
        update()

    cid = fig.canvas.mpl_connect('key_press_event', onkey)
    databuff = [(x,y,cls) for cls,ds in libras_fh.iteritems() for x,y in ds.value[['x','y']]]

    # draw first frame then start
    update()        
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    libras_fh.close()
    templates_fh.close()
