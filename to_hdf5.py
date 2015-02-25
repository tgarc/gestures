import numpy as np
import h5py

fn = '/home/tdos/gestures/movement_libras.data'
dtype = [('x',np.float,(45,)),('y',np.float,(45,))]

classes = {1: 'curved swing',
           2: 'horizontal swing',
           3: 'vertical swing',
           4: 'anti-clockwise arc',
           5: 'clockwise arc',
           6: 'circle',
           7: 'horizontal straight-line',
           8: 'vertical straight-line',
           9: 'horizontal zigzag',
           10: 'vertical zigzag',
           11: 'horizontal wavy',
           12: 'vertical wavy',
           13: 'face-up curve',
           14: 'face-down curve ',
           15: 'tremble'}

libras_raw = np.loadtxt(fn,delimiter=",")


with h5py.File("libras.hdf5",'w') as libras_fh:
    for clsid,clsname in classes.items():
        rows = libras_raw[libras_raw[:,-1] == clsid]

        libras = np.recarray(len(rows),dtype=dtype)
        libras['x'] = rows[:,:-1:2]
        libras['y'] = rows[:,1:-1:2]

        libras_fh.create_dataset(clsname,maxshape=(None,),data=libras,chunks=True)

        del libras
