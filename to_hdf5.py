import numpy as np
import h5py

fn = '/home/tdos/gestures/movement_libras.data'
dtype = [('class','|S32',),('x',np.float,(45,)),('y',np.float,(45,))]

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
libras = np.recarray(len(libras_raw),dtype=dtype)
libras['x'] = libras_raw[:,:-1:2]
libras['y'] = libras_raw[:,1:-1:2]
libras['class'] = map(lambda x: classes[int(x)], libras_raw[:,-1])


libras_fh = h5py.File("libras.hdf5",'w')
libras_fh.create_dataset("libras",maxshape=(None,),data=libras,chunks=True)
libras_fh.close()
