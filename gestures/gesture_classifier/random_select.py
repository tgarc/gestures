#!/usr/bin/python
'''
Create a test and training dataset by selecting randomly from each gesture class
'''
import h5py
import numpy as np
import sys
import numpy.random as npr

import pkg_resources
libras = pkg_resources.resource_filename('gestures', 'data/libras.hdf5')

libras_fh = h5py.File(libras,'r')
test_fh = h5py.File(sys.argv[1]+'_test.h5','w')
train_fh = h5py.File(sys.argv[1]+'_train.h5','w')

trainsize=8
try:
    for clsid in libras_fh:
        dtype = libras_fh[clsid].dtype.descr+[('index',np.int_,)]

        # Create random training data
        ridx = np.arange(len(libras_fh[clsid]))
        npr.shuffle(ridx)
        ridx = ridx[:trainsize]
        data = np.take(libras_fh[clsid],ridx)

        train = np.recarray(trainsize,dtype=dtype)
        train['x'] = data['x']
        train['y'] = data['y']
        train['index'] = ridx
        train_fh.create_dataset(clsid,maxshape=(None,),data=train,chunks=True)

        # Use the rest of the samples to make the test set
        ridx = list(set(np.arange(len(libras_fh[clsid]))).difference(ridx))
        data = np.take(libras_fh[clsid],ridx)

        test = np.recarray(len(libras_fh[clsid])-trainsize,dtype=dtype)
        test['x'] = data['x']
        test['y'] = data['y']
        test['index'] = ridx
        test_fh.create_dataset(clsid,maxshape=(None,),data=test,chunks=True)
        
        del train,test
finally:
    libras_fh.close()
    test_fh.close()
    train_fh.close()
