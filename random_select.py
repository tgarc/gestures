import h5py
import numpy as np

libras_fh = h5py.File('/home/tdos/gestures/libras.hdf5','r')
rlibras_fh = h5py.File('/home/tdos/gestures/libras_random.hdf5','w')

rlibras_fh.create_group('test')
rlibras_fh.create_group('train')

# Select a random subset of each class
import numpy.random as npr

samplesize=8
try:
    for clsid in libras_fh:
        dtype = libras_fh[clsid].dtype.descr+[('index',np.int_,)]

        # Create random training data
        ridx = np.arange(len(libras_fh[clsid]))
        npr.shuffle(ridx)
        ridx = ridx[:samplesize]
        data = np.take(libras_fh[clsid],ridx)

        train = np.recarray(samplesize,dtype=dtype)
        train['x'] = data['x']
        train['y'] = data['y']
        train['index'] = ridx
        rlibras_fh['train'].create_dataset(clsid,maxshape=(None,),data=train,chunks=True)

        # Use the rest of the samples to make the test set
        ridx = list(set(np.arange(len(libras_fh[clsid]))).difference(ridx))
        data = np.take(libras_fh[clsid],ridx)

        test = np.recarray(len(libras_fh[clsid])-samplesize,dtype=dtype)
        test['x'] = data['x']
        test['y'] = data['y']
        test['index'] = ridx
        rlibras_fh['test'].create_dataset(clsid,maxshape=(None,),data=test,chunks=True)
        
        del train,test
finally:
    libras_fh.close()
    rlibras_fh.close()
