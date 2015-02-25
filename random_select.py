import h5py

libras_fh = h5py.File('/home/tdos/gestures/libras.hdf5','r')
rlibras_fh = h5py.File('/home/tdos/gestures/libras_random.hdf5','w')

# Select a random subset of each class
import numpy.random as npr

samplesize=8
try:
    for clsid in libras_fh:
        ridx = np.arange(len(libras_fh[clsid]))
        npr.shuffle(ridx)
        ridx = ridx[:samplesize]
        dtype = libras_fh[clsid].dtype.descr+[('index',np.int_,)]
        data = np.take(libras_fh[clsid],ridx)

        rdata = np.recarray(samplesize,dtype=dtype)
        rdata['x'] = data['x']
        rdata['y'] = data['y']
        rdata['index'] = ridx
        
        rlibras_fh.create_dataset(clsid,maxshape=(None,),data=rdata,chunks=True)
        del rdata
finally:
    libras_fh.close()
    rlibras_fh.close()

    
