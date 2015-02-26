import h5py
import numpy as np
import classifier as c

libras_fh = h5py.File('/home/tdos/gestures/libras_random.hdf5','r')
tlibras_fh = h5py.File('/home/tdos/gestures/libras_templates.hdf5','w')

dtype = [('x',np.float,(45,)),('y',np.float,(45,))]

for clsid,ds in libras_fh['train'].iteritems():
    data = np.zeros(1,dtype=dtype)

    for sample in ds:
        x,y = c.resample(sample['x'],sample['y'])
        x,y = c.rotate(x,y)
        x,y = c.translate(x,y)
        x,y = c.scale(x,y)
        data[0]['x'] += x
        data[0]['y'] += y
    data[0]['x'] /= len(ds)
    data[0]['y'] /= len(ds)

    tlibras_fh.create_dataset(clsid,maxshape=(None,),chunks=True,data=data)
libras_fh.close()
tlibras_fh.close()
