import h5py
import numpy as np
import dollar
import os

libras_fh = h5py.File(os.environ.get('DATASET','libras_random.hdf5'),'r')
tlibras_fh = h5py.File(os.environ.get('TEMPLATES','libras_templates.hdf5'),'w')

N = 64      # resampling size
scale = 1   # scaling size
dtype = [('x',np.float,(N,)),('y',np.float,(N,))]
try:
    for clsid,ds in libras_fh['train'].iteritems():
        template = np.zeros(1,dtype=dtype)

        for sample in ds:
            x,y = dollar.preprocess(sample['x'],sample['y'],scale,N)
            template['x'] += x
            template['y'] += y
        template['x'] /= len(ds)
        template['y'] /= len(ds)

        tlibras_fh.create_dataset(clsid,maxshape=(None,),chunks=True,data=template)
finally:
    libras_fh.close()
    tlibras_fh.close()
