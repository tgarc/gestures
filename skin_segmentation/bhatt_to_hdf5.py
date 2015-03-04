import numpy as np
import h5py

fn = 'bhatt.txt'
dtype = [('b',np.uint8),('g',np.uint8),('r',np.uint8)]

classes = {1: 'skin',
           2: 'non-skin'}

bhatt_raw = np.loadtxt(fn,dtype=int,delimiter="\t")

with h5py.File("bhatt.hdf5",'w') as bhatt_fh:
    for clsid,clsname in classes.items():
        rows = bhatt_raw[bhatt_raw[:,-1] == clsid]

        data = np.recarray(len(rows),dtype=dtype)
        data['b'] = rows[:,0]
        data['g'] = rows[:,1]
        data['r'] = rows[:,2]

        bhatt_fh.create_dataset(clsname,maxshape=(None,),data=data,chunks=True)

        del data
