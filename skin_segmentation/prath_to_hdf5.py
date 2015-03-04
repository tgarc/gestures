import os
import cv2
import h5py
import numpy as np

root = os.path.abspath('pratheepan')
faces = os.path.join(root,'facephoto')
faces_gt = os.path.join(root,'facephoto_gt')
dtype = [('b',np.uint8),('g',np.uint8),('r',np.uint8)]

with h5py.File("prath.hdf5",'w') as prath_fh:
    skins = []
    samplesize = 0

    for fn in os.listdir(faces_gt):
        path = os.path.join(faces_gt,fn)
        img = cv2.imread(path)
        mask = img[:,:,0] > 0
        skins.append((os.path.basename(fn),mask))
        samplesize += np.sum(mask)

    i = 0
    data = np.recarray(samplesize,dtype=dtype)
    for fn,mask in skins:
        img = cv2.imread(os.path.join(faces,fn.replace('.png','.jpg')))
        masksz = np.sum(mask)
        data['b'][i:i+masksz] = img[mask,0]
        data['g'][i:i+masksz] = img[mask,1]
        data['r'][i:i+masksz] = img[mask,2]        
        i+=masksz

    prath_fh.create_dataset('skin',maxshape=(None,),data=data,chunks=True)
    
    
