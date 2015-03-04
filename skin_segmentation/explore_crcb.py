import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


data_fh = h5py.File('bhatt.hdf5','r')

fig = plt.figure()
axes = {}
axes['raw'] = fig.add_subplot(111)

for k,ax in axes.items(): ax.set_title(k)
fig.tight_layout()

try:
    data = np.zeros((len(data_fh['skin']),2),dtype=int)
    Y = 0.299*data_fh['skin']['r'] + 0.587*data_fh['skin']['g'] + 0.114*data_fh['skin']['b']
    data[:,0] = np.around(data_fh['skin']['r']-Y+128)
    data[:,1] = np.around(data_fh['skin']['b']-Y+128)

    heatmap, xedges, yedges = np.histogram2d(data[:,0], data[:,1],bins=np.arange(0,256))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    imdisp = axes['raw'].imshow(heatmap,extent=extent,interpolation='nearest')
    axes['raw'].set_xlabel('Cr')
    axes['raw'].set_ylabel('Cb')

    cbar = fig.colorbar(imdisp)

    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    data_fh.close()
