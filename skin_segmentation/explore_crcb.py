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

colorspace = np.zeros((255,255),dtype=np.uint8)

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]
try:
    data = np.ndarray((len(data_fh['skin']),3),dtype=np.uint8)
    data[:,0] = data_fh['skin']['b']
    data[:,1] = data_fh['skin']['g']
    data[:,2] = data_fh['skin']['r']
    data = data.reshape(5651,9,3) # factor samples into arbitrary 2d shape
    cimg = cv2.cvtColor(data,cv2.COLOR_BGR2YCR_CB).reshape(50859,3)

    heatmap, xedges, yedges = np.histogram2d(cimg[:,1], cimg[:,2], bins=255)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    axes['raw'].imshow(heatmap)
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
