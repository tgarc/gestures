import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import cv2

data_fh = h5py.File('bhatt.hdf5','r')

fig = plt.figure()
axes = {}
axes['skin'] = fig.add_subplot(211,projection='3d')
axes['non-skin'] = fig.add_subplot(212,projection='3d')

for k,ax in axes.items(): ax.set_title(k)
fig.tight_layout()

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]
try:
    data = np.random.choice(data_fh['skin'],5000,replace=False)
    axes['skin'].scatter(data['b'],data['g'],data['r']
                         ,marker='o',alpha=0.5,edgecolor='k',linewidth=0.15)
    data = np.random.choice(data_fh['non-skin'],5000,replace=False)
    axes['non-skin'].scatter(data['b'],data['g'],data['r']
                             ,marker='o',alpha=0.5,edgecolor='k',linewidth=0.15)

    for ax in axes.values():
        ax.set_xlabel('b')
        ax.set_ylabel('g')
        ax.set_zlabel('r')

    fig.canvas.draw()
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
