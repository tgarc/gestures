import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import cv2
from sys import argv


data_fh = h5py.File(argv[1],'r')

fig = plt.figure()
axes = {}
axes['skin'] = fig.add_subplot(111,projection='3d')

for k,ax in axes.items(): ax.set_title(k)
fig.tight_layout()

get_imdisp = lambda ax: ax.findobj(mpl.image.AxesImage)[0]
try:
    data = np.random.choice(data_fh['skin'],10000,replace=False)
    normdata = np.zeros((len(data),3),dtype=float)
    # sumdata = (data['b'] + data['g'] + data['r']).astype(float)
    normdata[:,0] = data['r']
    normdata[:,1] = data['g']
    normdata[:,2] = data['b']
    axes['skin'].scatter(normdata[:,0],normdata[:,1],normdata[:,2]
                         ,marker='o',alpha=0.5,edgecolor='k',linewidth=0.15)

    for ax in axes.values():
        ax.set_xlabel('r')
        ax.set_ylabel('g')
        ax.set_zlabel('b')

    fig.canvas.draw()
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
