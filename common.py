import numpy as np

trunc_coords = lambda shape,xy: tuple(x if x >= 0 and x <= dimsz
                                      else (0 if x < 0 else dimsz)
                                      for dimsz,x in zip(shape[::-1],xy))
inttuple = lambda xy: tuple(map(int,xy))

__imshape = (0,0)
__rownums = np.arange(__imshape[0],dtype=int)
__colnums = np.arange(__imshape[1],dtype=int)

def update_imshape(shape):
    global __imshape, __rownums, __colnums
    __imshape = shape
    __rownums = np.arange(__imshape[0],dtype=int)
    __colnums = np.arange(__imshape[1],dtype=int)

def findBBoxCoM(mask,roi=None):
    if mask.shape != __imshape: update_imshape(mask.shape)

    if roi:
        x,y,w,h = roi
        x0,y0,x1,y1 = x,y,x+w,y+h
        mask = mask[y0:y1,x0:x1]
    else:
        x0,y0,x1,y1 = 0,0,mask.shape[1],mask.shape[0]

    maskarea = np.sum(mask)
    if maskarea == 0: raise ValueError("all-zero array passed")

    masked_cols = mask*__colnums[x0:x1].reshape(1,-1)
    masked_rows = mask*__rownums[y0:y1].reshape(-1,1)
    x0,x1 = np.min(masked_cols[mask]), np.max(masked_cols[mask])+1
    y0,y1 = np.min(masked_rows[mask]), np.max(masked_rows[mask])+1
    xcom = np.sum(masked_cols)//maskarea
    ycom = np.sum(masked_rows)//maskarea

    return (x0,y0,x1-x0,y1-y0),(xcom,ycom)
