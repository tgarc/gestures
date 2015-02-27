import numpy as np

def translate(x,y):
    # x_c = ds['x']-np.mean(ds['x'],axis=1).reshape(-1,1)
    # y_c = ds['y']-np.mean(ds['y'],axis=1).reshape(-1,1)
    x_c = x-np.mean(x)
    y_c = y-np.mean(y)
    
    return x_c,y_c

def rotate(x,y):
    theta = np.arctan2(y[0],x[0])
    x_r = x*np.cos(theta)-y*np.sin(theta)
    y_r = x*np.sin(theta)+y*np.cos(theta)

    return x_r, y_r

def scale(x,y):
    return x,y

def resample(x,y):    
    return x,y

def compare(x1,y1,x2,y2):
    return np.sum(np.sqrt((y1-y2)**2 + (x1-x2)**2))/len(x1)

def preprocess(x,y):
    x,y = resample(x,y)
    x,y = rotate(x,y)
    x,y = translate(x,y)
    x,y = scale(x,y)
    return x,y

def query(x,y,templates):
    dists = []
    x,y = preprocess(x,y)
    for clsid,ds in templates.iteritems():
        dists.append((compare(x,y,ds['x'],ds['y']),clsid))

    return sorted(dists,key=lambda x: x[0])
