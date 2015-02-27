import numpy as np
import scipy as sp

def translate(x,y):
    """ translate point set so that its centroid is at the origin """
    # x_c = ds['x']-np.mean(ds['x'],axis=1).reshape(-1,1)
    # y_c = ds['y']-np.mean(ds['y'],axis=1).reshape(-1,1)
    x_c = x-np.mean(x)
    y_c = y-np.mean(y)
    
    return x_c,y_c

def rotate(x,y,theta):
    """ rotate to indicative angle """
    x_r = x*np.cos(theta)-y*np.sin(theta)
    y_r = x*np.sin(theta)+y*np.cos(theta)

    return x_r, y_r

def scale(x,y):
    return x,y

def resample(x,y):    
    return x,y

def path_dist(x1,y1,x2,y2):
    return np.sum(np.sqrt((y1-y2)**2 + (x1-x2)**2))/len(x1)

def compare(x,y,tmp):
    a = -np.pi/4
    b = np.pi/4
    phi_1 = b - (b-a)/sp.golden
    phi_2 = a + (b-a)/sp.golden

    while abs(phi_2-phi_1)*180/np.pi >= 2:
        x_r,y_r = rotate(x,y,phi_1)
        d1 = path_dist(x_r,y_r,*tmp)
        x_r,y_r = rotate(x,y,phi_2)
        d2 = path_dist(x_r,y_r,*tmp)
        if d1 < d2:
            phi_2=phi_1
            phi_1=b-(b-phi_1)/sp.golden
        else:
            phi_1=phi_2
            phi_2=a+(b-phi_2)/sp.golden
            
    return min(d1,d2)

def preprocess(x,y):
    # must be done in order: resample,translate,rotate,scale
    x,y = resample(x,y)
    x,y = translate(x,y)
    theta = np.arctan2(y[0],x[0])
    x,y = rotate(x,y,theta)
    x,y = scale(x,y)
    return x,y

def query(x,y,templates):
    dists = []
    x,y = preprocess(x,y)
    for clsid,ds in templates.iteritems():
        # d = compare(x,y,ds)
        d = path_dist(x,y,ds['x'],ds['y'])
        score = 1-2*d/np.sqrt(2)
        dists.append((score,clsid))

    return sorted(dists,key=lambda x: x[0],reverse=True)
