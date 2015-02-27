import numpy as np
from scipy.constants import golden

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

def goldensearch(f,a,b,tol=1e-6):
    xl = b - (b-a)/golden
    xu = a + (b-a)/golden
    eps = abs(b-a)

    while eps > tol:
        print a,b,eps
        if f(xl)<f(xu):
            b = xu
            xu = xl
            xl = b - (b-a)/golden
        else:
            a = xl
            xl = xu
            xu = a + (b-a)/golden
        eps = abs(b-a)

    return (xu+xl)/2.

def compare(x,y,tmp):
    def f(phi): x_r,y_r = rotate(x,y,phi); return path_dist(x_r,y_r,tmp['x'],tmp['y'])
    theta_hat = goldensearch(f,-np.pi/4,np.pi/4,tol=np.pi/90)
    x_r,y_r = rotate(x,y,theta_hat)

    return path_dist(x_r,y_r,tmp['x'],tmp['y']), theta_hat

def preprocess(x,y):
    # must be done in order: resample,translate,rotate,scale
    x,y = resample(x,y)
    x,y = translate(x,y)
    x,y = rotate(x,y,np.arctan2(y[0],x[0]))
    x,y = scale(x,y)
    return x,y

def query(x,y,templates):
    dists = []
    x,y = preprocess(x,y)
    for clsid,ds in templates.iteritems():
        d, theta = compare(x,y,ds)
        score = 1-2*d/np.sqrt(2)
        dists.append((score,theta,clsid))

    return sorted(dists,key=lambda x: x[0],reverse=True)
