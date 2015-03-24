"""
A variation of the $1 gesture recognition algorithm:

http://depts.washington.edu/aimgroup/proj/dollar/
"""
import numpy as np
from scipy.constants import golden

def rotate(x,y,theta,x_c=None,y_c=None):
    """ rotate to indicative angle """
    if x_c is None: x_c = np.mean(x)
    if y_c is None: y_c = np.mean(y)

    x_r = (x-x_c)*np.cos(theta)-(y-y_c)*np.sin(theta) + x_c
    y_r = (x-x_c)*np.sin(theta)+(y-y_c)*np.cos(theta) + y_c

    return x_r, y_r

def scale(x,y,size):
    w = max(x)-min(x)
    h = max(y)-min(y)
    x = np.array(x)*size/float(w)
    y = np.array(y)*size/float(h)
    return x,y

def resample(x,y,N):
    d = sum(np.sqrt((x[i]-x[i-1])**2+(y[i]-y[i-1])**2) for i in range(1,len(x)))
    itv = d/float(N-1)

    newpoints = []
    newpoints.append((x[0],y[0]))
    x = list(x)
    y = list(y)

    D = 0
    i = 1
    while i < len(x):
        d = np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 )
        if (D+d) >= itv:
            px = x[i-1] + ((itv-D)/d)*(x[i]-x[i-1])
            py = y[i-1] + ((itv-D)/d)*(y[i]-y[i-1])
            newpoints.append((px,py))
            x.insert(i,px)
            y.insert(i,py) 
            D = 0
        else:
            D += d
        i += 1
    if len(newpoints) < N: newpoints.append(newpoints[-1])

    return zip(*newpoints)

def path_dist(x1,y1,x2,y2):
    return np.sum(np.sqrt((y1-y2)**2 + (x1-x2)**2))/len(x1)

def goldensearch(f,a,b,tol=1e-6):
    xl = b - (b-a)/golden
    xu = a + (b-a)/golden
    eps = abs(b-a)

    while eps > tol:
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

def compare(x,y,tmp,phi_a,phi_b,tol):
    def f(phi): x_r,y_r = rotate(x,y,phi); return path_dist(x_r,y_r,tmp['x'],tmp['y'])

    theta_hat = goldensearch(f,phi_a,phi_b,tol)

    return f(theta_hat), theta_hat

def preprocess(x,y,size,N):
    # must be done in order: resample,rotate,scale,translate
    x,y = resample(x,y,N)
    # theta = np.arctan2(y_c-y[0],x_c-x[0])
    # x,y = rotate(x,y,theta,x_c,y_c)
    x,y = scale(x,y,size)

    x_c = np.mean(x)
    y_c = np.mean(y)
    x -= x_c
    y -= y_c

    return x,y

def query(x,y,size,N,templates,phi_a=-np.pi/6,phi_b=np.pi/6,tol=np.pi/90):
    dists = []
    x,y = preprocess(x,y,size,N)
    for clsid,ds in templates.iteritems():
        d, theta = compare(x,y,ds,phi_a,phi_b,tol=tol)
        score = 1-2*d/np.sqrt(2)
        dists.append((score,theta,clsid))

    return sorted(dists,key=lambda x: x[0],reverse=True)
