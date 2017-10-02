import batoid
import numpy as np

def RotX(th):
    sth, cth = np.sin(th), np.cos(th)
    return batoid.Rot3([1,0,0,0,cth,-sth,0,sth,cth])

def RotY(th):
    sth, cth = np.sin(th), np.cos(th)
    return batoid.Rot3([cth,0,sth,0,1,0,-sth,0,cth])

def RotZ(th):
    sth, cth = np.sin(th), np.cos(th)
    return batoid.Rot3([cth,-sth,0,sth,cth,0,0,0,1])
