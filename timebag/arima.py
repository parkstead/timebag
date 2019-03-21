#!/usr/bin/env python3

import numpy as np
import pandas as pd
from timebag import *

## Submodule: ARMA
def arma_init(phi,theta):
    p = phi.shape[0]
    q = theta.shape[0]
    if p is 1 and q is 0:
        V = np.ones((1,1))
        P = np.array([[1.0/(1.0-phi[0,0]**2)]])
        T = phi
        R = P**(0.5)
        return(T,R,V,P)
    r = np.max((p,q+1))
    if p is 0:
        phi = np.zeros((r-p,1))
    else:
        phi = np.vstack((phi,np.zeros((r-p,1))))
    if q is 0:
        theta = np.zeros((r-q-1,1))
    else:
        theta = np.vstack((theta,np.zeros((r-q-1,1))))
    if theta.shape[0] is 0:
        R = np.ones((1,1))
    else:
        R = np.vstack((np.ones((1,1)),theta))
    T = np.vstack((np.eye(r-1),np.zeros((1,r-1))))
    T = np.hstack((phi,T))
    V = cross(R.T)
    S = np.eye(r**2)-np.kron(T,T)
    P = np.linalg.solve(S,vec(V))
    P = uvec(P,r,r)
    return(T,R,V,P)
def arma_like(series,phi,theta):
    phi = np.array(phi).reshape(-1,1)
    theta = np.array(theta).reshape(-1,1)
    (T,R,V,P) = arma_init(phi,theta)
    (n,r) = (series.shape[0],T.shape[1])
    vtil = np.zeros((n,1))
    A = np.zeros((r,1))
    (Apred,Ppred) = (A.copy(),P.copy())
    (css,fadd) = (0,0)
    for i in range(n):    
        v = series[i,0]-A[0,0]
        f = P[0,0]
        vtil[i,0] = v*f**(-0.5)
        if i >= r: 
            css += vtil[i,0]**2
            fadd += np.log(f)
        A = Apred + Ppred[0,0]*v/f
        P = Ppred + Ppred[0,0]/f
        Apred = cross(T.T,A)
        Ppred = quadf(T.T,P)+V
    css /= n
    loglike = -n*np.log(css)/2+fadd/2
    return(css,loglike,vtil)
#TODO ARIMA Fit
#TODO auto ARIMA
#TODO ARIMA forecast
#TODO ARIMA generator
#TODO VARIMA
#TODO auto VARIMA
#TODO VARIMA forecast
#TODO ARCH/GARCH