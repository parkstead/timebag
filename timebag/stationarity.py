#!/usr/bin/env python3


import numpy as np
import pandas as pd
from timebag import *
from timebag.linearmodel import *
from timebag.summary import *

def univar_sanitize(y):
    if isinstance(y,pd.core.frame.DataFrame):
        y = y.values
    if not isinstance(y,np.ndarray):
        raise Exception('Input must be a univariate numpy or pandas series')
    if len(y.shape) is 1:
        y = y.reshape(-1,1)
    if len(y.shape) >= 2:
        for i in range(1,len(y.shape)):
            if y.shape[i] > 1: raise Exception('Input must be a univariate numpy or pandas series')
        y = y.reshape(-1,1)
    y = strip(y)
    return y
def kpss(series,diffs=0,trend=False,lagshort=True):
    y = univar_sanitize(series)
    for i in range(diffs):
        y = diff(y)
    n = y.shape[0]
    if lagshort: 
        lags = int(4*(n/100)**0.25)
    else: 
        lags = int(12*(n/100)**0.25)
    y = y.reshape(-1,1)
    x = mkconst(n)
    if trend:
        x = np.hstack((x,mktrend(n)))
        table = np.array([0.119,0.146,0.176,0.216])
    else:
        table = np.array([0.347,0.463,0.574,0.739])
    tablep = np.array([0.1,0.05,0.025,0.01])
    lm1 = lm(y,x)
    res = lm1.res
    s = res.cumsum()
    eta = cross(s)/n**2
    sig = [2*(1-i/(lags+1))*acovf(res,i) for i in range(1,lags+1)]
    sig = np.array(sig).sum()
    sig += acovf(res,0)
    stat = eta/sig
    p = np.interp(stat,table,tablep)
    df = pd.DataFrame(columns=['method','stat','lags','p'])
    df.loc[0] = ['kpss',stat,lags,p]
    return df
def adf(series,diffs=0,lags=None,explosive=False):
    y = univar_sanitize(series)
    if lags is None: lags=int((y.shape[0]-1)**(1/3))
    for i in range(diffs):
        y = diff(y)
    n = y.shape[0]
    y = y.reshape(-1,1)
    dy = diff(y)
    ly = y[lags:-1,:]
    n -= lags+1
    ymat = dy[lags:,:]
    xmat = [dy[lags-i:-i,:] for i in range(1,lags+1)]
    xmat = np.hstack((ly,mkconst(n),mktrend(n),*xmat))
    lm1 = lm(ymat,xmat)
    method = 'adf'
    stat = lm1.t[0,0]
    table = [[4.38, 4.15, 4.04, 3.99, 3.98, 3.96],\
        [3.95, 3.8 , 3.73, 3.69, 3.68, 3.66],\
        [3.6 , 3.5 , 3.45, 3.43, 3.42, 3.41],\
        [3.24, 3.18, 3.15, 3.13, 3.13, 3.12],\
        [1.14, 1.19, 1.22, 1.23, 1.24, 1.25],\
        [0.8 , 0.87, 0.9 , 0.92, 0.93, 0.94],\
        [0.5 , 0.58, 0.62, 0.64, 0.65, 0.66],\
        [0.15, 0.24, 0.28, 0.31, 0.32, 0.33]]
    tableT = np.array([25, 50, 100, 250, 500, 10000])
    tablep = np.array([0.01,0.025,0.05,0.10,0.90,0.95,0.975,0.88])
    tableilp = []
    ntmp = n
    if n > tableT.max(): 
        ntmp = tableT.max()
    for row in table:
        tableilp += [np.interp(ntmp,tableT,np.array(row))]
    tableilp = -np.array(tableilp)
    p = np.interp(stat,tableilp,tablep)
    df = pd.DataFrame(columns=['method','stat','lags','p'])
    df.loc[0] = ['adf',stat,lags,p]
    return df
def pp(series,diffs=0,lagshort=True,explosive=False,alpha=True):
    y = univar_sanitize(series)
    for i in range(diffs):
        y = diff(y)    
    n=y.shape[0]-1
    y=y.reshape(-1,1)
    yt = y[1:n+2,:]
    yt1 = y[:n,:]
    xmat = np.hstack((mkconst(n),mktrend(n),yt1))
    lm1 = lm(yt,xmat)
    res = lm1.res
    ssqru = cross(res)/n
    if lagshort: lags = int(4*(n/100)**0.25)
    else: lags = int(12*(n/100)**0.25)
    res = res.reshape(-1,1)
    ssqrtl = [2*(1-i/(lags+1))*acovf(res,i) for i in range(1,lags+1)]
    ssqrtl = np.array(ssqrtl).sum()
    ssqrtl += acovf(res,0)
    y_tmp1 = cross(yt1,np.arange(1,n+1).reshape(-1,1))
    y_tmp2 = yt1.sum()
    Dx = (n**2)*(n**2-1)*cross(yt1)/12 - n*y_tmp1**2 + \
         n*(n+1)*y_tmp1*y_tmp2 - n*(n+1)*(2*n+1)*y_tmp2**2/6
    if alpha:
        stat=n*(lm1.b[2]-1) - (n**6)/(24*Dx)*(ssqrtl-ssqru)
        table = [[22.5, 25.7, 27.4, 28.4, 28.9, 29.5],\
                 [19.9, 22.4, 23.6, 24.4, 24.8, 25.1],\
                 [17.9, 19.8, 20.7, 21.3, 21.5, 21.8],\
                 [15.6, 16.8, 17.5, 18,   18.1, 18.3],\
                 [3.66, 3.71, 3.74, 3.75, 3.76, 3.77],\
                 [2.51, 2.6,  2.62, 2.64, 2.65, 2.66],\
                 [1.53, 1.66, 1.73, 1.78, 1.78, 1.79],\
                 [0.43, 0.65, 0.75, 0.82, 0.84, 0.87]]
    else:
        stat=sqrt(ssqru)/sqrt(ssqrtl)*(lm1.b[2]-1)/lm1.se[2]-\
               (n**3)/(4*sqrt(3*Dx*ssqrtl))*(ssqrtl-ssqru) 
        table = [[4.38, 4.15, 4.04, 3.99, 3.98, 3.96],\
                 [3.95, 3.8,  3.73, 3.69, 3.68, 3.66],\
                 [3.6,  3.5,  3.45, 3.43, 3.42, 3.41],\
                 [3.24, 3.18, 3.15, 3.13, 3.13, 3.12],\
                 [1.14, 1.19, 1.22, 1.23, 1.24, 1.25],\
                 [0.8,  0.87, 0.9,  0.92, 0.93, 0.94],\
                 [0.5,  0.58, 0.62, 0.64, 0.65, 0.66],\
                 [0.15, 0.24, 0.28, 0.31, 0.32, 0.33]]
    stat = stat[0,0]
    tableT = np.array([25, 50, 100, 250, 500, 10000])
    tablep = np.array([0.01, 0.025, 0.05, 0.1, 0.9, 0.95, 0.975, 0.99])
    tableilp = []
    ntmp = n
    if n > tableT.max(): 
        ntmp = tableT.max()
    for row in table:
        tableilp += [np.interp(ntmp,tableT,np.array(row))]
    tableilp = np.array(tableilp)
    tableilp = -tableilp
    p = np.interp(stat,tableilp,tablep)
    if explosive: p = 1-p
    df = pd.DataFrame(columns=['method','stat','lags','p'])
    df.loc[0] = ['pp',stat,lags,p]
    return df
#TODO hegy test