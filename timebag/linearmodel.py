#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import norm
from timebag import *

## Submodule: Linear models
#TODO Add warnings for no constant
#TODO Add glance statistics
#TODO Add confidence intervals
def lm(y,x):
    y,x = lm_sanitize(y,x)
    lm = lm_init(y,x)
    lm.vb = np.linalg.inv(cross(x))*lm.sig
    lm.vmethod = 'ordinary'
    lm = lm_addt(lm)
    return lm
#TODO Prewhitening with lm (after VAR)
def lm_sanitize(y,x):
    if isinstance(y,pd.core.frame.DataFrame):
        y = y.values
    if not isinstance(y,np.ndarray):
        raise Exception('y input must be a univariate numpy or pandas series')
    if len(y.shape) is 1:
        y = y.reshape(-1,1)
    if len(y.shape) >= 2:
        for i in range(1,len(y.shape)):
            if y.shape[i] > 1: raise Exception('y input must be a univariate numpy or pandas series')
        y = y.reshape(-1,1)
    if isinstance(x,pd.core.frame.DataFrame):
        x = x.values
    if not isinstance(x,np.ndarray):
        raise Exception('x input must be a numpy matrix (n x r) or pandas DataFrame')
    if len(x.shape) is 1:
        x = x.reshape(-1,1)
    if len(x.shape) > 2:
        raise Exception('x input must be a numpy matrix (n x r) or pandas DataFrame')
    if x.shape[0] != y.shape[0]:
        raise Exception('x and y are not the same length in linear model')
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    z = np.hstack((y,x))
    z = strip(z)
    if np.any(np.isnan(z)):
        raise Exception('y and x cannot have internal NaN values. Use tserpy.interp(.) for linear interpolation or tserpy.dropna(.) to drop NaN values')
    y = z[:,0]
    x = z[:,1:]
    return (y,x)
#TODO Bartlette window
def nw(y,x,lags=None):
    y,x = lm_sanitize(y,x)
    lm = lm_init(y,x)
    if lags==None:
        lags = int(lm.n**(1/3))
    W = np.linalg.inv(cross(x)/lm.n)
    Q = cross(x)/lm.n*lm.sig
    xe = cross(lm.res.T,np.ones((1,lm.r)))
    xe = np.multiply(xe,x)
    for lag in range(1,lags):
        Q += 2*(1-lag/(lags+1))*acovf(xe,lag)
    lm.vb = W.dot(Q).dot(W)/lm.n
    lm.vmethod = 'newey-west'
    lm = lm_addt(lm)
    return lm
def lm_init(y,x):
    lm = type('', (), {})()
    lm.n = y.shape[0]
    lm.r = x.shape[1]
    lm.b = np.linalg.solve(cross(x),cross(x,y))
    lm.fit = np.dot(x,lm.b)
    lm.b = lm.b.reshape(-1,1)
    lm.res = y-lm.fit
    lm.sig = cross(lm.res)/(lm.n-lm.r)
    return lm
def lm_addt(lm):
    lm.se = np.sqrt(np.diagonal(lm.vb)).reshape(-1,1)
    lm.t = lm.b/lm.se
    lm.p = norm.cdf(-np.abs(lm.t))*2
    return lm