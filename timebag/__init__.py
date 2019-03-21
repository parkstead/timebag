#!/usr/bin/env python3

import numpy as np
import pandas as pd
from timebag import *

def cross(*args):
    if len(args) is 2: return np.dot(args[0].T,args[1])
    elif len(args) is 1: return np.dot(args[0].T,args[0])
    else: raise Exception('Input to cross(.) must be 1 or 2 arguments')
def vec(x): return x.T.reshape(-1,1)
def uvec(x,*args):
    if len(args) is 1: return x.reshape(args[0],x.shape[0]//args[0]).T
    elif len(args) is 2: return x.reshape(args[0],args[1]).T
    else: raise Exception("Input to uvec(.) must be 2 or 3 arguments")
def quadf(*args): 
    if len(args) is 2: return np.dot(np.dot(args[0].T,args[1]),args[0])
    elif len(args) is 3: return np.dot(np.dot(args[0].T,args[1]),args[2])
    else:raise Exception('Input to quadf(.) must be 2 or 3 arguments')
def mkconst(n): return np.ones(n).reshape(-1,1)
def mktrend(n,off=0): return np.arange(off,n+off).reshape(-1,1)
def diff(y): 
    y = y.copy()
    return y[1:,:]-y[:-1,:]
def strip(z):
    z = z.copy()
    where = np.where(np.all(np.isfinite(z),1))[0]
    start, end = (where[0],where[-1])
    z = z[start:end+1,:]
    return z
def dropna(z):
    z = z.copy()
    ndxs = np.where(np.all(np.isfinite(z),1))
    return z[ndxs,:]
def interp(z):
    z = z.copy()
    where = np.where(np.all(np.isfinite(z),1))[0]
    start, end = (where[0],where[-1])
    ndxs = np.zeros(z.shape[1],dtype=int)
    for i in range(start+1,end):
        for j in range(z.shape[1]):
            if np.isnan(z[i,j]):
                ndxs[j] += 1
                continue
            cnt = ndxs[j]
            if cnt == 0: continue
            frac = 1/(cnt+1)
            split = np.array([z[i-cnt-1,j],z[i,j]])
            z[(i-cnt):i,j] = np.interp(np.arange(frac,1,frac),np.arange(2),split)
            ndxs[j] = 0
    return z
