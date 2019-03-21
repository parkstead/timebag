#!/usr/bin/env python3

import numpy as np
import pandas as pd
from timebag import *

## Summary statistic functions:
#TODO Sanitize input
def acovf(y,lag):
    n = y.shape[0]
    if len(y.shape)==1:
        return cross(y[lag:],y[:n-lag])/n
    if y.shape[1]==1:
        return cross(y[lag:,:],y[:n-lag,:])[0,0]/n
    else:
        return (cross(y[lag:,:],y[:n-lag,:])+cross(y[:n-lag,:],y[lag:,:]))/n/2
##### ACF CODE IS WRONG ######
def acf(y,lag):
    n = y.shape[0]
    if y.shape[1]==1:
        return cross(y[lag:,:],y[:n-lag,:])[0,0]/n
    else:
        return (cross(y[lag:,:],y[:n-lag,:])+cross(y[:n-lag,:],y[lag:,:]))/n/2
#TODO PACF
#TODO rolling subsamples
#TODO spectral decomposition
#TODO Periodogram
#TODO Kalman filter
#TODO HP filter
#TODO Convolution
#TODO Cross-correlation