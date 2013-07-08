# -*- coding: utf-8 -*-
"""
Created on Mon Jul 08 10:46:17 2013

@author: Kyle Weber
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import conf

def readData(day,shot,scopeNo,chan):

    with open(conf.kdataDir + "/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r") as f:    
        for i in range(4):        
            f.readline() 
            # read (and discard) the first 4 lines            
        x = pd.read_csv(f) 
            # then use Pandas' built-in comma-separated-variable function            
            # using the with ... construct, f is automatically closed.
    return x


j = readData(22,0,3,2)

def find(j):
    thresh = pd.Series.mean(j.Ampl) - pd.Series.std(j.Ampl)
    rj = j.iloc[20:len(j)-20]  
    spike = pd.rolling_mean(rj,40)
    
    while count < 10:
        
        peak_amp = pd.DataFrame.min(j)[1]   
        peak_time_i = pd.DataFrame.idxmin(j)[1]
        peak_time = j['Time'][peak_time_i]
        
        if peak_amp < thresh:
            