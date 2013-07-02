# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:10:21 2013

@author: Kyle Weber
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import conf

def readData(day,shot,scopeNo,chan):

    with open(conf.dataDir + "/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r") as f:    
        for i in range(4):        
            f.readline() 
            # read (and discard) the first 4 lines            
        x = pd.read_csv(f) 
            # then use Pandas' built-in comma-separated-variable function            
            # using the with ... construct, f is automatically closed.
    return x


j = readData(22,0,3,2)
def findspikes(j):
    plt.plot(j)
    rj = j.iloc[20:len(j)-20]
    peak_amp = pd.DataFrame.min(j)[1]
    peak_time_i = pd.DataFrame.idxmin(j)[1]
    peak_time = j['Time'][peak_time_i]   
    
    spike = pd.rolling_mean(rj,40)
    #plt.plot(spike)
    if abs(peak_amp) > abs(pd.Series.mean(j['Ampl']) - 2 * pd.Series.std(j['Ampl'])):
        for i in range(peak_time_i-150,58,-1):
            front_der = (spike['Ampl'][i] - spike['Ampl'][i - 1]) / (spike['Time'][i] - spike['Time'][i - 1])
            a = 0        
            if front_der >= 0:
                start = spike['Time'][i]
                print "\nStarts at " + str(start)
                a = i
                break
        for i in range(peak_time_i + 150,len(j),1):
            end = 0        
            back_der = ( (spike['Ampl'][i + 1] - spike['Ampl'][i]) / (spike['Time'][i + 1] - spike['Time'][i]) )
            b = 0
            if back_der <= 0:
                end = spike['Time'][i]
                print "Ends at " + str(end)
                b = i
                break  
        print "Spike duration of " + str(end - start) + " seconds"
        print "Spike peak at t=" + str(peak_time) + " and amplitude " + str(peak_amp) + " millivolts"     
        j['Ampl'][a:b] = 0
        return j
    else: 
        print "No discernible spike in data"
    
count = 0
while count < 4:
    findspikes(j)
    count += 1
    