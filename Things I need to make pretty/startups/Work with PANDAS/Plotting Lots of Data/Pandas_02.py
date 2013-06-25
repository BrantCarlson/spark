# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:10:21 2013

@author: Kyle Weber
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def readData(day,shot,scopeNo,chan):

    with open("J:/SURE/sparkData_2013/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r") as f:    
        for i in range(4):        
            f.readline() 
            # read (and discard) the first 4 lines            
        x = pd.read_csv(f) 
            # then use Pandas' built-in comma-separated-variable function            
            # using the with ... construct, f is automatically closed.
    return x
print readData(22,0,2,2)

def multiData(day,shot):
    for scopeNo in range(1,4):
        chan = 1
        while chan < 5:
            x = readData(day,shot,scopeNo,chan)
            plt.figure(scopeNo)
            plt.subplot(410 + (chan))
            plt.plot(x)
            plt.ylabel("Amplitude (mV)")
            plt.xlabel("Time (microseconds)")
            chan += 1
            
            
j = readData(22,0,2,2)

def findspikes(j):
    peak_amp = pd.DataFrame.min(j)
    print peak_amp
"""    
    "This is the amplitude of spike's peak"
    peak_time_i = np.argmin(j[1]) 
    "This gives the i value for the spike's peak"
    peak_time = j[0][peak_time_i]
    spike = pd.rolling_mean(j,40)
    spike.plot()
    for i in range(peak_time_i-150,-1,-1):
        start = 0
        "derivative = change in peak height     /     change in time"
        front_der = ( (spike[i] - spike[i - 1]) / (j[0][i] - j[0][i - 1]))
        a = 0        
        if front_der >= 0:
            start = j[0][i]
            print "\nStarts at " + str(j[0][i])
            a = i
            break
    for i in range(peak_time_i + 150,len(j[1]),1):
        end = 0        
        back_der = ( (spike[i + 1] - spike[i]) / (j[0][i + 1] - j[0][i]) )
        b = 0
        if back_der <= 0:
            end = j[0][i]
            print "Ends at " + str(j[0][i])
            b = i
            break  
    print "Spike duration of " + str(end - start) + " seconds"
    print "Spike peak at t=" + str(peak_time) + " and amplitude " + str(peak_amp) + " millivolts"  
    return (a,b)

findspikes(j)
"""