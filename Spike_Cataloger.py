# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:06:38 2013

@author: Zach
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

day = 22
scopeNo = 3
chan = 2
shot = 0

spikex = []
spikey = []

def readData(filename):
    data = []
    with open(filename,'r') as f:
        for x in range(4):
            f.readline()
        data = pd.read_csv(f) 
        return data

y = readData("C:/Users/Zach.Zach-PC/Documents/Carthage/Summer 2013/Flashdrive contents/sparkData_2013/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
#y = readData("C:/Sparks/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))

def threshold(y):
    i = 0
    smoothed = pd.rolling_mean(y.Ampl,25)
    deviation = y.Ampl.std()
    print deviation
    while i < len(y['Ampl']):
        if smoothed[i] < deviation.any() * (-1):
            spikey == spikey.append(0.05)
            spikex == spikex.append(y.Time[i])
        else:
            spikey == spikey.append(0.0)
            spikex == spikex.append(y.Time[i])
        i += 1
    return spikex, spikey



index_count = 0

def time_intervals(x,z):
    #takes two lists as arguments
    #Defining variables
    results = []
    start = 0.0
    end = 0.0
    duration = 0.0
    index_count = 0
    peak = 0
    #Loops through indices and looks for beginnings and ends of spikes
    while index_count < len(z) - 1:
        for i in z[index_count:]:
            index_count += 1
            if i == 0.05 and z[index_count+1] == 0.05:
                start = x[index_count]
                break
        for i in z[index_count:]:
            index_count += 1
            if z[index_count] == 0.0 and z[index_count-1] == 0.05:
                end = x[index_count-1]
                break
        duration = end - start
        peak = y.Ampl[start:end].min()
        #Throws out any false-positives from noise
        if duration > 3.0e-9 and start != 0:
            results.append(start)
            results.append(end)
            results.append(peak)
            results.append(duration)
            print "Spike Duration: " + str(duration) + " seconds."
            print "Peak: " + str(peak)
            print "Start: " + str(start) + " seconds.", "End: " + str(end) + " seconds."
        #Resets start and end so as to not report the last spike twice
        start = 0
        end = 0
    return results
    
threshold(y)
time_intervals(spikex,spikey)

plt.plot(y.Time,y.Ampl)
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))

plt.plot(spikex,spikey,'r-')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))
plt.show()

