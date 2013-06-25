# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:56:26 2013

@author: Zach
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

day = 22
scopeNo = 3
chan = 2
shot = 0

#mydata = open("C:/Users/Zach.Zach-PC/Documents/Carthage/Summer 2013/Flashdrive contents/sparkData_2013/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r")
mydata = open("C:/Sparks/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r")

x = mydata.readlines()
data = x[5:]
spikex = []
spikey = []

def splitit(data):
    i = 0
    xplot = []
    yplot = []
    while i < len(data):
        singlepoint = data[i].split(',')
        xplot == xplot.append(float(singlepoint[0]))
        if float(singlepoint[1]) <= -0.007:
            spikey == spikey.append(0.05)
            spikex == spikex.append(float(singlepoint[0]))
            yplot == yplot.append(float(singlepoint[1]))
        else:
            spikey == spikey.append(0.0)
            spikex == spikex.append(float(singlepoint[0]))
            yplot == yplot.append(float(singlepoint[1]))
        i += 1
    return (xplot,yplot)
(x,y) = splitit(data)

index_count = 0

def time_intervals(x,y):
    #takes two lists as arguments
    results = []
    start = 0.0
    end = 0.0
    duration = 0.0
    index_count = 0
    #Defining variables
    while index_count <= len(x) - 1:
        for i in y[index_count:]:
            index_count += 1
            if i == 0.05 and y[index_count+1] == 0.05:
                start = x[index_count]
                break
            #Sorting through the data to find the start of a spike
        for i in y[index_count:]:
            index_count += 1
            if y[index_count] == 0.0 and y[index_count-1] == 0.05:
                end = x[index_count-1]
                break
            #sorting through the data to find the end of a spike
        duration = end - start
        #Finding the duration of the spike
        if duration > 3.0e-09 and start != 0:
            results.append(start)
            results.append(end)
            results.append(duration)
            print "Spike Duration: " + str(duration) + " seconds."
            print "Start: " + str(start) + " seconds.", "End: " + str(end) + " seconds."
            #Checks to see if the spike is a false-positive from noise
        start = 0
        end = 0
        #resets start and end so the last spike isn't reported twice
    return results
    
print time_intervals(spikex, spikey)

line = plt.plot(x,y,'b-')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))

spike_finder = plt.plot(spikex,spikey,'r-')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))
plt.show()

mydata.close()