# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:56:26 2013

@author: Zach
"""

import matplotlib.pyplot as plt
import numpy as np

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
        if float(singlepoint[1]) < -0.008:
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

s = np.array((spikex,spikey))

spike_indices = s.nonzero()
times = s[spike_indices]

def time_intervals(x):
    start = 0.0
    end = 0.0
    start = times[x[0]]
    for e in x:
        if x[e+1] - x[e] > 0.00000005:
            end = times[x[e]]
    return start, end
    
print time_intervals(spike_indices[0])

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