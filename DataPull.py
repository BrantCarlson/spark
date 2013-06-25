# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:09:29 2013

@author: Kyle Weber
"""

import matplotlib.pyplot as plt
import numpy as np

day = 22
scopeNo = 3
chan = 2
shot = 0

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