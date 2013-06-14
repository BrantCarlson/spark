# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:09:29 2013

@author: Kyle Weber
"""

import matplotlib.pyplot as plt
import numpy as np

day = int(raw_input("Testing Day?"))
scopeNo = int(raw_input("Oscilloscope Number?"))
chan = int(raw_input("Channel Number?"))
shot = int(raw_input("Shot Number?"))

mydata = open("C:/Users/Zach.Zach-PC/Documents/Carthage/Summer 2013/Flashdrive contents/sparkData_2013/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r")
x = mydata.readlines()
data = x[5:]

def splitit(data):
    i = 0
    xplot = []
    yplot = []
    while i < len(data):
        singlepoint = data[i].split(',')
        xplot == xplot.append(float(singlepoint[0]))
        yplot == yplot.append(float(singlepoint[1]))
        i += 1
    return (xplot,yplot)
(x,y) = splitit(data)

j = np.array((x,y))
print j

line = plt.plot(x,y,'b-')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))

mydata.close()