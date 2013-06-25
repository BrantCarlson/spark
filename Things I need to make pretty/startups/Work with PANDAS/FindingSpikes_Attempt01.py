# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:09:29 2013

@author: Kyle Weber
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
import pandas as pd
import scipy.signal as sig

day = 22
scopeNo = 2
chan = 2
shot = 0

mydata = open("J:/SURE/sparkData_2013/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r")
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
rx = pd.Series(j[0])
ry = pd.Series(j[1], index = j[0])

plt.plot(j[0],j[1])
pd.rolling_mean(ry,20).plot()

"def findspikes(x):"
    

"""p = trapz(j[1],j[0])
print p"""



"""line = plt.plot(x,y,'b-')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))
"""
mydata.close()
