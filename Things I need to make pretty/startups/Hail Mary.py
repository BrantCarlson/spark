# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:55:09 2013

@author: Kyle Weber
"""
import matplotlib.pyplot as plt
import numpy as np

def data_for_shot(day,shot):
    scopex = []
    scopey = []
    chanx = []
    chany = []
    for scopeNo in range(1,4):    
        for chan in range(1,5):
            mydata = open("J:/SURE/sparkData_2013/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r")
            plot = mydata.readlines()
            data = plot[5:]
            
    
            i = 0
            xplot = []
            yplot = []
            while i < len(data):
                singlepoint = data[i].split(',')
                xplot == xplot.append(float(singlepoint[0]))
                yplot == yplot.append(float(singlepoint[1]))
                i += 1
            chanx.append(xplot)
            chany.append(yplot)
            print scopeNo, chan, shot, xplot[556]
        scopex.append(chanx)
        scopey.append(chany)
        mydata.close()
    return (scopex,scopey)
        
data = data_for_shot(22,0)


plt.figure(4)
plt.subplot(411)
plt.plot(((data[0][0][0])),(data[1][0][0]))
plt.ylabel("Amplitude")
plt.xlabel("Time")


plt.subplot(412)
plt.plot(((data[0][0][1])),(data[1][0][1]))
plt.ylabel("Amplitude")
plt.xlabel("Time")


plt.subplot(413)
plt.plot(((data[0][0][2])),(data[1][0][2]))
plt.ylabel("Amplitude")
plt.xlabel("Time")


plt.subplot(414)
plt.plot(((data[0][0][3])),(data[1][0][3]))
plt.ylabel("Amplitude")
plt.xlabel("Time")




plt.figure(5)
plt.subplot(411)
plt.plot(((data[0][0][4])),(data[1][0][4]))
plt.ylabel("Amplitude")
plt.xlabel("Time")


plt.subplot(412)
plt.plot(((data[0][0][5])),(data[1][0][5]))
plt.ylabel("Amplitude")
plt.xlabel("Time")


plt.subplot(413)
plt.plot(((data[0][0][6])),(data[1][0][6]))
plt.ylabel("Amplitude")
plt.xlabel("Time")


plt.subplot(414)
plt.plot(((data[0][0][7])),(data[1][0][7]))
plt.ylabel("Amplitude")
plt.xlabel("Time")




plt.figure(6)
plt.title("Sensor 3 Data")
plt.subplot(411)
plt.plot(((data[0][0][8])),(data[1][0][8]))
plt.ylabel("Amplitude")
plt.xlabel("Time")


plt.subplot(412)
plt.plot(((data[0][0][9])),(data[1][0][9]))
plt.ylabel("Amplitude")
plt.xlabel("Time")


plt.subplot(413)
plt.plot(((data[0][0][10])),(data[1][0][10]))
plt.ylabel("Amplitude")
plt.xlabel("Time")


plt.subplot(414)
plt.plot(((data[0][0][11])),(data[1][0][11]))
plt.ylabel("Amplitude")
plt.xlabel("Time")

o = np.zeros( (8,8) )
