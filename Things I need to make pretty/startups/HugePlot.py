# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:57:55 2013

@author: Kyle Weber
"""

#import matplotlib.pyplot as plt



def data_for_day(day,shot):
    for scopeNo in range(1,4):
        for chan in range(1,5):
            for shot in range(150):
                mydata = open("J:/SURE/sparkData_2013/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r")
                plot = mydata.readlines()
                data = plot[5:]
                mydata.close()
            return data
            
            
"""def splitit(data):
    i = 0
    xplot = []
    yplot = []
    while i < len(data):
        singlepoint = data[i].split(',')
        xplot == xplot.append(float(singlepoint[0]))
        yplot == yplot.append(float(singlepoint[1]))
        i += 1
    return (xplot,yplot)
(x,y) = splitit(data_for_day(...))


line = plt.plot(x,y,linewidth=0.2)
plt.ylabel("Amplitude")
plt.xlabel("Time")


print data_for_day(22)"""
            
            


