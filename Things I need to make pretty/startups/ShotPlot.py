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
        scopex.append(chanx)
        scopey.append(chany)
        mydata.close()
        return (scopex,scopey)
        
data_for_shot(22,0)

plt.figure(1)
plt.subplot(411)
plt.plot(scopex(chanx[0]),scopey(chany[0]))

plt.subplot(412)
plt.plot(scopex(chanx[1]),scopey(chany[1]))

plt.subplot(413)
plt.plot(scopex(chanx[2]),scopey(chany[2]))

plt.subplot(414)
plt.plot(scopex(chanx[3]),scopey(chany[3]))






        



            