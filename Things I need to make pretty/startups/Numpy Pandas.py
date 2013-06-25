# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 05:51:25 2013

@author: Kyle Weber
"""

import matplotlib.pyplot as plt
import numpy as np
import pylab as py
from pandas import *

def data_for_shot(day,shot):
    scopex = []
    scopey = []
    chanx = []
    chany = []
    for scopeNo in range(1,4):    
        for chan in range(1,5):
            mydata = open("J:/SURE/sparkData_2013/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r")
            plot = mydata.readlines()
            'Turns raw data into ~20000 strings of the form (-2.001115e-007,0.0956727)'
            'This data comes from the individual spark files'
            data = plot[5:]
            "Returns only the data strings, now from 0 to just under 20000"
            """Returns 12 lists of these 20000 data strings:  
                                   sco1 chan1, sco1 chan2, sco1 chan3, sco1 chan4,
                                   sco2 chan1, sco2 chan2, sco2 chan3, sco2 chan4,
                                   sco3 chan1, sco3 chan2, sco3 chan3, sco3 chan4"""
            
    
            i = 0
            xplot = []
            yplot = []
            while i < len(data):
                singlepoint = data[i].split(',')
                "Splits the individual data strings for each channel"
                xplot.append(float(singlepoint[0]))
                yplot.append(float(singlepoint[1]))
                """This cuts the strings at the comma and says the first half
                goes is xplot, the second half is yplot, filling those lists"""
                i += 1
            chanx.append(xplot)
            chany.append(yplot)
            """After filling the whole xplot and yplot, those lists are placed
            into chanx and chany.  chanx = [[sco1 chan1], [sco1 chan2],...]"""
        scopex.append(chanx)
        scopey.append(chany)
        """The channel data is put into the scope lists. 
        scopex = [ [sco1[ [chan1],[chan2],[chan3],[chan4] ]] , [sco2[...]]]"""
        chanx = []
        chany = []
        """This resets chanx and chany, thereby separating the data from each
        of the scopes.  The overall data hierarchy goes 2>3>4:
        |2 axes|    for    |3 oscilloscopes|    with    |4 channels each|"""
        mydata.close()
    return (scopex,scopey)
    
def plot(x,y):
    data = data_for_shot(x,y)   
    r = np.array(data)
    for j in range(3):
        i = 0
        while i < 4:
            plt.figure(j + 1)
            figure(j+1).canvas.set_window_title('Jan %d, Shot %05d, Oscilloscope ' % (x, y) + str(j+1))
            plt.subplot(411 + (i))
            plt.plot(((r[0][j][i])),(r[1][j][i]))
            plt.ylabel("Amplitude")
            plt.xlabel("Time (microseconds)")
            i += 1



 
plot(22,0)

"Input the day and shot number into plot(x,y) to display graphs"