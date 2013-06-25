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

mydata.close()

j = np.array((x,y))
rx = pd.Series(j[0])
ry = pd.Series(j[1], index = j[0])
rsy = pd.Series(j[1][20:], index = j[0][:9982])


plt.plot(j[0],j[1])
#spike = pd.rolling_mean(ry,41)



def findspikes(j):
    start = 0
    end = 0
    peak_amp = np.min(j[1])
    "This is the amplitude of spike's peak"
    peak_time_i = np.argmin(j[1]) 
    "This gives the i value for the spike's peak"
    peak_time = j[0][peak_time_i]
    spike = pd.rolling_mean(rsy,40)
    spike.plot()
    for i in range(peak_time_i,-1,-1):
        "derivative = change in peak height     /     change in time"
        front_der = ( (spike[i] - spike[i - 1]) / (j[0][i] - j[0][i - 1]))
        if front_der >= 0:
            start = j[0][i]
            print "Starts at " + str(j[0][i])
            break
    for i in range(peak_time_i + 10,len(j[1]),1):
        back_der = ( (spike[i + 1] - spike[i]) / (j[0][i + 1] - j[0][i]) )
        if back_der <= 0:
            print back_der
            end = j[0][i]
            print "Ends at " + str(j[0][i])
            break  
    print "Spike duration of " + str(end - start) + " seconds"
    print "Spike peak at t=" + str(peak_time) + " and amplitude " + str(peak_amp) + " millivolts"
   
   
   
   
   
   
   
   
   
   
   
   
"""
start = 0
   end = 0
   peakx = np.argmin(j[1])
   peaky = np.min(j[1])
   spike = pd.rolling_mean(ry,35)
   spike.plot()
   for i in range(peakx,-1,-1):
       deriv = (spike[i + 1] - spike[i - 1]) / (j[0][i + 1] - j[0][i - 1])
       if deriv >= 0:
           start = j[0][i]
           print "Starts at " + str(j[0][i])
           break
   for i in range(peakx,10002,1):
       deriv = (spike[i + 1] - spike[i - 1]) / (j[0][i + 1] - j[0][i - 1])
       if deriv >= 1000000:
           end = j[0][i]
           print "Ends at " + str(j[0][i])
           break  
   print "Spike duration of " + str(end - start) + " seconds"
   print "Spike peak at t=" + str(j[0][peakx]) + " and amplitude " + str(peaky) + " millivolts"
"""
  
       
findspikes(j)
       

    
    
    

#p = trapz(j[1],j[0])
#$print p



#line = plt.plot(x,y,'b-')
#plt.ylabel("Amplitude")
#plt.xlabel("Time")
#plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))