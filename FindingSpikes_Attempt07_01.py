# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:09:29 2013

@author: Kyle Weber
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
import pandas as pd
import conf

day = 22
scopeNo = 2
chan = 2
shot = 0

mydata = open(conf.dataDir + "/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r")
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
rsy = pd.Series(j[1][19:], index = j[0][:(len(j[1])-19)])


plt.plot(j[0],j[1])
#spike = pd.rolling_mean(ry,41)



def findspikes(j):
    peak_amp = np.min(j[1])
    "This is the amplitude of spike's peak"
    peak_time_i = np.argmin(j[1]) 
    "This gives the i value for the spike's peak"
    print peak_time_i
    peak_time = j[0][peak_time_i]
    spike = pd.rolling_mean(rsy,38)
    spike.plot()
    for i in range(peak_time_i-150,-1,-1):
        start = 0
        "derivative = change in peak height     /     change in time"
        front_der = ( (spike[i] - spike[i - 1]) / (j[0][i] - j[0][i - 1]))
        a = 0        
        if front_der >= 0:
            start = j[0][i]
            print "\nStarts at " + str(j[0][i])
            a = i
            break
    for i in range(peak_time_i + 150,len(j[1]),1):
        end = 0        
        back_der = ( (spike[i + 1] - spike[i]) / (j[0][i + 1] - j[0][i]) )
        b = 0
        if back_der <= 0:
            end = j[0][i]
            print "Ends at " + str(j[0][i])
            b = i
            break  
    print "Spike duration of " + str(end - start) + " seconds"
    print "Spike peak at t=" + str(peak_time) + " and amplitude " + str(peak_amp) + " millivolts"
    ab = np.array(range(a,b))   
    print "Area under spike of " + str(trapz(j[1][ab],j[0][ab]) * 10**9) + " nWb"
    return (a,b)

bases = findspikes(j)

front = np.array(range(0,bases[0]))
jfront = j[1][front]
jfronti = 0

back = np.array(range(bases[1],len(j[1])))
jback = j[1][back]
jbacki = bases[1]

    
#jfront = j[1] in range(0,spikebases[0])
#jback = j[1] in range(spikebases[1],len(j[1]))

def findmorespikes(j):
        fpeak_time_i = np.argmin(jfront)
        fpeak_time = j[0][fpeak_time_i]
        fpeak_amp = np.min(jfront)
        spike = pd.rolling_mean(rsy,40)
        if abs(fpeak_amp) > abs(ry.mean() - ry.std()):
            for i in range(fpeak_time_i-150,-1,-1):
                start = 0
                "derivative = change in peak height     /     change in time"
                front_der = ( (spike[i] - spike[i - 1]) / (j[0][i] - j[0][i - 1]))
                a = 0        
                if front_der >= 0:
                    start = j[0][i]
                    print "\nStarts at " + str(j[0][i])
                    a = i
                    break
            for i in range(fpeak_time_i + 150,len(j[1]),1):
                end = 0        
                back_der = ( (spike[i + 1] - spike[i]) / (j[0][i + 1] - j[0][i]) )
                b = 0
                if back_der <= 0:
                    end = j[0][i]
                    print "Ends at " + str(j[0][i])
                    b = i
                    break  
            print "Spike duration of " + str(end - start) + " seconds"
            print "Spike peak at t=" + str(fpeak_time) + " and amplitude " + str(fpeak_amp) + " millivolts"
            ab = np.array(range(a,b))   
            print "Area under spike of " + str(trapz(j[1][ab],j[0][ab]) * 10**9) + " nWb"
        else: print "\nNo spikes before largest"
        bpeak_time_i = np.argmin(jback)
        bpeak_time = j[0][bpeak_time_i]
        bpeak_amp = np.min(jback) 
        
        if abs(bpeak_amp) > abs(ry.mean() - ry.std()):
            for i in range(bpeak_time_i-150,-1,-1):
                start = 0
                "derivative = change in peak height     /     change in time"
                front_der = ( (spike[i] - spike[i - 1]) / (j[0][i] - j[0][i - 1]))
                a = 0        
                if front_der >= 0:
                    start = j[0][i]
                    print "\nStarts at " + str(j[0][i])
                    a = i + jbacki
                    break
            for i in range(bpeak_time_i + 150,len(j[1]),1):
                end = 0        
                back_der = ( (spike[i + 1] - spike[i]) / (j[0][i + 1] - j[0][i]) )
                b = 0
                if back_der <= 0:
                    end = j[0][i]
                    print "Ends at " + str(j[0][i])
                    b = i + jbacki
                    break  
            print "Spike duration of " + str(end - start) + " seconds"
            print "Spike peak at t=" + str(bpeak_time) + " and amplitude " + str(bpeak_amp) + " millivolts"
            ab = np.array(range(a,b))
            print "Area under spike of " + str(trapz(j[1][ab],j[0][ab]) * 10**9) + " nWb"
        else: print "\nNo spikes after largest"


        
   
findmorespikes(j)
  
   
   
   
   
   
   
   
   
   
  
       

    
    
    

#p = trapz(j[1],j[0])
#$print p



#line = plt.plot(x,y,'b-')
#plt.ylabel("Amplitude")
#plt.xlabel("Time")
#plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))