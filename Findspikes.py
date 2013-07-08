# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 15:49:54 2013

@author: Kyle Weber
"""

import pandas as pd
import matplotlib.pyplot as plt

def readData(day,shot,scopeNo,chan):

    with open("C:/Users/Kyle Weber/Documents/Carthage/SURE/sparkData_2013/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r") as f:    
        for i in range(4):        
            f.readline() 
            # read (and discard) the first 4 lines            
        x = pd.read_csv(f) 
            # then use Pandas' built-in comma-separated-variable function            
            # using the with ... construct, f is automatically closed.
    return x
    
dcon = readData(22,0,2,3)
dvar = readData(22,0,2,3)

def findspikes(dcon,dvar):
    plt.plot(dcon)
    rolld = dcon.iloc[20:len(dcon)-20]
    spike = pd.rolling_mean(rolld,40)
    peak_amp = pd.DataFrame.min(dvar)[1] #this one changes
    peak_time_i = pd.DataFrame.idxmin(dvar)[1] #this one doesn't
    peak_time = dvar['Time'][peak_time_i] #neither does this one
    while peak_amp < (pd.Series.mean(dcon['Ampl']) - pd.Series.std(dcon['Ampl'])):
        plt.plot(dvar)
        for i in range(peak_time_i-150,58,-1):
            front_der = (spike['Ampl'][i] - spike['Ampl'][i - 1]) / (spike['Time'][i] - spike['Time'][i - 1])
            a = 0        
            if front_der >= 0:
                start = spike['Time'][i]
                print "\nStarts at " + str(start)
                a = i
                break
        for i in range(peak_time_i + 150,len(dcon),1):
            end = 0        
            back_der = ( (spike['Ampl'][i + 1] - spike['Ampl'][i]) / (spike['Time'][i + 1] - spike['Time'][i]) )
            b = 0
            if back_der <= 0:
                end = spike['Time'][i]
                print "Ends at " + str(end)
                b = i
                break
        print "Spike duration of " + str(end - start) + " seconds"
        print "Spike peak at t=" + str(peak_time) + " and amplitude " + str(peak_amp) + " millivolts"     
        dvar['Ampl'][a:b] = 0
        return dvar

count = 1
while count < 5:
    findspikes(dcon,dvar)
    count += 1
