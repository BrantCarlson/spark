# -*- coding: utf-8 -*-
"""
Created on Mon Jul 08 10:46:17 2013

@author: Kyle Weber
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import conf

def readData(day,shot,scopeNo,chan):

    with open(conf.kdataDir + "/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r") as f:    
        for i in range(4):        
            f.readline() 
            # read (and discard) the first 4 lines            
        x = pd.read_csv(f) 
            # then use Pandas' built-in comma-separated-variable function            
            # using the with ... construct, f is automatically closed.
    return x




def find(day,shot,scopeNo,chan):
    j = readData(day,shot,scopeNo,chan)
    thresh = pd.Series.mean(j.Ampl) - pd.Series.std(j.Ampl)
    rj = j.iloc[20:len(j)-20]  
    spike = pd.rolling_mean(rj,40)
    count = 1
    frame = pd.DataFrame()
    
    while count < 100:
        
        peak_amp = pd.DataFrame.min(j)[1]   
        peak_time_i = pd.DataFrame.idxmin(j)[1]
        peak_time = j['Time'][peak_time_i]
        plt.plot(j)
        
        if peak_amp < thresh:
            for i in range(peak_time_i-150,58,-1):
                front_der = (spike['Ampl'][i] - spike['Ampl'][i - 1]) / (spike['Time'][i] - spike['Time'][i - 1])
                a = 0        
                if front_der >= 0:
                    start = spike['Time'][i]
                    print "\nStarts at " + str(start)
                    a = i
                    break
            for i in range(peak_time_i + 150,len(j),1):
                end = 0        
                back_der = ( (spike['Ampl'][i + 1] - spike['Ampl'][i]) / (spike['Time'][i + 1] - spike['Time'][i]) )
                b = 0
                if back_der <= 0:
                    end = spike['Time'][i]
                    print "Ends at " + str(end)
                    b = i
                    break
            intg = np.trapz(j.Ampl[a:b])
            print "Spike duration of " + str(end - start) + " seconds"
            print "Spike peak at t=" + str(peak_time) + " and amplitude " + str(peak_amp) + " millivolts"
            print "Area under spike of " + str(intg) + " nWb"            
            j['Ampl'][a:b] = 0
            plt.plot(j)
            sp = 'sp' + str(count)
            d = {
            'amp':[peak_amp],
            'area':[intg],
            'time':[peak_time],
            'start':[start],
            'end':[end],
            'dur':[end-start]}
            df = pd.DataFrame(d,index = [sp])
            #dF = pd.DataFrame([peak_amp,peak_time,start,end,(end-start)], columns = ['amp','time','start','end','dur'])
            frame = frame.append(df)
            count += 1
        else:
            return frame
            count = 100
print find(22,0,2,3)

chd = find(22,0,2,3)