# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:45:07 2013

@author: Kyle Weber
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import conf

def readData(day,shot,scopeNo,chan):

    with open(conf.dataDir + "/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r") as f:    
        for i in range(4):        
            f.readline() 
            # read (and discard) the first 4 lines            
        x = pd.read_csv(f) 
            # then use Pandas' built-in comma-separated-variable function            
            # using the with ... construct, f is automatically closed.
    return x
    
def find(day,shot,scopeNo,chan,roll,threshroll,noise,sig,fin):
    #plt.clf()
    j = -(readData(day,shot,scopeNo,chan))
    avg = j.Ampl[:(len(j)/noise)].mean()
    j.Ampl-=avg
    std = np.sqrt( j.Ampl[:(len(j)/noise)].var() / threshroll)
    thresh = sig*std
    j.Ampl[len(j)/noise] = avg
    j.Ampl[len(j)-(len(j)/noise):] = avg 
    
    spike = pd.rolling_mean(j,roll*2,center=True)
    #plt.plot(j)
    #plt.plot(spike)
    
    count = 1
    frame = pd.DataFrame()  
    
    
    while np.logical_and(any(spike.Ampl > sig*std),count < fin):  
    
        peak = j.Ampl.max()
        peaktime_i = j.Ampl.idxmax()
        peaktime = j.Time[peaktime_i]

        
        a = 0
        b = 0
        start = 0
        end = 0
        front_der = 0
        back_der = 0
        
        for i in range(peaktime_i-10,1,-1):

            front_der = ( spike.Ampl[i] - spike.Ampl[i-1] ) / ( spike.Time[i] - spike.Time[i-1] )
            if front_der > 0 or spike.Ampl[i] < thresh:
                a = i
                start = spike.Time[i]
                break
            
            
        for i in range(peaktime_i+10,len(j)-1,1):
            back_der = ( ((spike.Ampl[i+1] + spike.Ampl[i+2])/2) - ((spike.Ampl[i] + spike.Ampl[i-1])/2) / ( ((spike.Time[i+1] + spike.Time[i+2])/2) - ((spike.Time[i] + spike.Time[i-1])/2)))
            if back_der < 0 or spike.Ampl[i] < thresh:
                b = i
                end = spike.Time[i]
                break
        #print a,peaktime_i,b,back_der,front_der
        intg = np.trapz(j.Ampl[a:b])        
        
        j.Ampl[a:b] = 0
        #plt.plot(j)
        spike = pd.rolling_mean(j,roll*5,center=True)
        #plt.plot(spike)
        
        sp = 'sh%dosc%dchn%dspk%d' % (shot,scopeNo,chan,count)
        d = {
        'shot':["%05d" %(shot)],
        'chan':[chan],
        'osc':[scopeNo],
        'day':[day],
        'amp':[peak],
        'area':[intg],
        'timei':[peaktime_i],
        'time':[peaktime],
        'start':[start],
        'end':[end],
        'dur':[end-start]}
        df = pd.DataFrame(d,index = [sp])
        frame = frame.append(df)
        count += 1
    else:

        #plt.hlines(thresh,0,5000)
        #print thresh
        count = fin
        return frame
        
    


g = find(26,555,3,1,41,2,8,6,5)


