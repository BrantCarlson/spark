# -*- coding: utf-8 -*-
"""
Created on Tue Jul 09 11:53:18 2013

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
   

def thresh(j,roll,noise,sig):
    cut = len(j)/noise
    ave = j.Ampl[:cut].mean()
    offset = j.Ampl - ave
    std = np.sqrt( (j.Ampl[:cut].var()) / roll)
    slope = pd.rolling_mean(offset,roll*2)
    #plt.plot(slope)
    #plt.plot(r)
    if any(slope < -sig*std):
        return True
    else: return False
            


    
def find(day,shot,scopeNo,chan):
    roll = 20
    j = readData(day,shot,scopeNo,chan)
    rj = j.iloc[roll:len(j)-roll]  
    spike = pd.rolling_mean(rj,roll*2)
    count = 1
    frame = pd.DataFrame()
    noise = 5
    cut = len(j)/noise
    ave = j.Ampl[:cut].mean()
    j.Ampl[cut] = ave
    j.Ampl[len(j)-cut:] = ave
    
    while thresh(j,2,5,6.7) == True:
        
        peak_amp = pd.DataFrame.min(j)[1]   
        peak_time_i = pd.DataFrame.idxmin(j)[1]
        peak_time = j['Time'][peak_time_i]
        #plt.plot(j)
        
        for i in range(peak_time_i-150,58,-1):
            front_der = (spike['Ampl'][i] - spike['Ampl'][i - 1]) / (spike['Time'][i] - spike['Time'][i - 1])
            a = 0        
            if any(front_der >= 0 or j.Ampl <= 0):
                start = spike['Time'][i]
                #print "\nStarts at " + str(start)
                a = i
                break
        for i in range(peak_time_i + 150,len(j),1):
            end = 0        
            back_der = ( (spike['Ampl'][i + 1] - spike['Ampl'][i]) / (spike['Time'][i + 1] - spike['Time'][i]) )
            b = 0
            if back_der <= 0:
                end = spike['Time'][i]
                #print "Ends at " + str(end)
                b = i
                break
        intg = np.trapz(j.Ampl[a:b])
        #print "Spike duration of " + str(end - start) + " seconds"
        #print "Spike peak at t=" + str(peak_time) + " and amplitude " + str(peak_amp) + " millivolts"
        #print "Area under spike of " + str(intg) + " nWb"            
        j['Ampl'][a:b] = 0
        #plt.plot(j)
        sp = 'sh%dosc%dchn%dspk%d' % (shot,scopeNo,chan,count)
        d = {
        'shot':["%05d" %(shot)],
        'chan':[chan],
        'osc':[scopeNo],
        'day':[day],
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
        count = 10


#chd = find(22,0,2,3)
def findfor(day,shot,scopeNo):
    chf = pd.DataFrame()
    for chan in range(1,5):
        indv = find(day,shot,scopeNo,chan)
        chf = chf.append(indv)
    return chf
    
def findforshot(day,shot):
    chf = pd.DataFrame()
    for scopeNo in range(2,4):
        for chan in range(1,5):
            indv = find(day,shot,scopeNo,chan)
            chf = chf.append(indv)
    return chf
    
def findforday(day):
    chf = pd.DataFrame()
    for shot in range(150):
        for scopeNo in range(2,4):
            for chan in range(1,5):
                indv = find(day,shot,scopeNo,chan)
                chf = chf.append(indv)
                print shot,scopeNo,chan,len(chf)
    return chf




def findforall(day1,day2):
    chf = pd.DataFrame()
    for day in range(day1,day2):
        if day == 22:
            for shot in range(150):
                for scopeNo in range(2,4):
                    for chan in range(1,5):
                        indv = find(day,shot,scopeNo,chan)
                        chf = chf.append(indv)
                        print day,shot,len(chf)
    
        if day == 23:
            for shot in range(200):
                for scopeNo in range(2,4):
                    for chan in range(1,5):
                        indv = find(day,shot,scopeNo,chan)
                        chf = chf.append(indv)
                        print day,shot,len(chf)

        if day == 24:
            for shot in range(300):
                for scopeNo in range(2,4):
                    for chan in range(1,5):
                        indv = find(day,shot,scopeNo,chan)
                        chf = chf.append(indv)
                        print day,shot,len(chf)
                        
        if day == 25:
            for shot in range(300):
                for scopeNo in range(2,4):
                    for chan in range(1,5):
                        indv = find(day,shot,scopeNo,chan)
                        chf = chf.append(indv)
                        print day,shot,len(chf)
            return chf
        """if day == 26:
            for shot in range(6023):
                for scopeNo in range(2,4):
                    if scopeNo == 2:
                        chan = 3
                    elif scopeNo == 3:
                        chan = 1
                    indv = find(day,shot,scopeNo,chan)
                    chf = chf.append(indv)
                    print day,shot,len(chf)"""


z = findforall(22,27)
print z