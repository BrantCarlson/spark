# -*- coding: utf-8 -*-
"""
Created on Mon Jul 08 12:34:36 2013

@author: Zach
"""
import numpy as np
import pandas as pd
import scipy as si
import matplotlib as plt
import conf
"""
for day in range(22,27):
    if day == 22:
        for shot in range(0,150):
            for scopeNo in range(2,4):
                for chan in range(1,5):
                    asdf = 0
    elif day == 23:
        for shot in range(0,200):
            for scopeNo in range(2,4):
                for chan in range(1,5):
                    asdf = 0
    elif day == 24 or day == 25:
        for shot in range(0,300):
            for scopeNo in range(2,4):
                for chan in range(1,5):
                    asdf = 0
    elif day == 26:
        for shot in range(0,6022):
            for scopeNo in range(2,4):
                if scopeNo == 2:
                    chan = 3
                    
                elif scopeNo == 3:
                    chan = 1
"""
day = 22
scopeNo = 3
chan = 2
shot = 0

def readData(filename):
    data = []
    with open(filename,'r') as f:
        for x in range(4):
            f.readline()
        data = pd.read_csv(f) 
        return data

y = readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))

#Zack's version of threshold
def threshold(y):
                smoothed = pd.rolling_mean(y.Ampl,25)
                deviation = y.Ampl.std()
                return 0.05*(smoothed < -deviation)

#Zack's version of time_intervals
def time_intervals(x,z):
    #takes two lists as arguments
    #Defining variables
    results = []
    start = 0.0
    end = 0.0
    duration = 0.0
    index_count = 0
    s_index = 0
    e_index = 0
    peak = 0.0
    #Loops through indices and looks for beginnings and ends of spikes
    while index_count < len(z) - 2:
        for i in z[index_count:]:
            index_count += 1
            if i == 0.05 and z[index_count-1] == 0.05:
                start = x[index_count-1]
                s_index = index_count
                break
        for i in z[index_count:]:
            index_count += 1
            if z[index_count-1] == 0.00 and z[index_count-2] == 0.05:
                end = x[index_count-1]
                e_index = index_count
                break
            elif index_count >= len(z) - 5:
                #Breaks if no end has been found for a spike by the end of z
                end = x[index_count-1]
                e_index = index_count
                break
        duration = end - start
        peak = y.Ampl[s_index:e_index].min()
        integral_a = si.trapz(y[s_index:e_index])
        integral_b = np.sum(integral_a)
        if duration > 3.0e-9 and start != 0:
            #Checks the duration of the spike to throw out any false positives
            #that would come from noise.  If the spike is not a false positive
            #the data is appended to results and printed
            results.append(start)
            results.append(end)
            results.append(peak)
            results.append(duration)
            results.append(day)
            results.append(shot)
            results.append(scopeNo)
            results.append(chan)
            results.append(integral_b)
            print "Spike Duration: " + str(duration) + " seconds."
            print "Peak: " + str(peak)
            print "Start: " + str(start) + " seconds.", "End: " + str(end) + " seconds."
            print str(day) , str(shot) , str(scopeNo) , str(chan)
            #Resets start and end so as to not report the last spike twice
            start = 0
            end = 0
    return results


def list_to_frame(r):
    #Converts a list of lists into a data frame
    h = []
    for e in r:
        for q in e:
            h.append(q)
    a = []
    row_len = 0
    cols = []
    cols = ['Start', 'End', 'Peak', 'Duration', 'Day', 'Shot', 'Scope', 'Channel', 'Integral']
    row_len = len(h) / 9
    a = np.array(h).reshape(row_len, 9)
    df = pd.DataFrame(data = a, columns = cols)
    return df
    
#Kevin's threshold
def threshold_Kevin(y,sig=2,smoothPts=3):
    """Find regions where the amplitude variable is above threshold away from the mean.
       The threshold is defined in terms of the standard deviation (i.e. width) of the noise
       by the significance (sig).  I.e. a spike is significant if the rolling mean of the data
       (taken with the window smoothPts) is above (standard deviation)*sig/sqrt(smoothPts)."""
    m = y.Ampl[:len(y)/4].mean()
    s = np.sqrt(y.Ampl[:len(y)/4].var())
    return pd.rolling_mean(y.Ampl-m,smoothPts) < -s*sig/np.sqrt(smoothPts)
    
#Kevin's time_intervals function
def time_intervals_Kevin(x,z):
    #takes two lists as arguments
    #Defining variables
    results = []
    start = 0.0
    end = 0.0
    duration = 0.0
    index_count = 0
    s_index = 0
    e_index = 0
    peak = 0.0
    #Loops through indices and looks for beginnings and ends of spikes
    while index_count < len(z) - 2:
        for i in z[index_count:]:
            index_count += 1
            if i == True and z[index_count+1] == True:
                start = x[index_count]
                s_index = index_count
                break
        for i in z[index_count:]:
            index_count += 1
            if (z[index_count] == False and z[index_count-1] == True) or index_count >= len(z)-2:
                end = x[index_count-1]
                e_index = index_count
                break
        duration = end - start
        peak = y.Ampl[s_index:e_index].min()
        #Throws out any false-positives from noise
        
        integral_a = si.trapz(y[s_index:e_index])
        integral_b = np.sum(integral_a)
        
        if duration > 2.5e-9 and start != 0:
            results.append(start)
            results.append(end)
            results.append(peak)
            results.append(duration)
            results.append(day)
            results.append(shot)
            results.append(scopeNo)
            results.append(chan)
            results.append(integral_b)
        #Resets start and end so as to not report the last spike twice
        start = 0
        end = 0
    return results

n_smooth = 25
significance = 12

#For turning a list of values into a dataframe
def dataFrame(r):
    cols = ['Start', 'End', 'Peak', 'Duration', 'Day', 'Shot', 'Scope', 'Channel', 'Integral']
    num = len(r) / 9
    a = np.array(r).reshape(num, 9)
    df = pd.DataFrame(data = a, columns = cols)
    return df
    
#Function from Kyle's Final Spikefinder
def find(j):
    thresh = pd.Series.mean(j.Ampl) - pd.Series.std(j.Ampl)
    rj = j.iloc[20:len(j)-20]  
    spike = pd.rolling_mean(rj,40)
    count = 0
    
    while count < 100:
        
        peak_amp = pd.DataFrame.min(j)[1]   
        peak_time_i = pd.DataFrame.idxmin(j)[1]
        peak_time = j['Time'][peak_time_i]
        
        if peak_amp < thresh:
            plt.plot(j)
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
            print "Spike duration of " + str(end - start) + " seconds"
            print "Spike peak at t=" + str(peak_time) + " and amplitude " + str(peak_amp) + " millivolts"     
            j['Ampl'][a:b] = 0
            count += 1
        else: count = 100

#Kyle's readData
def readData_Kyle(day,shot,scopeNo,chan):

    with open(conf.kdataDir + "/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r") as f:    
        for i in range(4):        
            f.readline() 
            # read (and discard) the first 4 lines            
        x = pd.read_csv(f) 
            # then use Pandas' built-in comma-separated-variable function            
            # using the with ... construct, f is automatically closed.
    return x