# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:06:38 2013

@author: Zach
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import conf
import scipy.integrate as si
import toolbox

#day = 22
#scopeNo = 3
#chan = 2
#shot = 0


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
            if i == True and z[index_count-1] == True:
                start = x[index_count-1]
                s_index = index_count
                break
        for i in z[index_count:]:
            index_count += 1
            if z[index_count-1] == False and z[index_count-2] == True:
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

spikex = []
spikey = []
final_results = []
n_smooth = 25
significance = 12


for day in range(22,27):
    if day == 22:
        for shot in range(0,150):
            for scopeNo in range(2,4):
                for chan in range(1,5):
                    
                    y = toolbox.readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
                    #y = readData("C:/Sparks/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
                    
                    spikex, spikey = y.Time, toolbox.threshold_Kevin(y, significance, n_smooth)
                    results = time_intervals(spikex,spikey)
                    final_results.append(results)
                    
    elif day == 23:
        for shot in range(0,200):
            for scopeNo in range(2,4):
                for chan in range(1,5):
                    
                    y = toolbox.readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
                    #y = readData("C:/Sparks/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
                    
                    spikex, spikey = y.Time, toolbox.threshold_Kevin(y, significance, n_smooth)
                    results = time_intervals(spikex,spikey)
                    final_results.append(results)
    elif day == 24 or day == 25:
        for shot in range(0,300):
            for scopeNo in range(2,4):
                for chan in range(1,5):
                    
                    y = toolbox.readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
                    #y = readData("C:/Sparks/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
                    
                    spikex, spikey = y.Time, toolbox.threshold_Kevin(y, significance, n_smooth)
                    results = time_intervals(spikex,spikey)
                    final_results.append(results)
                    """
    elif day == 26:
        for shot in range(0,6022):
            for scopeNo in range(2,4):
                if scopeNo == 2:
                    chan = 3
                    
                    
                    y = toolbox.readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
                    #y = readData("C:/Sparks/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
                    
                    spikex, spikey = y.Time, toolbox.threshold_Kevin(y, significance, n_smooth)
                    results = time_intervals(spikex,spikey)
                    final_results.append(results)
                elif scopeNo == 3:
                    chan = 1
                    
                    y = toolbox.readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
                    #y = readData("C:/Sparks/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
                    
                    spikex, spikey = y.Time, toolbox.threshold_Kevin(y, significance, n_smooth)
                    results = time_intervals(spikex,spikey)
                    final_results.append(results)
    """

spike_info = toolbox.list_to_frame(final_results)

print spike_info

"""
plt.plot(y.Time,y.Ampl)
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))

plt.plot(spikex,spikey,'r-')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))
plt.show()

"""