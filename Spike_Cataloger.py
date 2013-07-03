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

day = 22
#scopeNo = 3
#chan = 2
shot = 0

spikex = []
spikey = []
final_results = []

def readData(filename):
    data = []
    with open(filename,'r') as f:
        for x in range(4):
            f.readline()
        data = pd.read_csv(f) 
        return data
for scopeNo in range(2,4):
    for chan in range(1,5):
        y = readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
        #y = readData("C:/Sparks/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
        
        def threshold(y):
            smoothed = pd.rolling_mean(y.Ampl,25)
            deviation = y.Ampl.std()
            return y.Time, 0.05*(smoothed < -deviation)
            
            
        index_count = 0
        
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
            while index_count < len(z) - 1:
                for i in z[index_count:]:
                    index_count += 1
                    if i == 0.05 and z[index_count+1] == 0.05:
                        start = x[index_count]
                        s_index = index_count
                        break
                for i in z[index_count:]:
                    index_count += 1
                    if z[index_count] == 0.0 and z[index_count-1] == 0.05:
                        end = x[index_count-1]
                        e_index = index_count
                        break
                    elif index_count >= len(z) - 3:
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
                    #Resets start and end so as to not report the last spike twice
                start = 0
                end = 0
            return results
        spikex, spikey = threshold(y)
        results = time_intervals(spikex,spikey)
        final_results.append(results)
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
        
    
spike_info = list_to_frame(final_results)

print spike_info

plt.plot(y.Time,y.Ampl)
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))

plt.plot(spikex,spikey,'r-')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))
plt.show()

