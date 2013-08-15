# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:23:22 2013

@author: Zach
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

a = pd.load("C:/Program Files (x86)/pythonxy/a_programs/spike_info")
b = pd.load("C:/Program Files (x86)/pythonxy/a_programs/spike_info_sig200")
z = pd.load('C:/Users/Zach.Zach-PC/Documents/GitHub/spark/SparkSpikeStuff')

d = a.set_index(['Day', 'Scope', 'Channel', 'Shot']) #heirachrically indexing dataframe a

#Creating a dataframe of the calibration data.  Repeating same processes done before.
ci = d.Peak.ix[25].ix[3].groupby(level =['Shot', 'Channel']).min()
ciu = ci.unstack(1)
ciu = ciu.fillna(0)

ciu = ciu[ciu[1] > -0.46][ciu[2] > -0.44][ciu[3] > -0.43][ciu[4] > -0.043] #Removing saturated data from each channel

statistics = []
#iiu = iiu[iiu[1] < 2.5e-9][iiu[2] < 2.5e-9][iiu[3] < 2.5e-9][iiu[4] < 2.5e-9]
#Two loops to plot channels against each other
count = 0  
for e in range(1,5): 
    for f in range(1,5):
        if f > e: #f and e both act as variables for a channel.  This line makes sure that there are no redundant plots.
            count += 1
            fig = plt.figure(count) #Creates a separate figure for each plot
            
            plt.scatter(-ciu[f][0:50], -ciu[e][0:50]) #Creates a scatter plot of the two channels
            plt.xlabel('Scope 3 Channel ' + str(f) + ' (UB' + str(f) + ') Amplitude')
            plt.ylabel('Scope 3 Channel ' + str(e) + ' (UB' + str(e) + ') Amplitude')
            plt.ylim(-0.05, 0.3)
            if f == 4:
                plt.xlim(-0.005, 0.03)
            else:
                plt.xlim(-0.05, 0.3)
            plt.title("Cross Calibration of UB" + str(f)+ " and UB" + str(e) + " on day 25")            
            
            slope, intercept, r_value, p_value, error= stats.linregress(-ciu[f][0:50],-ciu[e][0:50]) #Gives calibration statistics for each plot
            plt.plot([0, 0.5], [intercept, (slope / 2) + intercept], "r-") #Plots a calibration line over the scatter plot
            statistics.append("UB" + str(f))
            statistics.append("UB" + str(e))
            statistics.append(slope)
            statistics.append(intercept)
            statistics.append(r_value)
            statistics.append(p_value)
            #Reporting statistics for the plot  
            print "---"
            print "X-axis: Channel " + str(f) + "  Y-axis: Channel " + str(e) 
            print "Slope: " + str(slope) + "  Intercept: " + str(intercept)
            print "r value: " + str(r_value) + "  p value: " + str(p_value)
            #plt.savefig(str(f) + "_versus_" +str(e) + "_with_calibration.png")
    f = 0
    
hi = d.Peak.ix[25].ix[2].groupby(level =['Shot', 'Channel']).min()
hiu = hi.unstack(1)
hiu = hiu.fillna(0)
hiu = hiu[hiu[3] > -0.7]

fig = plt.figure(13)
plt.scatter(-hiu[3][0:50], -hiu[4][0:50]) #Creates a scatter plot of the two channels
plt.xlabel('Scope 2 Channel ' + str(3) + ' (H1) Amplitude')
plt.ylabel('Scope 2 Channel ' + str(4) + ' (H3) Amplitude')
plt.ylim(-0.05, 0.6)
plt.xlim(-0.05, 0.6)
plt.title("Cross Calibration of H1 and H3 on day 25")

slope, intercept, r_value, p_value, error= stats.linregress(-hiu[3][0:50],-hiu[4][0:50]) #Gives calibration statistics for each plot
plt.plot([0, 0.5], [intercept, (slope /2) + intercept], "r-") #Plots a calibration line over the scatter plot
statistics.append("H1")
statistics.append("H3")
statistics.append(slope)
statistics.append(intercept)
statistics.append(r_value)
statistics.append(p_value)  

print "---"
print "X-axis: Scope 2 Channel " + str(3) + "  Y-axis: Scope 2 Channel " + str(4) 
print "Slope: " + str(slope) + "  Intercept: " + str(intercept)
print "r value: " + str(r_value) + "  p value: " + str(p_value)
      
fig = plt.figure(14)
plt.scatter(-ciu[1][0:50], -hiu[3][0:50]) #Creates a scatter plot of the two channels
plt.xlabel('Scope 3 Channel ' + str(1) + ' (UB1) Amplitude')
plt.ylabel('Scope 2 Channel ' + str(3) + ' (H1) Amplitude')
plt.ylim(-0.05, 0.6)
plt.xlim(-0.05, 0.6)
plt.title("Cross Calibration of UB1 and H1 on day 25")

slope, intercept, r_value, p_value, error= stats.linregress(-ciu[1][0:50],-hiu[3][0:50]) #Gives calibration statistics for each plot
plt.plot([0, 0.5], [intercept, (slope /2) + intercept], "r-") #Plots a calibration line over the scatter plot
statistics.append("UB1")
statistics.append("H1")
statistics.append(slope)
statistics.append(intercept)
statistics.append(r_value)
statistics.append(p_value)  
    
print "---"
print "X-axis: Scope 3 Channel " + str(1) + "  Y-axis: Scope 2 Channel " + str(3) 
print "Slope: " + str(slope) + "  Intercept: " + str(intercept)
print "r value: " + str(r_value) + "  p value: " + str(p_value)

def dataFrame(r):
    cols = ['X-axis', 'Y-axis', 'Slope', 'Intercept', 'R-value', 'P-value']
    num = len(r) / 6
    a = np.array(r).reshape(num, 6)
    df = pd.DataFrame(data = a, columns = cols)
    return df

results = dataFrame(statistics)
print results





