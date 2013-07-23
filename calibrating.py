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

#Creating a dataframe of the useful data from the radial shots
ai = d.Peak.ix[23].ix[3].groupby(level =['Shot', 'Channel']).min() #a dataframe of the biggest peak in each shot/channel combination on day 23, scope 3
aiu = ai.unstack(1) #Channels are now the columns, shot number is the index
aiu = aiu.fillna(0) #Replacing all NaNs with 0

#Creating a dataframe of the calibration data.  Repeating same processes done before.
ci = d.Peak.ix[25].ix[3].groupby(level =['Shot', 'Channel']).min()
ciu = ci.unstack(1)
ciu = ciu.fillna(0)

ciu = ciu[ciu[1] > -0.46][ciu[2] > -0.44][ciu[3] > -0.43][ciu[4] > -0.043] #Removing saturated data from each channel

#Two loops to plot channels against each other
count = 0  
for e in range(1,5): 
    for f in range(1,5):
        if f > e: #f and e both act as variables for a channel.  This line makes sure that there are no redundant plots.
            count += 1
            fig = plt.figure(count) #Creates a separate figure for each plot
            #To plot data from radial shots instead of calibration shots,
            #change ciu to aiu and change 0:50 to 0:100 in the plt.scatter function
            plt.scatter(-ciu[f][0:50], -ciu[e][0:50]) #Creates a scatter plot of the two channels
            plt.xlabel('Channel ' + str(f) + ' Amplitude')
            plt.ylabel('Channel ' + str(e) + ' Amplitude')
            
            slope, intercept, r_value, p_value, error= stats.linregress(-ciu[f][0:50],-ciu[e][0:50]) #Gives calibration statistics for each plot
            plt.plot([0, 0.5], [intercept, slope / 2], "r-") #Plots a calibration line over the scatter plot
            #Reporting statistics for the plot            
            print "---"
            print "X-axis: Channel " + str(f) + "  Y-axis: Channel " + str(e) 
            print "Slope: " + str(slope) + "  Intercept: " + str(intercept)
            print "r value: " + str(r_value) + "  p value: " + str(p_value)
            #plt.savefig(str(f) + "_versus_" +str(e) + "_with_calibration.png")
    f = 0