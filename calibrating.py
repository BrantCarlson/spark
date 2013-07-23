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

a.shot = np.array(map(int, a.Shot))

d = a.set_index(['Day', 'Scope', 'Channel', 'Shot'])

#sel = np.logical_and(a.Day == 24, a.Scope == 3)
#sc = a[a.Day == 24][a.Scope == 3]
ai = d.Peak.ix[23].ix[3].groupby(level =['Shot', 'Channel']).min()
aiu = ai.unstack(1)
aiu = aiu.fillna(0)

ci = d.Peak.ix[25].ix[3].groupby(level =['Shot', 'Channel']).min()
ciu = ci.unstack(1)
ciu = ciu.fillna(0)

ciu = ciu[ciu[1] > -0.46][ciu[2] > -0.44][ciu[3] > -0.43][ciu[4] > -0.043]

count = 0
for e in range(1,5):
    for f in range(1,5):
        if f > e:
            count += 1
            fig = plt.figure(count)
            plt.scatter(-ciu[f][0:50], -ciu[e][0:50])
            plt.xlabel('Channel ' + str(f) + ' Amplitude')
            plt.ylabel('Channel ' + str(e) + ' Amplitude')
            
            slope, intercept, r_value, p_value, error= stats.linregress(-ciu[f][0:50],-ciu[e][0:50])
            plt.plot([0, 0.5], [intercept, slope / 2], "r-")
            print "---"
            print "X-axis: Channel " + str(f) + "  Y-axis: Channel " + str(e) 
            print "Slope: " + str(slope) + "  Intercept: " + str(intercept)
            print "r value: " + str(r_value) + "  p value: " + str(p_value)
            #plt.savefig(str(f) + "_versus_" +str(e) + "_with_calibration.png")
    f = 0