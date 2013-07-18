# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:32:37 2013

@author: Zach
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = pd.load("C:/Program Files (x86)/pythonxy/a_programs/spike_info")
b = pd.load("C:/Program Files (x86)/pythonxy/a_programs/spike_info_sig200")
z = pd.load('C:/Users/Zach.Zach-PC/Documents/GitHub/spark/SparkSpikeStuff')

a.shot = np.array(map(int, a.Shot))

ai = a.set_index(['Day', 'Scope', 'Channel', 'Shot'])

sel = np.logical_and(a.Day == 24, a.Scope == 3)
sc = a[a.Day == 24][a.Scope == 3]
ai = ai.Peak.ix[22].groupby(level =['Shot', 'Channel']).min()
aiu = ai.unstack(1)
aiu = aiu.fillna(0)
print aiu.head()


"""
#kp.ix[:,3,:,:]
s = 2
c = 1
bins = 100
while s < 4:
    #if s == 3:
        #bins = 100
    while c < 5:
        plt.figure(s+2)
        ax = plt.subplot(4,1,c)
#plt.plot(a.Start[a.Scope == 2][a.Channel == 1], a.Peak[a.Scope == 2][a.Channel == 1], 'bo')
        if s == 2 and (c == 1 or c == 2):
            cut = 0.35
        if s == 2 and c == 3:
            cut = 0.695
        if s == 2 and c == 4:
            cut = 0.159
        if s == 3 and c == 1:
            cut = 0.469
        if s == 3 and c == 2:
            cut = 0.45
        if s == 3 and c == 3:
            cut = 0.44
        if s == 3 and c == 4:
            cut = 0.043
        plt.hist(-a.Peak[a.Scope == s][a.Channel == c][-a.Peak < cut], bins)
        #plt.loglog()
        plt.subplots_adjust(hspace=0.45, bottom=0.125)
        #plt.axis([10e-5, 10e-1, 10e-1, 10e3])
        c+=1
    s+=1
    c = 1
#plt.plot(z.start, z.amp, 'bo')
#plt.xlabel('Start')
#plt.ylabel('Amplitude')
#plt.plot(-b.Peak, b.Duration, 'ro')


sel = np.logical_and(spike_info.Start < 1.55171e-7, spike_info.Channel == 2)
index = -1
for n in sel:
    index += 1
    if n == True:
        print spike_info.irow(index)
"""