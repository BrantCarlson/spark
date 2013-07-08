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

day = 22
#scopeNo = 3
#chan = 2
#shot = 0

spikex = []
spikey = []
final_results = []


for shot in range(0,150):
    for scopeNo in range(2,4):
        for chan in range(1,5):
            y = toolbox.readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
            #y = readData("C:/Sparks/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))
            
            index_count = 0
                
            
            spikex, spikey = toolbox.threshold(y)
            results = toolbox.time_intervals(spikex,spikey)
            final_results.append(results)
        

        
    
spike_info = toolbox.list_to_frame(final_results)

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

