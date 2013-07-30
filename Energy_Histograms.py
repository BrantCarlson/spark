# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:08:50 2013

@author: Zach
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cal = pd.load('C:/Program Files (x86)/pythonxy/a_programs/day_26_data')#Loads spike data from day 26 gathered by Zack's spike finder


intbg = (-cal.Integral[cal.Scope == 2][cal.Shot <= 3006])#integral of the background spikes in scope 2 channel 3
intsr = (-cal.Integral[cal.Scope == 2][cal.Shot >3006])#integral of the Sr-90 spikes in scope 2 channel 3

peakbg = (-cal.Peak[cal.Scope == 2][cal.Shot <= 3006])#amplitudes of spikes in background spikes in scope 2 channel 3
peaksr = (-cal.Peak[cal.Scope == 2][cal.Shot >3006])#amplitudes of spikes in Sr-90 spikes in scope 2 channel 3

intbg_2 = (-cal.Integral[cal.Scope == 3][cal.Shot > 3013])#integral of background spikes in scope 3 channel 1
intsr_2 = (-cal.Integral[cal.Scope == 3][cal.Shot <= 3013])#integral of Sr-90 spikes in scope 3 channel 1

peakbg_2 = (-cal.Peak[cal.Scope == 3][cal.Shot > 3013])#amplitudes of background spikes in scope 3 channel 1
peaksr_2 = (-cal.Peak[cal.Scope == 3][cal.Shot <= 3013])#amplitudes of Sr-90 spikes in scope 3 channel 1

def histogram(bg, bin_count, time, figure_num):
    #Function turns data into a normalized histogram
    base, bins = np.histogram(bg, bins = bin_count, density = True)#creates a numpy histogram of given data, attributing the array to the value "base".
    #density = True adjusts for bin width
    hist = base / time #adjusts the histogram for the time the detector was running
    widths = np.diff(bins - 1)#figures out the width of the bins.  -1 is necessary, since np.diff adds 1 to the data.
    
    plt.figure(figure_num)
    plt.step(np.arange(0.0, widths[0] * bin_count, widths[0]), hist)#Makes a step plot that functions as a normalized histogram.
    #x-axis is set to go from 0 to the width of the bins * the number of bins, i.e. the length of the data.  Y-axis is the normalized count per bin.
    plt.loglog() #Plots on a logarithmic scale
    return hist
#Next 19 lines make the plots using the histogram function.  Background and Sr-90 are plotted in the same figure
hist_intbg = histogram(intbg, 200, 563, 1)
hist_intsr = histogram(intsr, 200, 296, 1)
plt.xlabel('Integral with Sr-90')
plt.title('Day 26 Scope 2 Channel 3')

histogram(peakbg, 200, 563, 2)
histogram(peaksr, 200, 296, 2)
plt.xlabel('Amplitude with Sr-90')
plt.title('Day 26 Scope 2 Channel 3')

histogram(intbg_2, 200, 192, 3)
histogram(intsr_2, 200, 178, 3)
plt.xlabel('Integral with Sr-90')
plt.title('Day 26 Scope 3 Channel 1')

histogram(peakbg_2, 200, 192, 4)
histogram(peaksr_2, 200, 178, 4)
plt.xlabel('Amplitude with Sr-90')
plt.title('Day 26 Scope 3 Channel 1')

tf = hist_intbg < hist_intsr
sr_data = hist_intsr[tf] - hist_intbg[tf]
sr_data = np.mean(sr_data)
print "Average Difference in Scope 2: " + str(sr_data)













