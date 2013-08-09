# -*- coding: utf-8 -*-
"""
Created on Wed Aug 07 12:35:56 2013

@author: Zach
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def shot_analysis(day, shot_start, shot_end, title, figure_offset, axes_start, axes_end):
    a = pd.load("C:\Users\Owner\Documents\GitHub\spark\spike_info")
    
    d = a.set_index(['Day', 'Scope', 'Channel', 'Shot']) #heirachrically indexing dataframe a
    
    filler = np.zeros(shape = (shot_end,5))
    filler = pd.DataFrame(filler)
    
    #ciu and c2iu are used to make the calibrated energy linregress lines
    ci = d.Peak.ix[22].ix[3].groupby(level =['Shot', 'Channel']).min()
    ciu = ci.unstack(1)
    ciu = ciu.fillna(0)
    ciu = ciu[ciu[1] > -0.46][ciu[2] > -0.44][ciu[3] > -0.43][ciu[4] > -0.043] #Removing saturated data from each channel
    
    ciu = ciu.add(filler, fill_value = 0)
    del ciu[0]
    
    c2i = d.Peak.ix[25].ix[3].groupby(level =['Shot', 'Channel']).min()
    c2iu = c2i.unstack(1)
    c2iu = c2iu.fillna(0)
    c2iu = c2iu[c2iu[1] > -0.46][c2iu[2] > -0.44][c2iu[3] > -0.43][c2iu[4] > -0.043] #Removing saturated data from each channel
    
    c2iu = c2iu.add(filler, fill_value = 0)
    del c2iu[0]
    
    #Creating a dataframe of the useful data from the radial shots
    ri = d.Peak.ix[day].ix[3].groupby(level =['Shot', 'Channel']).min() #a dataframe of the biggest peak in each shot/channel combination on day 23, scope 3
    riu = ri.unstack(1) #Channels are now the columns, shot number is the index
    riu = riu.fillna(0) #Replacing all NaNs with 0
    
    riu = riu[riu[1] > -0.46][riu[2] > -0.44][riu[3] > -0.43][riu[4] > -0.043] #Removing saturated data from each channel
    
    riu = riu.add(filler, fill_value = 0)
    del riu[0]    
    
    statistics = []
    
    count = figure_offset  
    for e in range(1,5): 
        for f in range(1,5):
            if e > f: #f and e both act as variables for a channel.  This line makes sure that there are no redundant plots.
                count += 1
                plt.figure(count) #Creates a separate figure for each plot
                #To plot data from radial shots instead of calibration shots,
                #change ciu to aiu and change 0:50 to 0:100 in the plt.scatter function
                plt.scatter(-riu[f][shot_start:shot_end], -riu[e][shot_start:shot_end]) #Creates a scatter plot of the two channels
                
                #calibrated amplitudes from day 22. Makes green line
                lr1 = stats.linregress(-ciu[f],-ciu[e])
                plt.plot([0,0.5],[lr1[1],(lr1[0]/2)+lr1[1]],'g-')
                
                #calibrated amplitudes from day 25. Makes yellow line
                lr2 = stats.linregress(-c2iu[f].ix[:50],-c2iu[e].ix[:50])
                plt.plot([0,0.5],[lr2[1],(lr2[0]/2)+lr2[1]],'y-')
                
                
                plt.xlabel('Scope 3 Channel ' + str(f) + ' (UB' + str(f) + ') Amplitude')
                plt.ylabel('Scope 3 Channel ' + str(e) + ' (UB' + str(e) + ') Amplitude')
                plt.xlim(axes_start, axes_end)
                if e == 4:
                    plt.ylim(axes_start / 10, axes_end / 10)
                else:
                    plt.ylim(axes_start, axes_end)
                plt.title(title + " UB" + str(f)+ " versus UB" + str(e))            
                
                slope, intercept, r_value, p_value, error= stats.linregress(-riu[f][shot_start:shot_end],-riu[e][shot_start:shot_end]) #Gives calibration statistics for each plot
                plt.plot([0, 0.5], [intercept, (slope / 2) + intercept], "r-") #Plots a calibration line over the scatter plot
                statistics.append(title)                
                statistics.append(day)
                statistics.append(str(shot_start) + ":" + str(shot_end))                
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
    
    hi = d.Peak.ix[day].ix[2].groupby(level =['Shot', 'Channel']).min()
    hiu = hi.unstack(1)
    hiu = hiu.fillna(0)
    hiu = hiu[hiu[3] > -0.7]
    
    hiu = hiu.add(filler, fill_value = 0)
    del hiu[0]    
    
    # For calibration lines for H detectors on the 22nd
    bi = d.Peak.ix[22].ix[2].groupby(level =['Shot', 'Channel']).min()
    biu = bi.unstack(1)
    biu = biu.fillna(0)
    biu = biu[biu[3]>-0.7] #Removing saturated data from each channel
    
    biu = biu.add(filler, fill_value = 0)
    del biu[0]
    
    #For calibration lines for H detectors on the 25th
    b2i = d.Peak.ix[25].ix[2].groupby(level =['Shot', 'Channel']).min()
    b2iu = b2i.unstack(1)
    b2iu = b2iu.fillna(0)
    b2iu = b2iu[b2iu[3]>-0.7] #Removing saturated data from each channel
    
    b2iu = b2iu.add(filler, fill_value = 0)
    del b2iu[0]
    
    plt.figure(count + 1)
    plt.scatter(-hiu[3][shot_start:shot_end], -hiu[4][shot_start:shot_end]) #Creates a scatter plot of the two channels
    
    plt.xlabel('Scope 2 Channel ' + str(3) + ' (H1) Amplitude')
    plt.ylabel('Scope 2 Channel ' + str(4) + ' (H3) Amplitude')
    plt.xlim(axes_start, axes_end)
    plt.ylim(axes_start, axes_end)
    plt.title(title + " H1 versus H3")
    
    slope, intercept, r_value, p_value, error= stats.linregress(-hiu[3][shot_start:shot_end],-hiu[4][shot_start:shot_end]) #Gives calibration statistics for each plot
    plt.plot([0, 0.5], [intercept, (slope /2) + intercept], "r-") #Plots a calibration line over the scatter plot
    
    #calibrated amplitudes from day 22. Makes green line
    lr3 = stats.linregress(-biu[3][shot_start:shot_end],-biu[4][shot_start:shot_end])
    plt.plot([0,0.5],[lr3[1],(lr3[0]/2)+lr3[1]],'g-')
    
    #calibrated amplitudes from day 25. Makes yellow line
    lr4 = stats.linregress(-b2iu[3][shot_start:shot_end],-b2iu[4][shot_start:shot_end])
    plt.plot([0,0.5],[lr4[1],(lr4[0]/2)+lr4[1]],'y-')
    
    
    statistics.append(title)    
    statistics.append(day)
    statistics.append(str(shot_start) + ":" + str(shot_end))    
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
     
    plt.figure(count + 2)
    plt.scatter(-riu[1][shot_start:shot_end], -hiu[3][shot_start:shot_end]) #Creates a scatter plot of the two channels
    plt.xlabel('Scope 3 Channel ' + str(1) + ' (UB1) Amplitude')
    plt.ylabel('Scope 2 Channel ' + str(3) + ' (H1) Amplitude')
    plt.xlim(axes_start, axes_end)
    plt.ylim(axes_start, axes_end)
    plt.title(title + " UB1 versus H1")
    
    slope, intercept, r_value, p_value, error= stats.linregress(-riu[1][shot_start:shot_end],-hiu[3][shot_start:shot_end]) #Gives calibration statistics for each plot
    plt.plot([0, 0.5], [intercept, (slope /2) + intercept], "r-") #Plots a calibration line over the scatter plot
    
    #calibrated amplitudes from day 22. Makes green line
    lr5 = stats.linregress(-ciu[1][shot_start:shot_end],-biu[3][shot_start:shot_end])
    plt.plot([0,0.5],[lr5[1],(lr5[0]/2)+lr5[1]],'g-')
    
    #calibrated amplitudes from day 25. Makes yellow line
    lr6 = stats.linregress(-c2iu[1][shot_start:shot_end],-b2iu[3][shot_start:shot_end])
    plt.plot([0,0.5],[lr6[1],(lr6[0]/2)+lr6[1]],'y-')
    
    
    statistics.append(title)    
    statistics.append(day)
    statistics.append(str(shot_start) + ":" + str(shot_end))    
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
    
    return statistics

radial_data = shot_analysis(23, 0, 100, 'Radial Shots', 0, -0.05, 0.6)
azimuthal_data = shot_analysis(23, 100, 200, 'Azimuthal Shots', 8, -0.05, 0.6)
broad_azimuthal_data = shot_analysis(24, 0, 100, 'Broad Azimuthal Shots', 16, -0.05, 0.6)
lower_broad_azimuthal_data = shot_analysis(24, 100, 200, 'Lower Broad Azimuthal Shots', 24, -0.05, 0.6)
polar_data = shot_analysis(24, 200, 300, 'Polar Shots', 32, -0.05, 0.6)

def dataFrame(r):
    cols = ['Data','Day', 'Shots','X-axis', 'Y-axis', 'Slope', 'Intercept', 'R-value', 'P-value']
    num = len(r) / 9
    a = np.array(r).reshape(num, 9)
    df = pd.DataFrame(data = a, columns = cols)
    return df

radial_results = dataFrame(radial_data)
azimuthal_results = dataFrame(azimuthal_data)
broad_azimuthal_results = dataFrame(broad_azimuthal_data)
lower_broad_azimuthal_results = dataFrame(lower_broad_azimuthal_data)
polar_results = dataFrame(polar_data)

final_results = radial_data
for blah in azimuthal_data:
    final_results.append(blah)
for blah in broad_azimuthal_data:
    final_results.append(blah)
for blah in lower_broad_azimuthal_data:
    final_results.append(blah)
for blah in polar_data:
    final_results.append(blah)    
    
final_results = dataFrame(final_results)
print final_results

