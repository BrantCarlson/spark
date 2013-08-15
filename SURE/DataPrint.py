# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:10:21 2013

@author: Kyle Weber
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def readData(day,shot,scopeNo,chan):

    with open("J:/SURE/sparkData_2013/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r") as f:    
        for i in range(4):        
            f.readline() 
            # read (and discard) the first 4 lines            
        x = pd.read_csv(f) 
            # then use Pandas' built-in comma-separated-variable function            
            # using the with ... construct, f is automatically closed.
    return x


def printData(day):
    for shot in range(0,150):
        for scopeNo in range(1,4):
            for chan in range(1,5):
                x = readData(day,shot,scopeNo,chan)
                plt.figure(day,shot,scopeNo)
                figure(chan).canvas.set_window_title('Jan %d, Shot %05d, Oscilloscope ' % day, shot, scopeNo)
                plt.subplot(410 + chan)
                plt.plot(x)
                plt.ylabel("Amplitude")
                plt.xlabel("Time (microseconds)")
                
            