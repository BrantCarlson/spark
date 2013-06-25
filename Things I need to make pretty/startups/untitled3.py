# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:33:45 2013

@author: Kyle Weber
"""

mydata = open("C:\Users\Kyle Weber\Documents\Carthage\Sophomore SURE\Spark Data\C4osc3-00087.txt", "r+")
x=mydata.readlines()
data = x[5:]

def splitit(data):
    i = 0
    xplot = []
    yplot = []
    while i < len(data):
        singlepoint = data[i].split(',')
        xplot == xplot.append(float(singlepoint[0]))
        yplot == yplot.append(float(singlepoint[1]))
        i += 1
    return xplot and yplot
print splitit(data)
    
    
        
        
    