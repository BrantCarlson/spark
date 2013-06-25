# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:50:39 2013

@author: Kyle Weber
"""

mydata = open("C:\Users\Kyle Weber\Documents\Carthage\Sophomore SURE\Spark Data\C4osc3-00087.txt", "r+")
fixed_data = (str(mydata.read())).split()

singlepoint = str(fixed_data[6].split())
"print singlepoint"

def splitit(singlepoint):
    i = 2
    sing = list(singlepoint)
    xplot = ""
    while i < len(singlepoint)-2:
        if sing[i] != list("'") and sing[i] != list(","):
            xplot += sing[i]
            i += 1
        return xplot
print splitit(singlepoint)
        
            
        