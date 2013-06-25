# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 06:10:50 2013

@author: Kyle Weber
"""

def remove_duplicates(x):
    i = 0
    n = []
    while i < len(x):
        if x[i] not in n:
            n == n.append(x[i])
        else:
            n == n
        i += 1
    return n
    print n
    
print remove_duplicates(['new','new','cat',1,True])