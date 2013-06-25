# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:30:59 2013

@author: Kyle Weber
"""

from pandas import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def side_by_side(*objs, **kwds):
    from pandas.core.common import adjoin
    space = kweds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print adjoin(space, *reprs)
    
plt.rc('figure', figsize=(10,6))
#pandas.set_printoptions(notebook_repr_html=False)

labels = ['a','b','c','d','e']
s = Series(randn(5), index=labels)

mapping = s.to_dict()
s = Series(mapping)

df = DataFrame({'Chan1': ray[0][0][0],
                'Chan2': ray[0][0][1],
                'Chan3': ray[0][0][2],
                'Chan4': ray[0][0][3]})
                
