# -*- coding: utf-8 -*-
"""
Created on August 12

@author: Brant
"""

import toolbox as tb
import pandas as pd
import numpy as np
from IPython.parallel import Client

# searchSettings[scope,chan] --> (sigThresh,smoothingWindow,detName)
searchSettings = {}
searchSettings[2,1]={'st':10,'wnd':75,'name':'LaBr1'}
searchSettings[2,2]={'st':10,'wnd':75,'name':'LaBr2'}
searchSettings[2,3]={'st':10,'wnd':25,'name':'H1'}
searchSettings[2,4]={'st':10,'wnd':25,'name':'H2'} # significance <15 suspect?

searchSettings[3,1]={'st':10,'wnd':100,'name':'UB1'}
searchSettings[3,2]={'st':10,'wnd':100,'name':'UB2'}
searchSettings[3,3]={'st':10,'wnd':100,'name':'UB3'}
searchSettings[3,4]={'st':10,'wnd':100,'name':'UB4'}

def procChan(day,scope,chan,shot):
  d = tb.findSpikes(tb.findReadData(day,scope,chan,shot),
      searchSettings[scope,chan]['st'],
      searchSettings[scope,chan]['wnd'])
  d['day'] = day
  d['scope'] = scope
  d['chan'] = chan
  d['shot'] = shot
  d['det'] = searchSettings[scope,chan]['name']
  #d.set_index(['day','shot','det'],inplace=True)
  #d.sort_index(inplace=True)
  return d

def procShot(day,shot):
  print day,shot
  return pd.concat([procChan(day,scope,chan,shot) for scope in [2,3] for chan in [1,2,3,4]])

def procDay(day,nshots):
  return pd.concat([procShot(day,i) for i in xrange(nshots)])

def readAndProcessAllShots():
  shotInfo = [(22,150),(23,200),(24,300),(25,300)]
  df = pd.concat([procDay(day,nshots) for (day,nshots) in shotInfo])
  return df

badShots = [(22,105), #cabinet was open.
    ]
