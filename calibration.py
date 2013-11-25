# -*- coding: utf-8 -*-
"""
Created on August 13, 2013

Calibration functions

@author: Brant
"""

import statistics as st
import cleanupData as cd
import numpy as np
import pandas as pd

# well-behaved fiber detectors.
goodDets = ['UB1','UB2','UB3','UB4','H1']
#goodDets = ['UB1','UB2','UB3','UB4']

def meanCrossCal(df,var,allowedDets=goodDets):
  """
  Cross correlate each detector to the mean of all detectors.
  Takes a ...H or ...S data frame, returns a data frame with calibration fit
  results data, indexed by the detector name.
  """
  df = df[var] # subset now only has all detector data for this variable.
  dets = [x for x in np.unique(df.columns.values) if x in allowedDets]
  mcol = 'm%s'%var
  df[mcol] = df.mean(axis=1) # mean of rows

  calData = pd.concat([st.findCorr(df,None,det,mcol,False) for det in dets])
  calData = calData.set_index('det1')
  calData.calVar = var # note: not Pandas, just using the object to store extra info.
  return calData

def detCrossCal(df,var,baseDet='UB1',allowedDets=goodDets):
  """
  Cross correlate each detector to the signal from UB1.
  Takes a ...H or ...S data frame, returns a data frame with calibration fit
  results data, indexed by the detector name.
  """
  df = df[var] # subset now only has all detector data for this variable.
  dets = [x for x in np.unique(df.columns.values) if x in allowedDets]

  calData = pd.concat([st.findCorr(df,None,det,baseDet,False) for det in dets])
  calData = calData.set_index('det1')
  calData.calVar = var # note: not Pandas, just using the object to store extra info.
  return calData

def ub1CrossCal(df,var,allowedDets=goodDets):
  return detCrossCal(df,var,baseDet='UB1',allowedDets=allowedDets)

def applyCrossCal(df,caldf):
  """
  Applies a cross correlation to the mean as calculated by meanCrossCal.
  Adds columns to the data frame corresponding to the mean of the calibrated variable,
  as estimated by the signals from each detector.
  """
  var = caldf.calVar
  cmname = 'cal_%s'%var
  for det in caldf.index:
    df[('cs_%s'%var,det)] = df[(var,det)]*caldf['slp0'].ix[det]
    df[('csi_%s'%var,det)] = df[(var,det)]*caldf['slp'].ix[det] + caldf['intcpt'].ix[det]
    # notation here: cs_... calibation slope only, 
    # csi_... calibration wtih slope and intercept.

def beforeAfterComparison():
  """
  Compares cross calibration to the mean for cal1S and cal2S.
  Mostly just a place to store the code that does this.  Take a look and edit
  as needed to examine whatever you want.
  """
  # filter out allowed detectors?
  c1 = cd.cal1S[[x for x in cd.cal1S.columns if x[1] in goodDets]]
  c2 = cd.cal2S[[x for x in cd.cal2S.columns if x[1] in goodDets]]

  x1 = ub1CrossCal(c1,'intSum')
  x2 = ub1CrossCal(c2,'intSum')
  x12 = pd.merge(x1,x2,how='outer',left_index=True,right_index=True)
  x12[['rSpearman_x','rSpearman_y']].plot()
  return x12

