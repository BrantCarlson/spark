# -*- coding: utf-8 -*-
"""
Created on August 13, 2013

Plot correlations between shots as found by statistics.py.

@author: Brant
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as op
import pandas as pd
import cleanupData as cd
import toolbox as tb
import statistics as st

def makeCorrAzimDepPlot_azim(var):
  """
  Using cleaned azimuthal data frames, find cross correlations and plot as function of detector separation.
  """
  def plotDF(df,color,symbol):
    stats = st.findCrossCorrs(df,var)
    stats = stats.ix[[(x[1] in ['UB1','UB2','UB3','UB4']) and (x[2] in ['UB1','UB2','UB3','UB4']) for x in stats.index]]
    #return plt.scatter(stats.dPos,stats.rPearson,c=color,marker=symbol,edgecolor=color)
    return plt.scatter(stats.dPos,stats.rSpearman,c=color,marker=symbol,edgecolor=color)

  c = plotDF(cd.cal1S,tb.cbColMap[1],'o')
  a1s = plotDF(cd.azim1S,tb.cbColMap[2],'*')
  a2s = plotDF(cd.azim2S,tb.cbColMap[3],'s')
  a3s = plotDF(cd.azim3S,tb.cbColMap[7],'v')
  c2 = plotDF(cd.cal2S,tb.cbColMap[4],'p')

  plt.legend([c,a1s,a2s,a3s,c2],['calibration','$15^\circ$','$37^\circ$','$37^\circ$ lower','recalibration'])
    
  plt.xlabel(r'Separation ($^\circ$)')
  plt.ylabel("Spearman rank correlation for %s"%var)
  plt.xticks([0,30,60,90,120,150,180])
  plt.xlim(-5,185)
  plt.ylim(0,1)

def makeCorrAzimDepPlot_pol(var):
  """
  Using cleaned polar data frames, find cross correlations and plot as function of detector separation.
  """
  def plotDF(df,color,symbol):
    stats = st.findCrossCorrs(df,var)
    stats = stats.ix[[(x[1] in ['UB1','UB2','UB3','UB4']) and (x[2] in ['UB1','UB2','UB3','UB4']) for x in stats.index]]
    #return plt.scatter(stats.dPos,stats.rPearson,c=color,marker=symbol,edgecolor=color)
    return plt.scatter(stats.dPos,stats.rSpearman,c=color,marker=symbol,edgecolor=color)

  c = plotDF(cd.cal1S,tb.cbColMap[1],'o')
  p = plotDF(cd.polS,tb.cbColMap[2],'*')

  plt.legend([c,p],['calibration','polar spacing'])
    
  plt.xlabel(r'Separation ($^\circ$)')
  plt.ylabel("Spearman rank correlation for %s"%var)
  plt.xticks([0,10,20,30,40])
  plt.xlim(-5,40)
  plt.ylim(0,1)

def makeCorrAzimDepPlot_rad(var):
  """
  Using cleaned radial data frames, find cross correlations and plot as function of detector separation.
  """
  def plotDF(df,color,symbol):
    stats = st.findCrossCorrs(df,var)
    stats = stats.ix[[(x[1] in ['UB1','UB2','UB3','UB4']) and (x[2] in ['UB1','UB2','UB3','UB4']) for x in stats.index]]
    #return plt.scatter(stats.dPos,stats.rPearson,c=color,marker=symbol,edgecolor=color)
    return plt.scatter(stats.dPos,stats.rSpearman,c=color,marker=symbol,edgecolor=color)

  c = plotDF(cd.cal1S,tb.cbColMap[1],'o')
  r = plotDF(cd.radS,tb.cbColMap[2],'*')

  plt.legend([c,r],['calibration','radial spacing'])
    
  plt.xlabel(r'Separation (cm)')
  plt.ylabel("Spearman rank correlation for %s"%var)
  plt.ylim(0,1)


