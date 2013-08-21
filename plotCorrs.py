# -*- coding: utf-8 -*-
"""
Created on August 13, 2013

Plot correlations between shots as found by statistics.py.

@author: Brant
"""
import matplotlib.pyplot as plt
import cleanupData as cd
import toolbox as tb
import statistics as st

def makeCorrPlot(var,dfs,captions,xlab="separation",ticks=None,xlim=(-5,185),ylim=(0,1),allowedDets=['UB1','UB2','UB3','UB4','H1']):
  def plotDF(df,color,symbol):
    stats = st.findCrossCorrs(df,var)
    stats = stats.ix[[(x[1] in allowedDets) and (x[2] in allowedDets) for x in stats.index]]
    #return plt.scatter(stats.dPos,stats.rPearson,c=color,marker=symbol,edgecolor=color)
    return plt.scatter(stats.dPos,stats.rSpearman,c=color,marker=symbol,edgecolor=color)
    #return plt.scatter(stats.dPos,stats.rKendall,c=color,marker=symbol,edgecolor=color)

  colors = tb.cbColMap[1:] #skip black
  symbols = ['o','*','s','v','p']
  plts = [plotDF(df,col,smb) for (df,col,smb) in zip(dfs,colors,symbols)]

  plt.legend(plts,captions)
    
  plt.xlabel(xlab)
  plt.ylabel("Spearman rank correlation for %s"%var)
  if ticks:
    plt.xticks(ticks)
  if xlim:
    plt.xlim(xlim)
  plt.ylim(ylim)

def makeCorrDepPlot_azim(var):
  """
  Using cleaned azimuthal data frames, find cross correlations and plot as function of detector separation.
  """
  return makeCorrPlot(var,[cd.cal1S,cd.azim1S,cd.azim2S,cd.azim3S,cd.cal2S],
      ["calibration",'$15^\circ$ separation','$40^\circ$ separation','$40^\circ$ separation, lower','recalibration'],
      xlab=r'separation ($^\circ$)',
      ticks=[0,30,60,90,120,150,180],
      xlim=(-5,185),
      ylim=(0,1))

def makeCorrDepPlot_pol(var):
  """
  Using cleaned polar data frames, find cross correlations and plot as function of detector separation.
  """
  return makeCorrPlot(var,[cd.cal1S,cd.polS,cd.cal2S],
      ["calibration",'polar separation','recalibration'],
      xlab=r'separation ($^\circ$)',
      ticks=[0,10,20,30,40],
      xlim=(-5,40),
      ylim=(0,1))

def makeCorrDepPlot_rad(var):
  """
  Using cleaned radial data frames, find cross correlations and plot as function of detector separation.
  """
  return makeCorrPlot(var,[cd.cal1S,cd.radS,cd.cal2S],
      ["calibration",'radial separation','recalibration'],
      xlab=r'separation (cm)',
      ticks=None,
      xlim=None,
      ylim=(0,1))

def saveAllCorrPlots():
  """
  Saves all three correlation variation plots.
  You can resize the resulting plots by opening a new window and resizing it first.
  """
  plt.clf(); pcrs.makeCorrDepPlot_pol('intSum'); plt.savefig('plots/corrPol.pdf');
  plt.clf(); pcrs.makeCorrDepPlot_azim('intSum'); plt.savefig('plots/corrAzim.pdf');
  plt.clf(); pcrs.makeCorrDepPlot_rad('intSum'); plt.savefig('plots/corrRad.pdf');
