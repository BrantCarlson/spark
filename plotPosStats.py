# -*- coding: utf-8 -*-
"""
Created on August 21, 2013

Plot position dependence of statistics.

@author: Brant
"""

import cleanupData as cd
import calibration as cal
import matplotlib.pyplot as plt
import toolbox as tb
import numpy as np

# NOTE: calibration.py was corrupted in the repository at some point
# this file makes reference to it, so commits since calibration.py were lost
# and may be needed here.
def makePosPlot(d,var,xlab="position",ylim=None,divMean=False):
  c = cal.ub1CrossCal(cd.cal1S,var,['UB1','UB2','UB3','UB4'])
  cal.applyCrossCal(d,c)
  ps = []
  symbols = ['o','*','s','v','p']
  varcol = 'cs_%s'%var

  ## pruning turns out to not be useful either.
  #d = d.ix[d[varcol].mean(1) > 0.001]

  # dividing by the mean turns out to not be useful.
  ydf = d[varcol].copy()
  if divMean:
    ydf = ydf.div(ydf.mean(1),axis=0)

  #plt.yscale('log')
  ps = [plt.scatter(d[('pos',det)],ydf[det],marker='o',s=30)
          for det in c.index]
          #for (det,sym,col) in zip(c.index,symbols,tb.cbColMap[1:])]
  #plt.legend(ps,list(c.index.values))

  for det in c.index:
    plt.text(d[('pos',det)].max(),ydf[det].min(),"\n%s"%det,
        horizontalalignment='center',verticalalignment='top')

  x = d.pos.max()[c.index]
  order = np.argsort(x)
  x = x[order]
  for i in d.index:
    plt.plot(x,ydf.ix[i][c.index][order],c='grey')

  plt.plot(x,ydf.mean(0)[c.index][order],c='black',linewidth=4)
  #plt.plot(x,ydf.median(0)[c.index][order],c='blue',linewidth=4)

  plt.xlabel(xlab)
  plt.ylabel("intensity estimate for %s"%var)
  
  if ylim:
    plt.ylim(ylim)

def makeRadPlot():
  makePosPlot(cd.radS,'intSum',"distance from HV (cm)")
def makeRadPlotZoom():
  makePosPlot(cd.radS,'intSum',"distance from HV (cm)",(-0.001,0.008))

def makePolPlot():
  makePosPlot(cd.polS,'intSum',"angle from HV-gnd axis ($^\circ$)")
def makePolPlotZoom():
  makePosPlot(cd.polS,'intSum',"angle from HV-gnd axis ($^\circ$)",(-0.0005,0.0025))

def make4PanelPosPlots():
  a1 = plt.subplot(221)
  makePolPlot()
  a2 = plt.subplot(222)
  makeRadPlot()
  plt.subplot(223,sharex=a1)
  makePolPlotZoom()
  plt.subplot(224,sharex=a2)
  makeRadPlotZoom()

def saveAllPosPlots():
  """
  Saves all four position variation plots.
  You can resize the resulting plots by opening a new window and resizing it first.
  """
  plt.clf(); makeRadPlot(); plt.savefig('plots/posDepRad.pdf');
  plt.clf(); makeRadPlotZoom(); plt.savefig('plots/posDepRadZoom.pdf');
  plt.clf(); makePolPlot(); plt.savefig('plots/posDepPol.pdf');
  plt.clf(); makePolPlotZoom(); plt.savefig('plots/posDepPolZoom.pdf');
