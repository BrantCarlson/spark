# -*- coding: utf-8 -*-
"""
Created on August 12

@author: Brant
"""

import toolbox as tb
import pandas as pd
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def findHits(df,sigThresh=10,smoothingWindow=25,makePlot=False):
  """Finds hits in data.  Preconditions data, computes threshold, smooths data, finds windows, ..."""

  n = df.Ampl.shape[0]
  thr = sigThresh*tb.thresh(df.Ampl)/np.sqrt(smoothingWindow)

  # rolling mean:
  #sm = pd.rolling_mean(amp,smoothingWindow)
  # butterworth low-pass:
  b,a = sig.butter(8,2.0/smoothingWindow)
  sm = sig.filtfilt(b,a,df.Ampl)

  aboveThresh = np.where(sm>thr,1,0)
  d = np.ediff1d(aboveThresh)
  starts = (d>0).nonzero()[0] + 1
  ends = (d<0).nonzero()[0] + 1 # +1 to mark point in data, not point in ediff1d.

  d1tol = 0.0001 # deriv "nonzero" definition
  sgWind = smoothingWindow/2 + ((smoothingWindow/2) % 2) + 1
  d1 = tb.savitzky_golay(df.Ampl.values,sgWind,2,1)
  d1f = np.where(d1>d1tol,1,np.where(d1<-d1tol,-1,0)) # filter 1st derivative
  d1fi = np.arange(d1f.shape[0]) # indices...
  d1fi = d1fi[d1f != 0] # filter indices to remove spots where 1st deriv is "zero"
  d1f = d1f[d1f != 0] # same for 1st deriv
  # this is so I can look for transitions

  d1fdTrans = np.where(np.ediff1d(d1f) == 2)[0] # find transitions
  d1mask = np.zeros(n) # mask for alignment with aboveThresh
  d1mask[(d1fi[d1fdTrans+1]+d1fi[d1fdTrans])/2] = 1 # mark 1st deriv transitions in mask
  d1mask = np.logical_and(d1mask,aboveThresh) 
  # i.e. throw out 1st deriv transitions where signal was below threshold

  d1bds = d1mask.nonzero()[0] # indices in signal where 1st deriv sign change happens.

  # add 1st derivative transitions to starts and ends
  starts = np.concatenate((starts,d1bds+1)); starts.sort()
  ends = np.concatenate((ends,d1bds)); ends.sort()

  if makePlot:
    plt.plot(df.Time,df.Ampl)
    plt.plot(df.Time,sm)
    plt.plot([df.Time.min(),df.Time.max()],[thr,thr])
    plt.plot([df.Time.min(),df.Time.max()],[d1tol,d1tol])
    plt.plot([df.Time.min(),df.Time.max()],[-d1tol,-d1tol])
    plt.plot(df.Time,d1)
    plt.scatter(df.Time[d1bds],np.repeat(0.1,d1bds.shape[0]),c='r')

  # if there are hits...
  if(starts.shape[0] > 0 or ends.shape[0] > 0):
    # ensure starts and ends are logical
    if starts.shape[0] == 0:
      starts = np.zeros(1,dtype=np.int64)
    if ends.shape[0] == 0:
      ends = np.repeat(np.int64(n-1),1)
    if ends[0]<starts[0]:
      starts = np.concatenate((np.zeros(1,dtype=np.int64),starts))
    if ends[-1]<starts[-1]:
      ends = np.concatenate((ends,np.repeat(np.int64(n-1),1)))

    assert len(starts)==len(ends),"something wrong with interval identification."

    lengths = ends-starts

    def findMax(st,en):
      if(en>st):
        return np.max(sm[st:en])
      else:
        return np.nan
    vfindMax = np.vectorize(findMax)
    maxs = vfindMax(starts,ends)

    def findIMax(st,en):
      if(en>st):
        return np.argmax(sm[st:en])+st
      else:
        return 0
    vfindIMax = np.vectorize(findIMax)
    imaxs = vfindIMax(starts,ends)

    dt = df.Time[1]-df.Time[0]
    def integrate(st,en):
      if en>st:
        return np.trapz(sm[st:en],dx=dt)
      else:
        return np.nan
    vintegrate = np.vectorize(integrate)
    integrals = vintegrate(starts,ends)

    def findSat(st,en):
      if(en>st):
        m = np.max(df.Ampl.values[st:en])
        maxRun = np.sum(df.Ampl.values[st:en] == m)
        return maxRun
      return 0
    vfindSat = np.vectorize(findSat)
    satCts = vfindSat(starts,ends)

    # cuts go here, if necessary

    if makePlot:
      tb.plotsegs(df.Time[starts],maxs,df.Time[ends],maxs)
      plt.scatter(df.Time[starts],maxs)
      plt.scatter(df.Time[imaxs],maxs)
      tb.fudgePlotLimits(df.Time,df.Ampl)

    return pd.DataFrame({'iStart':starts,'iEnd':ends,'iMax':imaxs,
      'tStart':df.Time[starts].values,'tEnd':df.Time[ends].values,
      'tMax':df.Time[imaxs].values,
      'amp':maxs,'len':lengths,'dur':lengths*dt,'int':integrals,
      'sig':maxs/thr*sigThresh,'satCt':satCts},index=range(starts.shape[0]))
  else:
    return pd.DataFrame()


# searchSettings[scope,chan] --> (sigThresh,smoothingWindow,detName)
searchSettings = {}
searchSettings[2,1]={'st':10,'wnd':75,'name':'LaBr1'}
searchSettings[2,2]={'st':10,'wnd':75,'name':'LaBr2'}
searchSettings[2,3]={'st':10,'wnd':25,'name':'H1'}
searchSettings[2,4]={'st':10,'wnd':25,'name':'H2'} # significance <15 suspect?

searchSettings[3,1]={'st':10,'wnd':75,'name':'UB1'}  #UB1-3 also ok with 100 for filter window?
searchSettings[3,2]={'st':10,'wnd':75,'name':'UB2'}
searchSettings[3,3]={'st':10,'wnd':75,'name':'UB3'}
searchSettings[3,4]={'st':10,'wnd':200,'name':'UB4'} # note aggressive filter on UB4.

def procChan(day,scope,chan,shot):
  d = findHits(tb.findReadData(day,scope,chan,shot),
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
