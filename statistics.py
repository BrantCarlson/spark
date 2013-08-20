# -*- coding: utf-8 -*-
"""
Created on August 13, 2013

Code to calculate statistics on hits and shots as found by cleanupData.

@author: Brant
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as op
import pandas as pd

def fitWithZeroIntercept(x,y):
  """
  Fits y = m*x to the given x,y data.
  """
  def fit(x,m):
    return x*m
  def err(p):
    return fit(x,p[0])-y
  p1,success = op.leastsq(err,1.0)
  return p1

def findCorr(df,var,det1,det2,makePlot=True,title='correlation plot'):
  """
  Takes a data frame of correlation info (the cleanupData.(...)S or (...)H data frames),
  a variable (e.g. 'ampMax'), and two detectors (e.g. 'UB1' and 'UB2') and finds
  regression lines and correlation coefficients,  optionally plotting.
  """
  x = df[(var,det1)]
  y = df[(var,det2)]

  xl = np.min(x); xu = np.max(x)
  yl = np.min(y); yu = np.max(y)
  margin=0.05
  xlow = xl-(xu-xl)*margin
  xhigh = xu+(xu-xl)*margin
  ylow = yl-(yu-yl)*margin
  yhigh = yu+(yu-yl)*margin

  slope,intercept,rp,pp,err = stats.linregress(x,y)
  xx = np.array([xlow,xhigh])

  rs,ps = stats.spearmanr(x,y)
  
  slope2 = fitWithZeroIntercept(x,y)
  
  p1 = df[('pos',det1)].max()
  p2 = df[('pos',det2)].max()

  if makePlot:
    plt.scatter(x,y)

    plt.xlim(xlow,xhigh)
    plt.ylim(ylow,yhigh)

    plt.plot(xx,slope*xx+intercept)
    plt.plot(xx,slope2*xx)

    plt.hlines(0,xlow,xhigh)
    plt.vlines(0,ylow,yhigh)

    plt.xlabel(det1)
    plt.ylabel(det2)

    plt.text(np.min(x)+xhigh*margin,np.max(y),
        plt.dedent("""
        y=mx+b: m=%.4f, b=%.4f
        y=mx: m=%.4f
        Pearson r=%.3f (p=%.2g)
        Spearman r=%.3f (p=%.2g)"""
        %(slope,intercept,slope2,rp,pp,rs,ps)), verticalalignment='top')

    plt.title("%s for %s, shots %d to %d"%(title, var,
        reduce(lambda x,y: min(x,y[0]), df.index, 1000),
        reduce(lambda x,y: max(x,y[0]), df.index, 0)))

  return pd.DataFrame({
    'slp':slope, 'intcpt':intercept, 'slp0':slope2,
    'rPearson':rp, 'pPearson':pp,
    'rSpearman':rs, 'pSpearman':ps,
    'var':var, 'det1':det1, 'det2':det2,
    'dPos':np.abs(p2-p1)},
    index=[0])

def findCrossCorrs(df,var,makePlots=False):
  """
  Finds cross correlation statistics (findCorr) for all detector pairs in data frame.
  """
  vars = np.unique([x[0] for x in df.columns])
  dets = np.unique([x[1] for x in df.columns])
  dfs = [] # accumulator for data frames

  #for d1,d2 in [(det1,det2) for det1 in dets for det2 in dets if det1<det2]:
  for d1,d2 in [(det1,det2) for det1 in dets for det2 in dets if det1!=det2]:
    dfs.append(findCorr(df,var,d1,d2,makePlot=makePlots))

  return pd.concat(dfs).set_index(['var','det1','det2'])

# df = df.ix[[(a[2] in ['UB1','UB2','UB3','UB4']) and (a[1] in ['UB1','UB2','UB3','UB4']) for a in df.index]].ix['intSum']
