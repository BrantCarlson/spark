# -*- coding: utf-8 -*-
"""
Created on August 13, 2013

@author: Brant
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as op

def fitWithZeroIntercept(x,y):
  def fit(x,m):
    return x*m
  def err(p):
    return fit(x,p[0])-y
  p1,success = op.leastsq(err,1.0)
  return p1

def plotCorr(df,var,det1,det2,title='correlation plot'):
  x = df[(var,det1)]
  y = df[(var,det2)]

  plt.scatter(x,y)

  xl = np.min(x); xu = np.max(x)
  yl = np.min(y); yu = np.max(y)
  margin=0.05
  xlow = xl-(xu-xl)*margin
  xhigh = xu+(xu-xl)*margin
  ylow = yl-(yu-yl)*margin
  yhigh = yu+(yu-yl)*margin

  plt.xlim(xlow,xhigh)
  plt.ylim(ylow,yhigh)

  slope,intercept,rp,pp,err = stats.linregress(x,y)
  xx = np.array([xlow,xhigh])
  plt.plot(xx,slope*xx+intercept)

  rs,ps = stats.spearmanr(x,y)
  
  slope2 = fitWithZeroIntercept(x,y)
  plt.plot(xx,slope2*xx)

  plt.hlines(0,xlow,xhigh)
  plt.vlines(0,ylow,yhigh)

  plt.xlabel(det1)
  plt.ylabel(det2)

  plt.text(np.min(x)+xhigh*margin,np.max(y),
      """y=mx+b: m=%.4f, b=%.4f
y=mx: m=%.4f
Pearson r=%.3f (p=%.2g)
Spearman r=%.3f (p=%.2g)"""
      %(slope,intercept,slope2,rp,pp,rs,ps), verticalalignment='top')

  print var
  plt.title("%s for %s, shots %d to %d"%(title, var,
      reduce(lambda x,y: min(x,y[0]), df.index, 1000),
      reduce(lambda x,y: max(x,y[0]), df.index, 0)))
