# -*- coding: utf-8 -*-
"""
Created on Mon Jul 08 12:34:36 2013

@author: Zach, Brant
"""

import numpy as np
import pandas as pd
import conf
import matplotlib.pyplot as plt

def precondition(amp):
  """Subtracts mean of first fifth of data and flips voltage sign."""
  n = len(amp)
  mean = np.mean(amp[:n/5])
  return -(amp-mean)

def readData(filename,timeDelay=0.0, ampMult=1.0):
    """Reads data from given file (file as produced by oscilloscope)
    returns pandas data frame with Ampl and Time columns (Time in sec, Ampl in volts)"""
    data = []
    with open(filename,'r') as f:
        for x in range(4):
            f.readline()
        data = pd.read_csv(f) 
        data.Ampl = precondition(data.Ampl)*ampMult # convert amplitudes, possibly.
        data.Time = data.Time*1.0e6 - timeDelay # convert to microseconds, offset by delay in signals.
        return data

def findReadData(day,scope,chan,shot):
  """Utility function to load data given shot info.  Uses conf.py to find data, offset timing."""
  return readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scope, chan, scope, shot),
                  conf.timeDelay[scope,chan],
                  conf.ampMult[scope,chan])

def plotScope(day,scope,shot):
  """Plots a given shot as seen on the given scope."""
  axs = []
  for p in range(411,415):
    if p == 411:
      ax = plt.subplot(p)
      plt.title("day %d, shot %d, scope %d"%(day,shot,scope))
    else:
      plt.subplot(p,sharex=ax)

    x = findReadData(day,scope,p-410,shot)
    plt.plot(x.Time,x.Ampl)
    plt.ylabel("channel %d"%(p-410))
    fudgePlotLimits(x.Time,x.Ampl)
  plt.xlabel("time ($\mu$s)")

def plotScopes12p(day,shot):
  """Plots a given shot in eight panels."""
  axs = []
  for p in range(4):
    if p == 0:
      ax = plt.subplot(4,3,3*p+1)
      plt.title("day %d, shot %d, scope 1"%(day,shot))
    else:
      plt.subplot(4,3,3*p+1,sharex=ax)

    x = findReadData(day,1,p+1,shot)
    plt.plot(x.Time,x.Ampl)
    plt.ylabel("channel %d"%(p+1))
    fudgePlotLimits(x.Time,x.Ampl)
    if p==3:
      plt.xlabel("time ($\mu$s)")

    plt.subplot(4,3,3*p+2,sharex=ax)
    if p == 0:
      plt.title("day %d, shot %d, scope 2"%(day,shot))
    x = findReadData(day,2,p+1,shot)
    plt.plot(x.Time,x.Ampl)
    plt.ylabel("channel %d"%(p+1))
    fudgePlotLimits(x.Time,x.Ampl)
    if p==3:
      plt.xlabel("time ($\mu$s)")

    plt.subplot(4,3,3*p+3,sharex=ax)
    if p == 0:
      plt.title("day %d, shot %d, scope 3"%(day,shot))
    x = findReadData(day,3,p+1,shot)
    plt.plot(x.Time,x.Ampl)
    plt.ylabel("channel %d"%(p+1))
    fudgePlotLimits(x.Time,x.Ampl)
    if p==3:
      plt.xlabel("time ($\mu$s)")

def plotScopeHits(day,scope,shot,df):
  """plotScope, but adds hits to it as well, marking afterpulses in green."""
  plotScope(day,scope,shot)
  x = df[np.logical_and(df.day==day,np.logical_and(df.scope==scope,df.shot==shot))]
  for p in range(411,415):
    if p==411:
      ax = plt.subplot(p)
    else:
      plt.subplot(p,sharex=ax)
    xx = x[x.chan==p-410]
    plt.scatter(xx.tStart[np.logical_not(xx.APflag)],
        xx.amp[np.logical_not(xx.APflag)],c='r',s=50)
    plt.scatter(xx.tStart[xx.APflag],
        xx.amp[xx.APflag],c='g',s=50)

def plotScopeHitGroups(day,scope,shot,df):
  """
  plotScope, but adds hit groups (i.e. the ...H data frame from cleanupData),
  shown in boxes.  All hits and all boxes are shown on all plots.
  """

  plotScope(day,scope,shot)
  #x = df.ix[day].ix[shot]
  x = df.ix[shot]
  for p in range(411,415):
    if p==411:
      ax = plt.subplot(p)
    else:
      plt.subplot(p,sharex=ax)
    for id in x.index:
      row = x.ix[id]
      xx = row[[('tMax',det) for det in ['H1','H2','LaBr1','LaBr2','UB1','UB2','UB3','UB4']]]
      yy = row[[('ampMax',det) for det in ['H1','H2','LaBr1','LaBr2','UB1','UB2','UB3','UB4']]]
      plt.scatter(xx,yy)
      x1 = np.nanmin(xx); x2 = np.nanmax(xx)
      y1 = np.nanmin(yy); y2 = np.nanmax(yy)
      plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1])

def fudgePlotLimits(x,y,marfrac=0.1):
  """Expand limits to just beyond the given x and y arrays."""
  xl = np.min(x); yl = np.min(y)
  xh = np.max(x); yh = np.max(y)
  dx = xh-xl; dy = yh-yl
  plt.xlim(xl-dx*marfrac,xh+dx*marfrac)
  plt.ylim(yl-dy*marfrac,yh+dy*marfrac)

def plotsegs(x1,y1,x2,y2):
  """plot segments from x1,y1 to x2,y2."""
  x = np.repeat(np.nan,x1.shape[0]*3)
  y = np.repeat(np.nan,x1.shape[0]*3)
  x[0::3] = x1
  x[1::3] = x2
  y[0::3] = y1
  y[1::3] = y2
  plt.plot(x,y)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688

    # copypasted from http://wiki.scipy.org/Cookbook/SavitzkyGolay
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# coloblind-compatible color map.
cbColMap = [(0.0,0.0,0.0), (0.90,0.60,0.0), (0.35,0.70,0.90), (0.0,0.60,0.50),
    (0.95,0.90,0.25), (0.0,0.45,0.70), (0.80,0.40,0.0), (0.80,0.60,0.70)]

def plotAllShots():
  shotInfo = [(22,150),(23,200),(24,300),(25,300)]
  for day,nshot in shotInfo:
    for shot in xrange(nshot):
      print day,shot
      for scope in [2,3]:
        plt.clf()
        plotScope(day,scope,shot)
        plt.savefig(conf.dataPlotDir+"d%dsc%dsh%03d.png"%(day,scope,shot))
