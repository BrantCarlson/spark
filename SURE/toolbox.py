# -*- coding: utf-8 -*-
"""
Created on Mon Jul 08 12:34:36 2013

@author: Zach, Brant
"""

import numpy as np
import pandas as pd
import scipy as si
import matplotlib as plt
import scipy.signal as sig
import conf
import matplotlib.pyplot as plt

def precondition(amp):
  """Subtracts mean of first fifth of data and flips voltage sign."""
  n = len(amp)
  mean = np.mean(amp[:n/5])
  return -(amp-mean)

def readData(filename):
    """Reads data from given file (file as produced by oscilloscope)
    returns pandas data frame with Ampl and Time columns (Time in sec, Ampl in volts)"""
    data = []
    with open(filename,'r') as f:
        for x in range(4):
            f.readline()
        data = pd.read_csv(f) 
        data.Ampl = precondition(data.Ampl)
        return data

def findReadData(day,scope,chan,shot):
  """Utility function to load data given shot info.  Uses conf.py to find data."""
  return readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scope, chan, scope, shot))

def thresh(amp):
  """Returns variance of first fifth of data."""
  return np.sqrt(np.var(amp[:len(amp)/5]))

def findCorrHits_BS(df):
  assert False, "crap.  this function doesn't work."

  # pre-filter
  df = df[df.sig > 15]

  dets = df.det.unique()
  tms = apply(np.meshgrid, [df.ix[df.det == det].start.values for det in dets])
  tms = [tm.flatten() for tm in tms]
  tm = np.matrix(tms)
  print 'tm:',tm.shape

  diag = np.repeat(1.0/np.sqrt(tm.shape[0]),tm.shape[0]*tm.shape[1])
  diag = diag.reshape(tm.shape)

  print 'diag:',diag.shape

  dcomp = np.multiply(tm,diag).sum(0)
  dcomp = np.multiply(diag,np.repeat(dcomp,7,axis=0))

  dtsq = np.sum(np.power(tm-dcomp,2),axis=0)

  dtTol = 0.05e-6 **2 #this is probably too big

  return tm,dtsq<dtTol

def findCorrHits_BS2(df):

  assert False, "this version doesn't work either."

  print df.day.values[0],df.scope.values[0],df.chan.values[0],df.shot.values[0]
  df = df[df.sig > 15]

  ii = np.arange(len(df.index))
  df = df.set_index([ii]) # reset index so I can re-order later.

  dets = df.det.unique()
  dets.sort()
  df['detID'] = np.searchsorted(dets,df.det)

  ts = df.tMax.values
  ds = df.detID.values
  
  t1,t2 = np.meshgrid(ts,ts)
  d1,d2 = np.meshgrid(ds,ds)
  i1,i2 = np.meshgrid(ii,ii)
  t1 = t1.flatten()
  t2 = t2.flatten()
  d1 = d1.flatten()
  d2 = d2.flatten()
  i1 = i1.flatten()
  i2 = i2.flatten()

  # remove detector matches with itself
  mask = np.logical_and(t1 != t2,d1 != d2)
  t1 = t1[mask]; t2 = t2[mask]
  d1 = d1[mask]; d2 = d2[mask]
  i1 = i1[mask]; i2 = i2[mask]

  # remove everything that isn't a close pair
  dtTol = 0.03e-6
  pairMask = np.abs(t2-t1) < dtTol

  #t1 = t1[pairMask]; t2 = t2[pairMask]
  #d1 = d1[pairMask]; d2 = d2[pairMask]
  i1 = i1[pairMask]; i2 = i2[pairMask]

  pairID = np.arange(i1.shape[0])

  df1 = df.ix[i1]; df1.rename(columns=lambda x:"%s1"%x,inplace=True)
  df2 = df.ix[i2]; df2.rename(columns=lambda x:"%s2"%x,inplace=True)
  df1['pairID'] = pairID
  df2['pairID'] = pairID

  return pd.merge(df1,df2,on='pairID')

def plotScope(day,scope,shot):
  for p in range(411,415):
    plt.subplot(p)
    x = findReadData(day,scope,p-410,shot)
    plt.plot(x.Time*1.0e6,x.Ampl)
    plt.ylabel("channel %d"%(p-410))
    fudgePlotLimits(x.Time*1.0e6,x.Ampl)
  plt.xlabel("time ($\mu$s)")

def plotScopeHits(day,scope,shot,df):
  plotScope(day,scope,shot)
  x = df[np.logical_and(df.day==day,np.logical_and(df.scope==scope,df.shot==shot))]
  for p in range(411,415):
    plt.subplot(p)
    xx = x[x.chan==p-410]
    plt.scatter(xx.tStart[np.logical_not(xx.APflag)]*1.0e6,
        xx.amp[np.logical_not(xx.APflag)],c='r',s=50)
    plt.scatter(xx.tStart[xx.APflag]*1.0e6,
        xx.amp[xx.APflag],c='g',s=50)
    print xx.APflag

def fudgePlotLimits(x,y,marfrac=0.1):
  xl = np.min(x); yl = np.min(y)
  xh = np.max(x); yh = np.max(y)
  dx = xh-xl; dy = yh-yl
  plt.xlim(xl-dx*marfrac,xh+dx*marfrac)
  plt.ylim(yl-dy*marfrac,yh+dy*marfrac)

def plotsegs(x1,y1,x2,y2):
  x = np.repeat(np.nan,x1.shape[0]*3)
  y = np.repeat(np.nan,x1.shape[0]*3)
  x[0::3] = x1
  x[1::3] = x2
  y[0::3] = y1
  y[1::3] = y2
  plt.plot(x,y)

#Kevin's time_intervals function
def time_intervals_Kevin(x,z):
    #takes two lists as arguments
    #Defining variables
    results = []
    start = 0.0
    end = 0.0
    duration = 0.0
    index_count = 0
    s_index = 0
    e_index = 0
    peak = 0.0
    #Loops through indices and looks for beginnings and ends of spikes
    while index_count < len(z) - 2:
        for i in z[index_count:]:
            index_count += 1
            if i == True and z[index_count+1] == True:
                start = x[index_count]
                s_index = index_count
                break
        for i in z[index_count:]:
            index_count += 1
            if (z[index_count] == False and z[index_count-1] == True) or index_count >= len(z)-2:
                end = x[index_count-1]
                e_index = index_count
                break
        duration = end - start
        peak = y.Ampl[s_index:e_index].min()
        #Throws out any false-positives from noise
        
        integral_a = si.trapz(y[s_index:e_index])
        integral_b = np.sum(integral_a)
        
        if duration > 2.5e-9 and start != 0:
            results.append(start)
            results.append(end)
            results.append(peak)
            results.append(duration)
            results.append(day)
            results.append(shot)
            results.append(scopeNo)
            results.append(chan)
            results.append(integral_b)
        #Resets start and end so as to not report the last spike twice
        start = 0
        end = 0
        return results

#n_smooth = 25
#significance = 12

#For turning a list of values into a dataframe
def dataFrame(r):
    cols = ['Start', 'End', 'Peak', 'Duration', 'Day', 'Shot', 'Scope', 'Channel', 'Integral']
    num = len(r) / 9
    a = np.array(r).reshape(num, 9)
    df = pd.DataFrame(data = a, columns = cols)
    return df
    
#Function from Kyle's Final Spikefinder
def find(j):
    thresh = pd.Series.mean(j.Ampl) - pd.Series.std(j.Ampl)
    rj = j.iloc[20:len(j)-20]  
    spike = pd.rolling_mean(rj,40)
    count = 0
    
    while count < 100:
        
        peak_amp = pd.DataFrame.min(j)[1]   
        peak_time_i = pd.DataFrame.idxmin(j)[1]
        peak_time = j['Time'][peak_time_i]
        
        if peak_amp < thresh:
            plt.plot(j)
            for i in range(peak_time_i-150,58,-1):
                front_der = (spike['Ampl'][i] - spike['Ampl'][i - 1]) / (spike['Time'][i] - spike['Time'][i - 1])
                a = 0        
                if front_der >= 0:
                    start = spike['Time'][i]
                    print "\nStarts at " + str(start)
                    a = i
                    break
            for i in range(peak_time_i + 150,len(j),1):
                end = 0        
                back_der = ( (spike['Ampl'][i + 1] - spike['Ampl'][i]) / (spike['Time'][i + 1] - spike['Time'][i]) )
                b = 0
                if back_der <= 0:
                    end = spike['Time'][i]
                    print "Ends at " + str(end)
                    b = i
                    break  
            print "Spike duration of " + str(end - start) + " seconds"
            print "Spike peak at t=" + str(peak_time) + " and amplitude " + str(peak_amp) + " millivolts"     
            j['Ampl'][a:b] = 0
            count += 1
        else: count = 100

#Kyle's readData
def readData_Kyle(day,shot,scopeNo,chan):

    with open(conf.kdataDir + "/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r") as f:    
        for i in range(4):        
            f.readline() 
            # read (and discard) the first 4 lines            
        x = pd.read_csv(f) 
            # then use Pandas' built-in comma-separated-variable function            
            # using the with ... construct, f is automatically closed.
    return x

"""
for day in range(22,27):
    if day == 22:
        for shot in range(0,150):
            for scopeNo in range(2,4):
                for chan in range(1,5):
                    asdf = 0
    elif day == 23:
        for shot in range(0,200):
            for scopeNo in range(2,4):
                for chan in range(1,5):
                    asdf = 0
    elif day == 24 or day == 25:
        for shot in range(0,300):
            for scopeNo in range(2,4):
                for chan in range(1,5):
                    asdf = 0
    elif day == 26:
        for shot in range(0,6022):
            for scopeNo in range(2,4):
                if scopeNo == 2:
                    chan = 3
                    
                elif scopeNo == 3:
                    chan = 1
"""

#Zack's version of time_intervals
def time_intervals(x,z):
    #takes two lists as arguments
    #Defining variables
    results = []
    start = 0.0
    end = 0.0
    duration = 0.0
    index_count = 0
    s_index = 0
    e_index = 0
    peak = 0.0
    #Loops through indices and looks for beginnings and ends of spikes
    while index_count < len(z) - 2:
        for i in z[index_count:]:
            index_count += 1
            if i == True and z[index_count-1] == True:
                start = x[index_count-1]
                s_index = index_count
                break
        for i in z[index_count:]:
            index_count += 1
            if z[index_count-1] == False and z[index_count-2] == True:
                end = x[index_count-1]
                e_index = index_count
                break
            elif index_count >= len(z) - 5:
                #Breaks if no end has been found for a spike by the end of z
                end = x[index_count-1]
                e_index = index_count
                break
        duration = end - start
        peak = y.Ampl[s_index:e_index].min()
        integral_a = si.trapz(y[s_index:e_index])
        integral_b = np.sum(integral_a)
        if duration > 3.0e-9 and start != 0:
            #Checks the duration of the spike to throw out any false positives
            #that would come from noise.  If the spike is not a false positive
            #the data is appended to results and printed
            results.append(start)
            results.append(end)
            results.append(peak)
            results.append(duration)
            results.append(day)
            results.append(shot)
            results.append(scopeNo)
            results.append(chan)
            results.append(integral_b)
            print "Spike Duration: " + str(duration) + " seconds."
            print "Peak: " + str(peak)
            print "Start: " + str(start) + " seconds.", "End: " + str(end) + " seconds."
            print str(day) , str(shot) , str(scopeNo) , str(chan)
            #Resets start and end so as to not report the last spike twice
            start = 0
            end = 0
    return results


def list_to_frame(r):
    #Converts a list of lists into a data frame
    h = []
    for e in r:
        for q in e:
            h.append(q)
    a = []
    row_len = 0
    cols = []
    cols = ['Start', 'End', 'Peak', 'Duration', 'Day', 'Shot', 'Scope', 'Channel', 'Integral']
    row_len = len(h) / 9
    a = np.array(h).reshape(row_len, 9)
    df = pd.DataFrame(data = a, columns = cols)
    return df


#Kevin's threshold
def threshold_Kevin(y,sig=2,smoothPts=3):
    """Find regions where the amplitude variable is above threshold away from the mean.
       The threshold is defined in terms of the standard deviation (i.e. width) of the noise
       by the significance (sig).  I.e. a spike is significant if the rolling mean of the data
       (taken with the window smoothPts) is above (standard deviation)*sig/sqrt(smoothPts)."""
    m = y.Ampl[:len(y)/4].mean()
    s = np.sqrt(y.Ampl[:len(y)/4].var())
    return pd.rolling_mean(y.Ampl-m,smoothPts) < -s*sig/np.sqrt(smoothPts)
    

def threshold(y, sig, smoothPts):
    smoothed = pd.rolling_mean(y.Ampl, smoothPts)
    deviation = y.Ampl.std()
    return 0.05*(smoothed < (-deviation * sig))

# copypasted from http://wiki.scipy.org/Cookbook/SavitzkyGolay
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
