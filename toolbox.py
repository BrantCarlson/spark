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

def readData(filename):
    data = []
    with open(filename,'r') as f:
        for x in range(4):
            f.readline()
        data = pd.read_csv(f) 
        return data

def findReadData(day,scope,chan,shot):
  return readData(conf.dataDir + "%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scope, chan, scope, shot))

def precondition(amp):
  """Subtracts mean of first fifth of data and flips voltage sign."""
  n = len(amp)
  mean = np.mean(amp[:n/5])
  return -(amp-mean)

def thresh(amp):
  """Returns variance of first fifth of data."""
  return np.sqrt(np.var(amp[:len(amp)/5]))

def findSpikes(df,sigThresh=5,smoothingWindow=10,makePlot=False):
  """Preconditions data, computes threshold, smooths, finds windows, ..."""

  n = df.Ampl.shape[0]
  df.Ampl = precondition(df.Ampl)
  thr = sigThresh*thresh(df.Ampl)/np.sqrt(smoothingWindow)

  # rolling mean:
  #sm = pd.rolling_mean(amp,smoothingWindow)
  # butterworth low-pass:
  b,a = sig.butter(5,2.0/smoothingWindow)
  sm = sig.filtfilt(b,a,df.Ampl)

  d = np.ediff1d(np.where(sm>thr,1,0))
  starts = (d>0).nonzero()[0]
  ends = (d<0).nonzero()[0]

  # ensure starts and ends are logical
  if(starts.shape[0] > 0  and  ends.shape[0] > 0):
    if ends[0]<starts[0]:
      starts = np.concatenate((np.zeros(1,dtype=np.int64),starts))
    if ends[-1]<starts[-1]:
      ends = np.concatenate((ends,np.repeat(np.int64(n-1),1)))

    assert len(starts)==len(ends),"something wrong with interval identification."

    lengths = ends-starts

    #d1 = np.ediff1d(sm)
    #d2 = np.ediff1d(d1)

    def findMax(st,en):
      if(en>st):
        return np.max(sm[st:en])
      else:
        return np.nan
    vfindMax = np.vectorize(findMax)
    maxs = vfindMax(starts,ends)

    dt = df.Time[1]-df.Time[0]
    def integrate(st,en):
      if en>st:
        return np.trapz(sm[st:en],dx=dt)
      else:
        return np.nan
    vintegrate = np.vectorize(integrate)
    integrals = vintegrate(starts,ends)

    # cuts go here, if necessary

    if makePlot:
      plt.plot(df.Time,df.Ampl)
      plt.plot(df.Time,sm)
      #plt.plot(df.Time[1:]-dt/2,d1)
      #plt.plot(df.Time[2:]-dt,d2)
      plt.plot([df.Time.min(),df.Time.max()],[thr,thr])
      segx = np.concatenate([np.array([df.Time[starts[i]],df.Time[ends[i]],None]) for i in xrange(starts.shape[0])])
      segy = np.concatenate([np.array([maxs[i],maxs[i],None]) for i in xrange(starts.shape[0])])
      plt.plot(segx,segy)

    return pd.DataFrame({'sidx':starts,'eidx':ends,
      'start':df.Time[starts].values,'end':df.Time[ends].values,
      'amp':maxs,'len':lengths,'dur':lengths*dt,'int':integrals,
      'sig':maxs/thr*sigThresh},index=range(starts.shape[0]))
  else:
    return pd.DataFrame()

def detStatsForShot(df):
  """process a set of hits for a shot, return a data frame of statistics."""

  sel = np.logical_and(df.start > 0.3e-6, df.start < 1.4e-6)

  df = df.ix[sel]

  if df.amp.count()>0:
    return pd.DataFrame({'nHit':len(df.index),
      'ampMax':df.amp.max(),
      'ampSum':df.amp.sum(),
      'intMax':df.int.max(),
      'intSum':df.int.sum(),
      'tmax':df.amp.argmax()},index=[1])
  else:
    return pd.DataFrame()


def plotScope(day,scope,shot):
  for p in range(411,415):
    plt.subplot(p)
    x = findReadData(day,scope,p-410,shot)
    plt.plot(x.Time*1.0e6,x.Ampl)


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


