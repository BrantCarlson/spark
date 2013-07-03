import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import conf

day = 22
scopeNo = 2
chan = 3
shot = 10

def readData(filename):
    data = []
    with open(filename,'r') as f:
        for x in range(4):
            f.readline()
        data = pd.read_csv(f)
        return data

y = readData(conf.dataDir + "/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot))

def threshold(y,sig=2,smoothPts=3):
    """Find regions where the amplitude variable is above threshold away from the mean.
       The threshold is defined in terms of the standard deviation (i.e. width) of the noise
       by the significance (sig).  I.e. a spike is significant if the rolling mean of the data
       (taken with the window smoothPts) is above (standard deviation)*sig/sqrt(smoothPts)."""
    m = y.Ampl[:len(y)/4].mean()
    s = np.sqrt(y.Ampl[:len(y)/4].var())
    return pd.rolling_mean(y.Ampl-m,smoothPts) < -s*sig/np.sqrt(smoothPts)


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
    while index_count < len(z) - 1:
        for i in z[index_count:]:
            index_count += 1
            if i == True and z[index_count+1] == True:
                start = x[index_count]
                s_index = index_count
                break
        for i in z[index_count:]:
            index_count += 1
            if z[index_count] == False and z[index_count-1] == True:
                end = x[index_count-1]
                e_index = index_count
                break
            elif index_count >= len(z)-2:
                end = x[index_count-1]
                e_index = index_count
                break
        duration = end - start
        peak = y.Ampl[s_index:e_index].min()
        #Throws out any false-positives from noise
        if duration > 2.5e-9 and start != 0:
            results.append(start)
            results.append(end)
            results.append(peak)
            results.append(duration)
            results.append(day)
            results.append(shot)
            results.append(scopeNo)
            results.append(chan)
            print "Spike Duration: " + str(duration) + " seconds."
            print "Peak: " + str(peak)
            print "Start: " + str(start) + " seconds.", "End: " + str(end) + " seconds."
        #Resets start and end so as to not report the last spike twice
        start = 0
        end = 0
    return results

def dataFrame(r):
    cols = ['Start', 'End', 'Peak', 'Duration', 'Day', 'Shot', 'Scope', 'Channel']
    num = len(r) / 8
    a = np.array(r).reshape(num, 8)
    df = pd.DataFrame(data = a, columns = cols)
    return df



n_smooth = 25
significance = 12
spikey = threshold(y,significance,n_smooth)
results = time_intervals(y.Time,spikey)

print dataFrame(results)

plt.plot(y.Time,y.Ampl)
plt.plot(pd.rolling_mean(y.Time,n_smooth),pd.rolling_mean(y.Ampl,n_smooth))
plt.plot(y.Time,spikey/50.0,'r-')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Scope " + str(scopeNo) +", Channel " +str(chan) + ", Shot " + str(shot) + " on Jan " + str(day))

plt.show()

