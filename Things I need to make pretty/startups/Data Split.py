"Property of Kyle Weber"

import matplotlib.pyplot as plt

mydata = open("C:\Users\Kyle Weber\Documents\Carthage\Sophomore SURE\Spark Data\C4osc3-00087.txt", "r+")
x=mydata.readlines()
data = x[5:]

def splitit(data):
    i = 0
    xplot = []
    yplot = []
    while i < len(data):
        singlepoint = data[i].split(',')
        xplot == xplot.append(float(singlepoint[0]))
        yplot == yplot.append(float(singlepoint[1]))
        i += 1
    return (xplot,yplot)
(x,y) = splitit(data)


line = plt.plot(x,y,linewidth=0.2)
plt.ylabel("Amplitude")
plt.xlabel("Time")

mydata.close()        
        
    