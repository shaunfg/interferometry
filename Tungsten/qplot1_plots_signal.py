#!/usr/bin/python

###################################################################################################
### RETURNS ERRORS FOR MERCURY AND TUNGSTEN
### DONE BY PLOTTING HISTOGRAM AND FINDING STD DEV
###################################################################################################

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

# tell it what file to open
fname = "green_tungsten.txt"


f=open(fname,'r')

signal=[]
t_sec=[]
t_usec=[]
position=[]


#read in the data
lines=f.readlines()
for line in range(len(lines)):
    signal.append(lines[line].split()[0])
    t_sec.append(lines[line].split()[1])
    t_usec.append(lines[line].split()[2])
    position.append(lines[line].split()[3])

plt.figure(figsize=[13,8]) 

pl.plot(position,signal,'g-')
pl.xlabel("Position in mm")
pl.ylabel("Signal (a.u)")
plt.figure() 

signal = np.array(signal)
signal = signal.astype(float)

n,bins,patches = plt.hist(signal,bins = 50)
plt.xlabel("Signal Intensity")

plt.ylabel("Number of Entries")

sigma = signal.std()
mean = signal.mean()

print("std dev = ", sigma, ",mean = ", mean)

#plt.plot(np.full((280),mean),np.arange(0,280),'r:')

pl.show()
