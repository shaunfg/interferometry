# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:03:28 2019

@author: cnh17
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert,chirp

# Define some test data which is close to Gaussian
fname = 'white_light.txt'

f = open(fname,'r')

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

#data = random.sample(x,5)

x = np.array(position)
y = np.array(signal)

duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs

signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

plt.plot(x,y)