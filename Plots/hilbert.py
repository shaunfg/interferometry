#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 22:26:10 2019

@author: ShaunGan
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import hilbert, chirp
from scipy.optimize import curve_fit

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

x = x.astype(np.float)
y = y.astype(np.float)

def getEnvelope (inputSignal):

    # Taking the absolute value

    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append (abs (sample))
    print(len(absoluteSignal))
    # Peak detection

    intervalLength = 120 # Experiment with this number, it depends on your sample frequency and highest "whistle" frequency
    outputSignal = []
    
    for baseIndex in range (intervalLength, len (absoluteSignal)):
        maximum = 0
        for lookbackIndex in range (intervalLength):
            maximum = max (absoluteSignal [baseIndex - lookbackIndex], maximum)
        outputSignal.append (maximum)
    print(len(outputSignal))
    return outputSignal

envelope = getEnvelope(y)
plt.figure(figsize = [8,5])

plt.plot(position, signal)
plt.plot(position[120:], envelope)

fit, cov = curve_fit(getEnvelope,position,signal)

print(fit)


#%%

def analytic_signal(x):
    from scipy.fftpack import fft,ifft
    N = len(x)
    X = fft(x,N)
    h = np.zeros(N)
    h[0] = 1
    h[1:N//2] = 2*np.ones(N//2-1)
    h[N//2] = 1
    Z = X*h
    z = ifft(Z,N)
    return z

#plt.plot(position, signal)

plt.plot(position, analytic_signal(position))



#%%

#duration = 1.0
#fs = 40.0
#samples = int(fs*duration)
#t = np.arange(samples) / fs

#signal = chirp(t, 20.0, t[-1], 100.0)
#signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
#instantaneous_phase = np.unwrap(np.angle(analytic_signal))
#instantaneous_frequency = (np.diff(instantaneous_phase) /
#                           (2.0*np.pi) * fs)

#print(analytic_signal)

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(x, signal, label='signal')
ax0.set_xlabel("position /mm")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(x, amplitude_envelope, label='envelope')
#ax1.plot(t[1:], instantaneous_frequency)
#ax1.set_xlabel("time in seconds")
#ax1.set_ylim(0.0, 120.0)