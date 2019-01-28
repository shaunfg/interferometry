# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:03:28 2019

@author: cnh17
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert,butter,filtfilt,chirp
from scipy.optimize import curve_fit
#import random

# Define some test data which is close to Gaussian
#fname = 'blue_LED.txt'
fname = 'green_tungsten.txt'

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


position = np.array(position)
position = position.astype(float)

signal = np.array(signal)

analytic_signal = hilbert(signal,axis= -1)

#plt.plot(np.real(analytic_signal))
analytic_imag_signal = hilbert((np.imag(analytic_signal)))
amplitude_envelope = np.abs(analytic_imag_signal)

plt.figure(figsize = [13,8])
plt.plot(position,np.imag(analytic_signal)) #plots the curve along the y axis
plt.plot(position,amplitude_envelope)
plt.plot(position,-amplitude_envelope)

y = amplitude_envelope
x = position

#%% beating fit
y = amplitude_envelope
x = position

x2 = np.array([i for i in x if i >= 0.63])
y = np.array(y[:len(x2)])

def beating (t,amplitude,period,period_del,phase,tau):
    func = amplitude*(np.cos(t * 2. * np.pi/period + phase)*
                      np.cos(t * 2. * np.pi/period_del + phase))* np.exp(-t/tau)
    return func

guess_amplitude = 20
guess_period = 0.470015001032
guess_period_del = 0.001
guess_phase = 0.5
guess_tau = 1/20

p0 = [guess_amplitude,guess_period,guess_period_del,guess_phase,guess_tau]

fit = curve_fit (beating,x2,y,p0=p0,maxfev = 10000)

data_fit= beating(x2 , *fit[0])
plt.figure(figsize = [13,8])
plt.plot(x,np.imag(analytic_signal),label='data') #plots the curve along the y axis
plt.plot (x2, 20*data_fit,'r:')

#%% beating fit with different range ... takes a bit longer to load
y = amplitude_envelope
x = position

x2 = np.array([i for i in x if i >= 0.63])
x2 = np.array([i for i in x2 if i <= 0.67])
y = np.array(y[:len(x2)])

def beating (t,amplitude,period,period_del,phase,tau):
    func = amplitude*(np.cos(t * 2. * np.pi/period + phase)*
                      np.cos(t * 2. * np.pi/period_del + phase))* np.exp(-t/tau)
    return func

guess_amplitude = 6000
guess_period = 0.054
guess_period_del = 0.0021
guess_phase = 0.5
guess_tau = 1/20

p0 = [guess_amplitude,guess_period,guess_period_del,guess_phase,guess_tau]

fit = curve_fit (beating,x2,y,p0=p0,maxfev = 100000)

data_fit= beating(x2 , *fit[0])
plt.plot(x,np.imag(analytic_signal),label='data') #plots the curve along the y axis
plt.plot (x2, 300*data_fit,'r:')


#%%
#Gaussian Fit

y = amplitude_envelope
x = position

n = len(x)
mean = sum(x)/n
sigma = np.sqrt(sum((x-mean)**2)/n)

#Test Function
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma],maxfev=5000)
#
# 1548668277 266635  0.700009 322
# 1548668277 266635  0.700009 322
#49 1548668277 286679  0.700000 327
#49 1548668277 286679  0.700000 327
plt.figure()
plt.plot(x,np.imag(analytic_signal),label='data') #plots the curve along the y axis
plt.plot(x,gaus(x,*popt),'r',label='fit')
plt.legend()

FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma

print "Coherence Length =", FWHM

#%%

a = [1,2,3,4,5]

print(a[-3:])

#%%
 import numpy as np
 import matplotlib.pyplot as plt
 from scipy.signal import hilbert, chirp

 duration = 1.0
 fs = 1000.0
 samples = int(fs*duration)
 t = np.arange(samples) / fs

 signal = chirp(t, 20.0, t[-1], 100.0)
 signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

 analytic_signal = hilbert(signal)
 amplitude_envelope = np.abs(analytic_signal)
 instantaneous_phase = np.unwrap(np.angle(analytic_signal))
 instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs

 fig = plt.figure()
 ax0 = fig.add_subplot(211)
 ax0.plot(t, signal, label='signal')
 ax0.plot(t, amplitude_envelope, label='envelope')
 ax0.set_xlabel("time in seconds")
 ax0.legend()
 ax1 = fig.add_subplot(212)
 ax1.plot(t[1:], instantaneous_frequency)
 ax1.set_xlabel("time in seconds")
 ax1.set_ylim(0.0, 120.0)