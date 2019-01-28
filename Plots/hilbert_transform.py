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

#%% beating fit with sinc function
y = amplitude_envelope
x = position

def beating(x,amplitude,scale,phase):
    return amplitude * np.sinc(scale * x - phase)

guess_amplitude = 3000
guess_scale = 1
guess_phase = 0.63

plt.figure(figsize = [13,8])
plt.plot(np.linspace(-0.56,0.7,100),beating(np.linspace(-0.56,0.7,100),
                     guess_amplitude,guess_scale,guess_phase))

p0 = [guess_amplitude,guess_scale,guess_phase]

fit = curve_fit (beating,x,y,p0=p0,maxfev = 1000000)

data_fit = beating(x , *fit[0])
plt.figure(figsize = [13,8])
plt.plot(x,np.imag(analytic_signal),label='data') #plots the curve along the y axis
plt.plot (x, data_fit,'r:')



#%% Gaussian Fit

y = amplitude_envelope
x = position

n = len(x)
mean = sum(x)/n
sigma = np.sqrt(sum((x-mean)**2)/n)

#Test Function
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

#Curve Fit
popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma],maxfev=5000)

#Plotting
plt.figure()
plt.plot(x,np.imag(analytic_signal),label='data') #plots the curve along the y axis
plt.plot(x,gaus(x,*popt),'r',label='fit')
plt.legend()

#Analysis
c = 3e8
FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma #representing Coherence Length
spectral_width = c / (2 * np.pi * FWHM)  

def Wiens(lamda_peak):
    T = 2.898e-3 / lamda_peak
    return T

print ("Coherence Length =", FWHM)

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

