#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:38:58 2019

@author: ShaunGan
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy.signal import hilbert
import scipy.fftpack as spf
from scipy.optimize import curve_fit
import os

path = "/Users/ShaunGan/Desktop/interferometry/All_Other_Lights"
os.chdir(path)

# Define some test data which is close to Gaussian
#fname = 'blue_LED.txt'
fname = 'white_tungsten.txt'

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

plt.figure()#figsize = [13,8])
plt.plot(position,np.imag(analytic_signal)) #plots the curve along the y axis
plt.plot(position,amplitude_envelope,color = 'orange')
plt.plot(position,-amplitude_envelope,color = 'orange')
pl.xlabel("Position in mm")
pl.ylabel("Signal (a.u.)")

y = amplitude_envelope
x = position

n = len(x)
mean = sum(x)/n
sigma = np.sqrt(sum((x-mean)**2)/n)

#Test Function
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

#Curve Fit
popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma],maxfev=100000)

#Plotting
plt.figure()
plt.plot(x,np.imag(analytic_signal),label='Data') #plots the curve along the y axis
plt.plot(x,gaus(x,*popt),'r',label='Gaussian Fit')
plt.plot(x,-gaus(x,*popt),'r')
pl.xlabel("Position in mm")
pl.ylabel("Signal (a.u.)")



plt.legend()

#%%

x = position
y = signal
nsamp = len(x)
sampling_speed = 0.05e-3 #m/s
dsamp = 2 * sampling_speed / 50 #times two as something due to the path length

# take a fourier transform
yf=spf.fft(y)
xf=spf.fftfreq(nsamp) # setting the correct x-axis for the fourier transform. Osciallations/step

#now some shifts to make plotting easier (google if ineterested)
xf=spf.fftshift(xf)
yf=spf.fftshift(yf)

pl.figure(2)
pl.plot(xf,np.abs(yf))
pl.xlabel("Oscillations per sample")
pl.ylabel("Amplitude / a.u.")

# Now try to reconstruct the original wavelength spectrum
# only take the positive part of the FT
# need to go from oscillations per step to steps per oscillation
# time the step size

xx=xf[int(len(xf)/2+1):len(xf)]
repx=dsamp/xx
repy=abs(yf[int(len(xf)/2+1):len(xf)])
pl.figure(3)
pl.plot(repx,repy)
pl.xlabel("Wavelength (mm)")
pl.ylabel("Amplitude / a.u.")

#%% Extracting data from plots
max_y = max(repy)
max_x = repx[list(repy).index(max_y)]

n = len(repx)
mean = max_x
sigma = np.sqrt(sum((repx-mean)**2)/n)

#Curve Fit
popt,pcov = curve_fit(gaus,repx,repy,p0=[1,mean,sigma],maxfev=100000)

#Plotting
#plt.plot(repx,gaus(repx,*popt),'r',label='fit')

gaus_max_y = max(gaus(repx,*popt))
gaus_max_x = repx[list(gaus(repx,*popt)).index(gaus_max_y)]

peak_y = max(repy)
peak_x = repx[list(repy).index(peak_y)]
print("peak wavelength = ", peak_x)
print("mean wavelength from fit = ",gaus_max_x)
pl.show()

#%% Analysis ~ Task 9
#Error in signal and error in position. 
#Error in signal can multiply by 
#not a clear way to do error propagation in time. 
#just chuck on the error on wavelength and comment about the error intensity. 

c = 3e8
FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma #representing Coherence Length
spectral_width_freq = c / (2 * np.pi * FWHM)  
spectral_width_wave = spectral_width_freq * gaus_max_x ** 2 / c

def Wiens(lamda_peak):
    T = 2.898e-3 / lamda_peak
    return T

print ("Coherence Length =", FWHM,"m")
print("Spectral width of Frequency = ",spectral_width_freq)
print("Spectral width of Wavelength = ",spectral_width_wave)
print ("T from Wien's Law =",Wiens(gaus_max_x),"K, seems too small")