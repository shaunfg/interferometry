# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:03:28 2019

@author: cnh17
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert,butter,filtfilt,chirp
from scipy.optimize import curve_fit
import scipy.fftpack as spf
import os
import pylab as pl

path = "/Users/ShaunGan/Desktop/interferometry/Tungsten"
os.chdir(path)

# Define some test data which is close to Gaussian
#fname = 'blue_LED.txt'
fname = 'yellow_tungsten_3_good.txt'

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
ref = np.imag(analytic_signal)

plt.plot(x,y)
plt.figure(figsize = [8,5])

def beating(x,amplitude,scale,phase):
    return amplitude * np.sinc(scale * x - phase)

guess_amplitude = 3000
guess_scale = 1.094
guess_phase = 0.63

#plt.plot(np.linspace(-0.56,0.7,100),beating(np.linspace(-0.56,0.7,100),
#                     guess_amplitude,guess_scale,guess_phase))
#
p0 = [guess_amplitude,guess_scale,guess_phase]
fit = curve_fit (beating,x,y,p0=p0,maxfev = 1000000)
data_fit = beating(x , *fit[0])

plt.plot(x,ref,label='data') #plots the curve along the y axis
plt.plot (x, data_fit,'r:')
plt.figure(figsize = [8,5])

nsamp = len(x)
sampling_speed = 0.005e-3 #m/s
dsamp = 2 * sampling_speed / 50 #times two as something due to the path length
#broadness due to error , work out how to reduce this error

Fs= 50
Ts= dsamp/sampling_speed
t= np.linspace(-1,Ts,100)
f=1
N= nsamp

#t = np.linspace(-100,100,1000)
#f = 1 * np.pi

func_y = beating(t*f, *fit[0])

#plt.figure()
#plt.plot(t,func_y)
#plt.xlabel('Position / mm')
#plt.ylabel(' magnitude')

fy=(spf.fft(func_y,N))

plt.figure()
fr = np.multiply(np.arange(0,N-1,1),Fs/N)
plt.plot(fr,spf.fftshift(abs(fy))[:nsamp-1])
plt.xlabel(' inverse of distance x^{-1}')
plt.ylabel(' magnitude')


# take a fourier transform
yf=spf.fft(y)
xf=spf.fftfreq(nsamp) # setting the correct x-axis for the fourier transform. Osciallations/step

#now some shifts to make plotting easier (google if ineterested)
xf=spf.fftshift(xf)
yf=spf.fftshift(yf)

plt.figure(figsize = [8,5])

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

pl.figure(3)
pl.plot(repx,abs(yf[int(len(xf)/2+1):len(xf)]))
pl.xlabel("Wavelength (m)")
pl.ylabel("Amplitude / a.u.")




#%%
Fs=1
Ts=1/Fs
t= np.linspace(-1,Ts,100)
f=5

#t = np.linspace(-100,100,1000)
#f = 1 * np.pi

func_y = beating(t*f, *fit[0])

#plt.figure()
#plt.plot(t,func_y)
plt.xlabel('Position / mm')
plt.ylabel(' magnitude')

N=512

fy=(spf.fft(func_y,N))

#plt.figure()
fr = np.multiply(np.arange(0,N-1,1),Fs/N)
plt.plot(fr,spf.fftshift(abs(fy))[:511])
plt.xlabel(' distance in mm ')
plt.ylabel(' magnitude')

#%% Ignore, test code

Fs=1
# list of values
#f=50 #scale value

#t = np.linspace(-100,100,1000)
#f = 1 * np.pi

func_y = beating(x, *fit[0])

#plt.figure()
#plt.plot(t,func_y)
plt.xlabel('Position / mm')
plt.ylabel(' magnitude')

N=nsamp

fy=(spf.fft(func_y,N))

plt.figure()
fx = np.multiply(np.arange(0,N-1,1),Fs/N)
fy = spf.fftshift(abs(fy))
plt.plot(fx,spf.fftshift(abs(fy))[:nsamp-1])
plt.xlabel(' distance in ^(-1) ')
plt.ylabel(' magnitude')

#xf=spf.fftfreq(nsamp)

xx=fx[int(len(xf)/2+1):len(xf)]
repx=dsamp/xx

pl.figure(3)
pl.plot(repx,abs(fx[int(len(fx)/2+1):len(fx)]))
pl.xlabel("Wavelength (m)")
pl.ylabel("Amplitude / a.u.")

