#!/usr/bin/python

###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import numpy as np
import pylab as pl
import scipy.fftpack as spf
import matplotlib.pyplot as plt

# tell it what file to open
#fname ='yellow_tungsten_1.txt'
#fname = 'white_tungsten.txt'
#fname = 'blue_LED.txt'
#fname ='yellow_tungsten_2.txt'
fname = 'green_tungsten.txt'
#fname = 'mercury_6_RL.txt' #4,6,7 are good sets for mercury
#fname = 'Hg_green_1.txt'

#fname ='Output_data.txt'
#
f=open(fname,'r')

###################################################################################################
### The file format is: [signal] [seconds this epoch] [usec this second] [position
### All four will be in arrays for you to analyse
###################################################################################################

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


pl.plot(position,signal,'g-')
pl.xlabel("Position in mm")
pl.ylabel("Signal")

# if you had wanted to plot just against sample number
#x=range(len(signal))
#pl.plot(x,signal)
#pl.xlabel("Sample number")
#pl.ylabel("Signal")

pl.show()

x = position
y = signal
nsamp = len(x)
sampling_speed = 0.0005e-3 #m/s
dsamp = sampling_speed / 50

Fs= 50
Ts=1/Fs
t= np.linspace(-Ts,Ts,100000)
f= 1

N = nsamp
t = position

fy=(spf.fft(y,N))

plt.figure()
#fr = np.multiply(np.arange(0,N-1,1),Fs/N)
fr = position
plt.plot(fr,spf.fftshift(abs(fy)))
plt.xlabel(' distance / mm')
plt.ylabel(' magnitude')


#%% take a fourier transform
# Only difference between this one and the one above is that this one has the
# x values shifted to zero... i think 
# i'm not sure why you need to do something to go from fig. 2 to 3, 
# will ask demonstrator


yf=spf.fft(y)
xf=spf.fftfreq(nsamp) # setting the correct x-axis for the fourier transform. Osciallations/step

#now some shifts to make plotting easier (google if ineterested)
xf=spf.fftshift(xf)
yf=spf.fftshift(yf)


pl.figure(2)
pl.plot(xf,np.abs(yf))
pl.xlabel("Oscillations per sample")
pl.ylabel("Amplitude")


# Now try to reconstruct the original wavelength spectrum
# only take the positive part of the FT
# need to go from oscillations per step to steps per oscillation
# time the step size


xx=xf[int(len(xf)/2+1):len(xf)]
repx=dsamp/xx

pl.figure(3)
pl.plot(repx,abs(yf[int(len(xf)/2+1):len(xf)]))
pl.xlabel("Wavelength (m)")
pl.ylabel("Amplitude")


pl.show()

#%% TEST

import matplotlib.pyplot as plt
import scipy.fftpack as spf
import numpy as np

Fs=1
Ts=1/Fs
t= np.linspace(-1,Ts,100)
f=5

#t = np.linspace(-100,100,1000)
#f = 1 * np.pi

y = np.sinc(t*f)

plt.figure()
plt.plot(t,y)
plt.xlabel('x')
plt.ylabel(' magnitude')

N=512

fy=(spf.fft(y,N))

plt.figure()
fr = np.multiply(np.arange(0,N-1,1),Fs/N)
plt.plot(fr,spf.fftshift(abs(fy))[:511])
plt.xlabel(' frequency x^{-1}')
plt.ylabel(' magnitude')
