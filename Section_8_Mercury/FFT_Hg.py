#!/usr/bin/python

###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import numpy as np
import pylab as pl
import scipy.fftpack as spf

# tell it what file to open
#fname ='yellow_tungsten_1.txt'
#fname = 'white_tungsten.txt'
#fname = 'blue_LED.txt'
#fname ='yellow_tungsten_2.txt'
#fname = 'green_tungstensten.txt'
#fname = 'mercury_6_RL.txt' #4,6,7 are good sets for mercury
fname = 'Hg_green_1.txt'

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

# take a fourier transform
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
