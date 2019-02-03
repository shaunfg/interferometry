#!/usr/bin/python

###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import numpy as np
import pylab as pl
import scipy.fftpack as spf

# tell it what file to open
#fname = 'mercury_6_RL.txt' #4,6,7 are good sets for mercury
fname = 'Hg_green_good.txt'
#fname = 'Hg_yellow_good.txt' #shows the beating shown by a yellow doublet

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


pl.plot(position,signal,'gold')
pl.xlabel("Position / mm")
pl.ylabel("Signal / a.u.")

# if you had wanted to plot just against sample number
#x=range(len(signal))
#pl.plot(x,signal)
#pl.xlabel("Sample number")
#pl.ylabel("Signal")

pl.show()

x = position
y = signal
nsamp = len(x)
sampling_speed = 0.005e-3 #m/s
dsamp = 2 * sampling_speed / 50 #times two as something due to the path length
#broadness due to error , work out how to reduce this error

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

pl.figure(3)
pl.plot(repx,abs(yf[int(len(xf)/2+1):len(xf)]),color = 'gold')
pl.xlabel("Wavelength (m)")
pl.ylabel("Amplitude / a.u.")


pl.show()
#%% analysis

y = abs(yf[int(len(xf)/2+1):len(xf)])
y = list(y)

max_y = max(y)
max_x = repx[y.index(max_y)]

print(max_x,max_y)

err_x = 0.000025e-3

print("wavelength =",max_x, "+/-", err_x, "m")
print("True Value = 5.461e-7 m")