#!/usr/bin/python

###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import numpy as np
import pylab as pl

# tell it what file to open
#fname ='white_led_centre_farback.txt'
#fname = 'Output_data.txt'
fname = 'white_light.txt'


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


pl.plot(position,signal,'-')
pl.xlabel("Position in mm")
pl.ylabel("Signal")

# if you had wanted to plot just against sample number
#x=range(len(signal))
#pl.plot(x,signal)
#pl.xlabel("Sample number")
#pl.ylabel("Signal")

pl.show()
