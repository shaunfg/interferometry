#!/usr/bin/python

import numpy as np
import numpy.random as npr
import scipy as sp
import pylab as pl
import scipy.fftpack as spf

def add_line(x,y,wl,amp,width,nstep):
    """
    This little function adds the effect
    of a a new line on to the interferogram.
    It does this by assuming that each line is made up of lots of descrete delta functions. Also assumes a gausian line shape
    and calculates to +/- 3 sigma
    x is the separation between the mirrors
    y is the amplitude of the light
    wl is the wavelength
    amp is the amplitude (arbitrary scale)
    width is the line width (actually 1 sigma as we assume gaussian)
    nsteps is the number 
    """
    #nwidth=30.
    nsigma=5
    amplitude=amp*calc_amp(nsigma,nstep)
    wl_step=nsigma*2.0*width/nstep
    for i in range(len(amplitude)):
        wavelength=wl-nsigma*width+i*wl_step
        y=y+amplitude[i]*np.sin(np.pi*2.*x/wavelength)        
    return y

def calc_amp(nsigma,nsamp):
    """
    Just calculates the amplitude at the various steps
    """
    yy=np.empty(shape=[nsamp])
    step=nsigma*2.0/nsamp
    for i in range(nsamp):
        x=-nsigma+i*step
        size=np.exp(-x*x/4)
        yy[i]=size
    return yy
  




# Now set up the experiment that you want to do

# Na lines
l1=589e-9 # wavelength of spectral line in m
l2=589.6e-9 # wavelength of a second spectral line in m
w1=w2=120.e-9 # setting the lines to have the same width in m


'''
 When you perform the actual experiment you will move
 one mirror to change the path difference. This move will be 
 by a small, finite, amount. You will then take a reading with your detector. 
 Then you will move the mirror again and take another 
 reading and so on. Here you should set up the what these different 
 separations should be
'''
'''
Change these to set up the experiment 
'''
nsamp=1000 #number of samples that you will take (set in the software)
dsamp=40.e-9 #distance moved between samples


# set the starting point from null point
dstart= -1e-5 # start -3mm from null point

#epoint=dstart+dsamp*nsamp



x= np.linspace(dstart,-dstart,nsamp) #setting the x locations of the samples



y=np.zeros(shape=[len(x)]) #setting the array that will contain your results



# Na spectrum (roughly)

#y=add_line(x,y,l2,1.0,w1,50)
y=add_line(x,y,l1,1.0,w2,1000)#nstep smooths out gaussian profile



# plot the output

pl.figure(1)
pl.plot(x,y,'bo')
pl.xlabel("Distance from null point (m)")
pl.ylabel("Amplitude")





pl.show()
