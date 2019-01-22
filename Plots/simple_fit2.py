#!/usr/bin/python
import numpy as np
import scipy as sp
import pylab as pl
import scipy.optimize as spo
import scipy.fftpack as spf
import scipy.signal as sps

###############################################################################
###  This is just a little example fitting program
### in this example the fit is used to determine the wavelength of the light.
### It does this by fitting in small chunks of data
### to check consistency.
##########################################################################
### Note this should be used as a guide to the sort of thing that you can do 
### not as something to follow blindly
###########################################################################




def read_data(fname):
######################################################################
# A function that reads in the file. Written by DJC 1/10/2018
# Calling arguements:
#     fname = name of CSV file 
######################################################################
    file = open(fname,"r")
    lines= file.readlines()
    signal=[]
    t_sec=[]
    t_usec=[]
    position=[]
    for line in range(len(lines)):
        signal.append(int(lines[line].split()[0]))
        t_sec.append(int(lines[line].split()[1]))
        t_usec.append(int(lines[line].split()[2]))
        position.append(float(lines[line].split()[3]))
    
    
    
    file.close()
    return signal,t_sec,t_usec,position

def fit_func(x,amp,offset,phase,lam):
    #########################################
    ### x = position
    ### amp =amplitude of oscillation
    ### offset is the offfset from zero
    ### phase = phase
    ### lam = wavelength #remeber that the optical path is twice the stage movement
    #########################################
    return offset+amp*np.sin(2*np.pi*x/(lam/2.) + phase) # returns the expected "y" value 

def spectral_width(coherence_length):
    c = 3e8    
    del_v = c/(coherence_length*2*sp.pi)
    return del_v

def Wiens (wavelength_mean):
    T = 2.898e-3/wavelength_mean
    return T
 

#file="white_light_3_uni.txt"
#file="green_laser_0.002b.txt"
#file= "tung_y_0.0001_long.txt"
file="Output_data.txt"


y,t1,t2,x=np.array(read_data(file))
x=np.array(x)
pl.figure(1)

sp=pl.subplot(3,1,1)
pl.plot(x,y)

#pl.show() # useful to have an initial look at the data to set your initial guess

#now loop through the data fitting every 15 points
i=0
dstep=220
lamb=[]

#set un the initial initial_guess
#remember that the stage units are mm
#po=[1500000,1000000,0.,580e-6] #for green laser 
po=[150,5500,1.9,580e-6] # for tungsten with filter

yy2=np.zeros(dstep)
xx2=np.zeros(dstep)
xx3=np.zeros(dstep)
while (i+1)*dstep < len(x):
    xx=np.array(x[i*dstep:(i+1)*dstep]) 
    yy=np.array(y[i*dstep:(i+1)*dstep])
    sp=pl.subplot(3,1,1)
    pl.plot(xx,yy,'go')
    # now shift the data
    for j in range(dstep):
        xx2[j]=xx[j]-xx[0]    
    po,po_cov=spo.curve_fit(fit_func,xx2,yy,po, maxfev=10000 ,sigma=yy*0.01)
    #print "The Results",po
    #print "The covariance matrix",po_cov
    y1=fit_func(xx2,po[0],po[1],po[2],po[3])
    pl.plot(xx,y1,'b-',label='Fit results')
    #pl.show()
    lamb.append(po[3])
    sp=pl.subplot(3,1,2)
    pl.errorbar(xx.mean(),po[3], fmt='ro',yerr=np.sqrt(po_cov[3][3]))
    sp=pl.subplot(3,1,3)
    pl.errorbar(xx.mean(),np.abs(po[0]), fmt='bo',yerr=np.sqrt(po_cov[0][0]))

    i=i+1

sp=pl.subplot(3,1,1)
pl.xlabel("Stage position (mm)")
pl.ylabel("Signal (a.u.)")
sp=pl.subplot(3,1,2)
pl.xlabel("Stage position (mm)")
pl.ylabel("Fitted wavelength (mm)") 
sp=pl.subplot(3,1,3)
pl.xlabel("Stage position (mm)")
pl.ylabel("Fitted Amplitude")



#Histogram the results for the wavelength
pl.figure(55)
n,bins,p=pl.hist(lamb,bins=200)
pl.xlabel("Wavelength (mm)")
pl.ylabel("Number of entries")
#print "The n is",n
#print bins


pl.show()

