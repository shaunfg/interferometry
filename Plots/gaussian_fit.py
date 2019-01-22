# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:03:28 2019

@author: cnh17
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import random

# Define some test data which is close to Gaussian
file = 'white_led_3_uni.txt'

y,t1,t2,x=np.array(read_data(file))

data = random.sample(x,5)

n = len(x)                          #the number of data
mean = sum(x*y)/n                   #note this correction
sigma = sum(y*(x-mean)**2)/n   

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1., 0., 1.]

coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)

# Get the fitted curve
hist_fit = gauss(x, *coeff)

plt.plot(x, y, label='Test data')
plt.plot(x, hist_fit, label='Fitted data')

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print 'Fitted mean = ', coeff[1]
print 'Fitted standard deviation = ', coeff[2]

plt.show()