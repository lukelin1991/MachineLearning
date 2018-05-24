#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:04:44 2018

@author: ljlin
"""

# Numpy practice

#numpy ndarray

#import numpy
import numpy as np
data = [[2, 6, 1, 3, 7], [5, 10, 4, 9, 8]]
data = np.array(data)
print (data)
'''output: 
    [[2 6 1 3 7]
     [5 10 4 9 8]] '''
    
print (data.shape)
'''Output:
    (2, 5)'''

# produce an array/list of all 0's

print (np.zeros((2, 3)))
'''Output:
    [[ 0. 0. 0.]
     [ 0. 0. 0.]]'''

#produce an array/list of all 1's
print (np.ones((2, 3)))
'''Output:
    [[ 1. 1. 1.]
     [ 1. 1. 1.]]'''

array = np.arange(10)
print (array)

print (array[2:5])

array[5:8] = 0
print (array)
'''removes 5, 6, 7 from the array'''

