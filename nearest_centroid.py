# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:45:17 2018

@author: JoseEduardo
"""

#get the indexes of each category

#sum the vectors corresponding to these indexes

import numpy as np
import scipy as sp
import scipy.io as io
import pylab as pl

def load_usps_data(fname, digit=3):
    ''' Loads USPS (United State Postal Service) data from <fname> 
    Definition:  X, Y = load_usps_data(fname, digit = 3)
    Input:       fname   - string
                 digit   - optional, integer between 0 and 9, default is 3
    Output:      X       -  DxN array with N images with D pixels
                 Y       -  1D array of length N of class labels
                                 1 - where picture contains the <digit>
                                -1 - otherwise                           
    '''
    # load the data
    data = io.loadmat(fname)
    # extract images and labels
    X = data['data_patterns']
    Y = data['data_labels']
    Y = Y[digit,:]
    return X, Y

digit = 3

X,Y = load_usps_data('usps.mat',digit)

#indexes of the samples which belong to each category
ind1 = (1.0 == Y).nonzero()[0]
ind2 = (-1.0 == Y).nonzero()[0]
#centroids for each category
C1 = np.sum(X[:,ind1],axis=1)/ind1.shape
C2 = np.sum(X[:,ind2],axis=1)/ind2.shape
#distances to each Centroid
dist1 =  np.amax(X-np.outer(C1,np.ones(Y.shape)),axis=0)
dist2 =  np.amax(X-np.outer(C2,np.ones(Y.shape)),axis=0)
#indexes of the classification made by the algorithm for each category
Mind1 = (dist1<dist2).nonzero()[0]
Mind2 = (dist1>dist2).nonzero()[0]
#accuracy
acc = (X.shape[1]-(Mind1.shape[0]-ind1.shape[0]))/X.shape[1]






























