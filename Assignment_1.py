# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:39:11 2018

@author: JoseEduardo
"""

import scipy as sp
import scipy.io as io
import pylab as pl


#matplotlib inline

''' ---- Functions for loading and plotting the images ---- '''
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

def plot_img(a):
    ''' Plots one image 
    Definition: plot_img(a) 
    Input:      a - 1D array that contains an image 
    '''   
    a2 = sp.reshape(a,(int(sp.sqrt(a.shape[0])), int(sp.sqrt(a.shape[0]))))
    pl.imshow(a2, cmap='gray') 
    pl.colorbar()
    pl.setp(pl.gca(), xticks=[], yticks=[])
            
def plot_imgs(X, Y):   
    ''' Plots 3 images from each of the two classes 
    Definition:         plot_imgs(X,Y)
    Input:       X       -  DxN array of N pictures with D pixel
                 Y       -  1D array of length N of class labels {1, -1}                  
    '''
    pl.figure()
    for i in sp.arange(3):
        classpos = (Y == 1).nonzero()[0]
        m = classpos[sp.random.random_integers(0, classpos.shape[0]-1)]
        pl.subplot(2,3,1+i)
        plot_img(X[:, m])
    for i in sp.arange(3):
        classneg = (Y != 1).nonzero()[0]
        m = classneg[sp.random.random_integers(0, classneg.shape[0]-1)]
        pl.subplot(2,3,4+i)
        plot_img(X[:, m])
        
        
        
def train_perceptron(X,Y,iterations=200,eta=.1, option=0):
    ''' Trains a linear perceptron
    Definition:  w, b, acc  = train_perceptron(X,Y,iterations=200,eta=.1)
    Input:       X       -  DxN array of N data points with D features
                 Y       -  1D array of length N of class labels {-1, 1}
                 iter    -  optional, number of iterations, default 200
                 eta     -  optional, learning rate, default 0.1
                 option  -  optional, defines how eta is updated in each iteration
    Output:      w       -  1D array of length D, weight vector 
                 b       -  bias term for linear classification                          
                 acc     -  1D array of length iter, contains classification accuracies 
                            after each iteration  
                            Accuracy = #correctly classified points / N 
    '''
    assert option == 0 or option == 1 or option == 2
    acc = sp.zeros((iterations))
    #include the bias term by adding a row of ones to X 
    X = sp.concatenate((sp.ones((1,X.shape[1])), X))
    #initialize weight vector
    weights = sp.ones((X.shape[0]))/X.shape[0]
    for it in sp.arange(iterations):
        # indices of misclassified data
        wrong = (sp.sign(weights.dot(X)) != Y).nonzero()[0]
        # compute accuracy acc[it] (1 point)
        acc[it] = (Y.shape[0]-wrong.shape[0])/Y.shape[0]
        # ... your code here
        if wrong.shape[0] > 0:
            # pick a random misclassified data point (2 points)
            imiss = wrong[sp.random.randint(0,wrong.shape[0])]
            Xmiss = X[:,imiss]
            
            #update weight vector (using different learning rates ) (each 1 point)
            if option == 0:
                lr = eta/(1+it)
            elif option == 1:
                lr = eta
            elif option == 2:
                lr = eta*(1+it)
                
            weights += lr*Xmiss*Y[imiss]
                
                
    b = -weights[0] 
    w = weights[1:]
    #return weight vector, bias and accuracies
    return w,b,acc


def analyse_accuracies_perceptron(digit = 3, option=0):
    ''' Loads usps.mat data and plots digit recognition accuracy in the linear perceptron
    Definition: analyse_perceptron(digit = 3)
    '''
    X,Y = load_usps_data('usps.mat',digit)
    w_per,b_per,acc = train_perceptron(X,Y, option=option)
    
    pl.figure()
    pl.plot(sp.arange(len(acc)),acc)
    pl.title('Digit recognition accuracy')      
    pl.xlabel('Iterations')
    pl.ylabel('Accuracy')
    


X,Y = load_usps_data('usps.mat', digit=3) 

#w,b,acc = train_perceptron(X,Y,iterations=200,eta=.1, option=2)

analyse_accuracies_perceptron(digit = 3, option=0)   
