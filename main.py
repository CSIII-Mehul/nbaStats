import numpy as np
import matplotlib.pyplot as plt
import random
import math
import csv
from process_data import process

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def feedforward(X, B, W1, B2, W2):
    A= np.dot(X, W1)+B
    Z= sigmoid(A)
    A1= np.dot(Z, W2)+B
    Z2= sigmoid(A2)

def cost(T,Y):
    return (T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()
'''
def gradient_BCE(Y, T):
    
    return (T/Y) - (1-T)/(1-Y)
'''
# split into 80% train and 20% test
def run():
     x, y = process('nba_data2016-2018.csv')
         

     train_size=.8
     X_train = x[:(int)(x.shape[0]*train_size),:]
     X_test = x[(int)(x.shape[0]*train_size):,:]
     Y_train = y[:(int)(y.shape[0]*train_size)]
     Y_test = y[(int)(y.shape[0]*train_size):]

     D= x.shape[1]
     M= 20
     B= np.random.rand(M)
     W1= np.random.rand(D,M)
     B2 = np.random.rand(1)
     W2 = np.random.rand(M,1)

     #each batch is now 3 "lines" because 3732/1244=3
     batches = 1244

     X= np.split(X_train , batches, axis=0)
     Y = np.split(Y_train , batches, axis=0)

     for i in range(len(batches)):
         X= X[i]
         Y= Y[i]

         Z2= feedforward(X, B, W1, B2, W2)
         

        


     return 0



run()

