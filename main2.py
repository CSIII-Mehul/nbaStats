import numpy as np
import matplotlib.pyplot as plt
import random
import math
import csv
from process_data2 import process



def run2():
     x, y = process('nba_data2016-2018.csv')
         
     # learning rate for the algorithm
     learning_rate = 0.001

     # split into 75% train and 25% test

     train_size=.75
     X_train = x[:(int)(x.shape[0]*train_size),:]
     print(X_train[len(X_train)-1])
     X_test = x[(int)(x.shape[0]*train_size):,:]
     Y_train = y[:(int)(y.shape[0]*train_size)]
     Y_test = y[(int)(y.shape[0]*train_size):]

     D= x.shape[1]
     M= 20
     B= np.random.rand(M)
     W1= np.random.rand(D,M)
     B2 = np.random.rand(1)
     W2 = np.random.rand(M,1)

     #each batch is now 6 "lines" because 3498/583=6
     batches = 583


     X_t= np.split(X_train , batches, axis=0)
     Y_t = np.split(Y_train , batches, axis=0)

     # IMPORTANT: statistic = (Yes/No - No/Yes)^2 / (Yes/No + No/Yes), Is the Mcnemar's test (a type of chi-square), to compare between 2 binary classification algorithms; with an alpha level of .05, the critical value is 3.84
     losses= []
     rates = 0
     
     for i in range(len(X_t)):
         X= X_t[i]
         Y= Y_t[i]
         
  
         Z2, Z1= feedforward(X, B, W1, B2, W2)

         l = cost(Z2, Y)
         losses.append(-l)
         
         
         W2 += learning_rate* back_propW2(gradientDesc(Z2, Y), Z2, Z1)
         
         B2 += learning_rate* back_propB2(gradientDesc(Z2, Y), Z2)
         
         W1 += learning_rate* back_propW1(gradientDesc(Z2, Y), Z2, Z1, W2, X)
         
         B += learning_rate* back_propB1(gradientDesc(Z2, Y), Z2, Z1, W2)
         
         
         if(i>481):
           rates+=(accuracy(Z2,Y))

     
     
     plt.plot(losses)
     plt.show()  
    

     return (rates/100)



run2()