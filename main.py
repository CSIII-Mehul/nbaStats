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
    A1= np.dot(Z, W2)+B2
    Z2= sigmoid(A1)

    return Z2, Z

def cost(Z2,Y):

    y= Y
    x= Z2
    y= y.reshape((y.shape[0], 1))
    
    return (y * np.log(x) + (1 - y) * np.log(1 - x)).sum()

def gradientDesc(Z2, Y):

    y= Y
    x= Z2
    y= y.reshape((y.shape[0], 1))
    
    return (y/x) - (1-y)/(1-x)



def back_propW2(gradientCost, Z2, Z):
    gradientCost = gradientCost.reshape((gradientCost.shape[0], 1))

    return Z.T.dot(gradientCost* Z2 * (1-Z2))

def back_propB2(gradientCost, Z2):
    gradientCost = gradientCost.reshape((gradientCost.shape[0], 1))

    return (gradientCost* Z2 * (1-Z2)).sum(axis=0)

def back_propW1(gradientCost, Z2, Z, W2, X):
    gradientCost = gradientCost.reshape((gradientCost.shape[0], 1))
    
    preds = (gradientCost* Z2* (1-Z2))
    preds_0 = (preds.dot(W2.T)  * Z *(1-Z))
  
    weights= np.dot(X.T, preds_0)
  
    return weights

def back_propB1(gradientCost, Z2, Z, W2):
    gradientCost = gradientCost.reshape((gradientCost.shape[0], 1))

    preds = (gradientCost* Z2* (1-Z2))
    preds_0 = (preds.dot(W2.T)  * Z *(1-Z))
  
    return preds_0.sum(axis=0)

def accuracy(Z2, Y):
    correct = 0
    
    for i in range(len(Z2)):
        if(Y[i] == np.rint(Z2[i])):
            correct = correct+1
    class_rate = correct/len(Z2)
  
    return class_rate

def run():
     x, y = process('nba_data2016-2018.csv')
         
     # learning rate for the algorithm
     learning_rate = 0.00000001

     # split into 75% train and 25% test

     train_size=.75
     X_train = x[:(int)(x.shape[0]*train_size),:]
     print(X_train[0])
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
         
        #print(accuracy(Z2,Y))
     plt.plot(losses)
     plt.show()  


     return 0



run()

