import numpy as np
import matplotlib.pyplot as plt
import random
import math
import csv
from process_data2 import process

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def feedforward(X, B, W1, B2, W2):
    Z= np.dot(X, W1)+B
    A = sigmoid(Z)
    Z2= np.dot(A, W2)+B2
    A2= sigmoid(Z2)
    
    cache = { "Z": Z, "A": A, "W": W1, "B": B, "W2": W2, "B2": B2, "X": X} 

    return A2, cache

def cost(A2,Y):
    y= Y
    x= A2
    #y= y.reshape((y.shape[0], 1))
    m = Y.shape[1]
    
    cost = 1/m * (y * np.log(x) + (1 - y) * np.log(1 - x)).sum()
    cost = np.squeeze(cost)     

    return cost

def gradientDesc(A2, Y):

    y= Y
    x= A2
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

def accuracy(A2, Y):
    correct = 0
    
    for i in range(len(A2)):
        if(Y[i] == np.rint(A2[i])):
            correct = correct+1
    class_rate = correct/len(A2)
  
    return class_rate



def run2():
     x, y = process('./nbaStats/nba_data_2016-2018_control_real.csv')
     
         
     # learning rate for the algorithm
     learning_rate = .01

     # split into 75% train and 25% test

     train_size=.75
     X_train = x[:(int)(x.shape[0]*train_size),:]
     X_test = x[(int)(x.shape[0]*train_size):,:]
     Y_train = y[:(int)(y.shape[0]*train_size)]
     Y_test = y[(int)(y.shape[0]*train_size):]

     D= x.shape[1]
     M= 45
     B= np.random.rand(M)
     W1= np.random.rand(D,M)
     B2 = np.random.rand(1)
     W2 = np.random.rand(M,1)
     #print(D)
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
         #float problems results in necessary following problems
         X= np.array(X,dtype=np.float32)
         Y= np.array(Y,dtype=np.float32)

         A2, cache= feedforward(X, B, W1, B2, W2)

        
         
         l = cost(A2, Y)
         losses.append(-l)
         
         '''
         W2 += learning_rate* back_propW2(gradientDesc(Z2, Y), Z2, Z1)
         
         B2 += learning_rate* back_propB2(gradientDesc(Z2, Y), Z2)
         
         W1 += learning_rate* back_propW1(gradientDesc(Z2, Y), Z2, Z1, W2, X)
         
         B += learning_rate* back_propB1(gradientDesc(Z2, Y), Z2, Z1, W2)
         '''

         '''
         A2, cache = forwardprop(X, weights["W"],  weights["B"],  weights["W2"],  weights["B2"])
         compute_cost(A2, Y)
         grads= backprop(Y, A2, cache)
         weights = update_weights(weights, grads)
         '''
         #print(accuracy(Z2,Y))
         
         if(i>481):
           rates+=(accuracy(A2,Y)) *100
         
     
    
     print(rates/100)
     plt.title('Classifier 2 (control)')
     plt.plot(losses)
     plt.show()  
    
     return (rates/100)
  


run2()