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
    
    cost =  1/m * (Y*np.log(A2)).sum()

    cost = np.squeeze(cost)     

    return cost

def gradientDesc(A2, Y):

    y= Y
    x= A2
   # y= y.reshape((y.shape[0], 1))
    
    return (y/x) - (1-y)/(1-x)

def backprop(Y, A2, cache):
    m= Y.shape[1]

    #dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    dZ2 = (A2-Y)
  #  dZ2 = gradientDesc(A2,Y)

    dW2=  1/m * np.dot(cache["A"].T, dZ2)
    dB2=  1/m * np.sum(dZ2, axis=1, keepdims = True)
    dA = np.dot(dZ2, cache["W2"].T)
    dZ= dA * (1-dA)
    dW=  1/m * np.dot(cache["X"].T, dZ)
    dB= 1/m * np.sum(dZ, axis=1, keepdims = True)


    grads = { "dW": dW, "dB": dB, "dW2": dW2, "dB2": dB2}

    return grads

def update_weights(weights, grads, learning_rate = .25):

    weights["W2"] = weights["W2"] + learning_rate*grads["dW2"]
    weights["B2"] = weights["B2"] + learning_rate*grads["dB2"]
    weights["W1"] = weights["W1"] + learning_rate*grads["dW"]
    weights["B"] = weights["B"] + learning_rate*grads["dB"]


    return weights

def accuracy(A2, Y):
    correct = 0
    
    for i in range(len(A2)):
        if(Y[i] == np.rint(A2[i])):
            correct = correct+1
    class_rate = correct/len(A2) *100
  
    return class_rate



def run2():
     x, y = process('./nbaStats/nba_data_2016-2018_control_real.csv')
     
         

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

     weights = {"B": B, "W1": W1, "W2": W2, "B2": B2 }
     #print(D)
     #each batch is now 6 "lines" because 3498/583=6
     batches = 583


     X_t= np.split(X_train , batches, axis=0)
     Y_t = np.split(Y_train , batches, axis=0)

     # IMPORTANT: statistic = (Yes/No - No/Yes)^2 / (Yes/No + No/Yes), Is the Mcnemar's test (a type of chi-square), to compare between 2 binary classification algorithms; with an alpha level of .05, the critical value is 3.84
     losses= []
     rates = 0
     count = 0
     for i in range(len(X_t)):
         X= X_t[i]
         Y= Y_t[i]
         #float problems results in necessary following problems
         X= np.array(X,dtype=np.float32)
         Y= np.array(Y,dtype=np.float32)

         A2, cache= feedforward(X, weights["B"], weights["W1"], weights["B2"], weights["W2"])
 
         l = cost(A2, Y)
         losses.append(-l)

         grads= backprop(Y, A2, cache)
         weights = update_weights(weights, grads)
        
         
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
         
         
         #rates += (accuracy(A2,Y))
         #count += 1
         
     
    
     #print(rates/count)
     plt.title('Classifier 2 (control)')
     plt.plot(losses)
     plt.show()  
    
     return 0
  


run2()