import numpy as np
import matplotlib.pyplot as plt
import random
import math
import csv
from process_data import process

def sigmoid(Z):
    return 1/(1+np.exp(-Z))


'''
def cost:
    return (T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()

def gradient_BCE(Y, T):
    
    return (T/Y) - (1-T)/(1-Y)
'''
# split into 80% train and 20% test
def run():
     x, y = process('nba_data2016-2018.csv')
     D= x.shape[1]
     M= 20

     B= np.random.rand(M)
     W1= np.random.rand(D,M)
     B2 = np.random.rand(1)
     W2 = np.random.rand(M,1)


     return 0



run()

