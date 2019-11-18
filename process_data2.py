import random
import math
import csv
import pandas as pd
import numpy as np

# using the same example as above
df = pd.DataFrame({'team': [ 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'  ]})
data_top = df.head()  

#X['Price'] = X['Price'].str.replace('$', '')

Dict = {0: 'ATL', 1: 'BRK',  
        2:'BOS', 3: 'CHO' , 4: 'CHI' , 5: 'CLE' , 6: 'DAL', 7: 'DEN' , 8: 'DET' , 9: 'GSW', 10: 'HOU', 11: 'IND',12: 'LAC' , 13: 'LAL' , 14: 'MEM' , 15: 'MIA', 
        16: 'MIL', 17: 'MIN', 18: 'NOP', 19: 'NYK', 
        20: 'OKC', 21: 'ORL',
        22:'PHI', 23: 'PHO' ,24: 'POR' ,
        25: 'SAC',26: 'SAS',27: 'TOR' ,28: 'UTA',29: 'WAS'} 
    


def process(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        ball_data = list(lines)

    for x in range(len(ball_data)):
      for y in range(5):
       for i in range(30):
        if (ball_data[x][y]==Dict[i]):
            ball_data[x][y]= i
            ball_data[x][y] = float(ball_data[x][y])

        if(ball_data[x][y]=="W"):
           ball_data[x][y]= 1
        if(ball_data[x][y]=="L"):
           ball_data[x][y]= 0
        if(y>1 and x!=0):
          ball_data[x][y] = float(ball_data[x][y])

    y= np.delete(ball_data, 3, axis=1)
    z = np.delete(y, 0, axis=0)
    z2= np.delete(z, 1, axis=1)

    z1= np.array(z, dtype= np.float)    
    z3 = np.array(z2 , dtype= np.float)

    categories = np.unique(z3[:,0])

    z3[:,1]= (z3[:,1]-np.mean(z3[:,1]))/np.std(z3[:,1])
    z3[:,2]= (z3[:,2]-np.mean(z3[:,2]))/np.std(z3[:,2])

    D= 2
    N = z3.shape[0]
    catOHE = categories.shape[0]
    #print(N)
    X2 = np.zeros((N, D+catOHE))
    X2[:,0:D] = z3[:,1:D+1]
    for n in range(N):
        t = categories.tolist().index(z3[n,0])
        X2[n,t+D] = 1

    

    return 0
'''
X2 = np.zeros((N, D+catOHE))
    X2[:,0:D] = X[:,1:D+1]



    true_ball_data = []       
    Y= []
    for a in range(len(z3)):
        true_ball_data.append(X2[a])
        Y.append(z1[a][1])
    
    true_ball_data= np.array(true_ball_data)
    Y= np.array(Y)
    
    #print(true_ball_data)
    return true_ball_data, Y

'''
process('nba_data_2016-2018_control_real.csv')