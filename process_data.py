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
        2:'BOS', 3: 'CHA' , 4: 'CHI' , 5: 'CLE' , 6: 'DAL', 7: 'DEN' , 8: 'DET' , 9: 'GSW', 10: 'HOU', 11: 'IND',12: 'LAC' , 13: 'LAL' , 14: 'MEM' , 15: 'MIA', 
        16: 'MIL', 17: 'MIN', 18: 'NOP', 19: 'NYK', 
        20: 'OKC', 21: 'ORL',
        22:'PHI', 23: 'PHX' ,24: 'POR' ,
        25: 'SAC',26: 'SAS',27: 'TOR' ,28: 'UTA',29: 'WAS'} 
    
# iterating the columns 
for row in data_top.index: 
    print(row, end = " ") 

def process(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        ball_data = list(lines)

    for x in range(len(ball_data)):
      for y in range(4):
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

    #print(ball_data)

    z = np.delete(ball_data, 0, axis=0)
    z2= np.delete(z, 1, axis=1)
    print(z2[202])

    true_ball_data = []       
    Y= []
    for a in range(len(z2)):
        true_ball_data.append(z2[a])
        Y.append(z[a][1])
    
    return true_ball_data, Y
'''
X2 = np.zeros((N, D+catOHE))
    X2[:,0:D] = X[:,1:D+1]

    z2[:,0]= (z2[:,0]-np.mean(z2[:,0]))/np.std(z2[:,0])
    z2[:,1]= (z2[:,1]-np.mean(z2[:,1]))/np.std(z2[:,1])
    z2[:,2]= (z2[:,2]-np.mean(z2[:,2]))/np.std(z2[:,2])

'''
process('nba_data2016-2018.csv')