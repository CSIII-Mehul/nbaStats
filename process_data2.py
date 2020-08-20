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

    ball_data = pd.read_csv(filename)
    ball_data = ball_data.drop(['Team'], axis=1)
    dataset = ball_data.values
    
    #dataset= np.delete(dataset, 0, axis=0)

    #print(ball_data.head())
    '''
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        ball_data = list(lines)
    ''' 
    dataset[:,0:1] = convert_win(dataset[:,0:1])
    dataset[:,1:4] = dataset[:,1:4].astype(float)

    
    true_ball_data = dataset[:,1:4] #input features
    Y= dataset[:,0:1]  

    
    true_ball_data= np.array(true_ball_data)
    Y= np.array(Y)
    

    
    return true_ball_data, Y
   
      
def convert_win(col):
    for i in range(len(col)):
        if(col[i]=="W"):
            col[i]= 1
        elif(col[i]=="L"):
            col[i]= 0

    return col.astype(float)

'''
X2 = np.zeros((N, D+catOHE))
    X2[:,0:D] = X[:,1:D+1]




    
    return 

'''

process('./nbaStats/nba_data_2016-2018_control_real.csv')