import random
import math
import csv
import pandas as pd

# using the same example as above
df = pd.DataFrame({'team': [ 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'  ]})
data_top = df.head()  

X['Price'] = X['Price'].str.replace('$', '')

Dict = {0: 'ATL', 1: 'BKN',  
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
        if (ball_data[x][y]=='ATL'):
            ball_data[x][y]= i
             
        ball_data[x][y] = float(ball_data[x][y])
    return 0
'''
X2 = np.zeros((N, D+catOHE))
    X2[:,0:D] = X[:,1:D+1]
'''