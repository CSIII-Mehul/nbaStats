import numpy as np
import matplotlib.pyplot as plt
import random
import math
import csv
from process_data import process
from main import run
from main2 import run2
#form mlxtend.evaluate import mcnemar

#Yes = run()
#Yes2= run2()

def run3(Yes, Yes2):
    p1= Yes/100
    p2 = Yes2/100

   #not useful
   #Mcnemar_stat= (Yes/No2 -No/Yes2)**2 / (Yes/No2 + No/Yes2)

   # assume alpha level of .05
    z_statistic(p1,p2)
    
    if(conditions(Yes,Yes2) and z_statistic(p1,p2)):
      print("There is a significant statistical difference, the 3 point shot does have an effect on the number of wins")
    else:
     print("We cannot reject the null hypothesis. A higher percentage of 3s, alone, does not necessarily equate to wins.")
    
   #use 2 prop z test
    return 0

def conditions(Yes, Yes2):
   #SRS condition may not be met for the 2 prop z test
   #the sample sizes are definitiely less than 10% of the population (by the way the population is about 4664)
    boole = False
   
    if(100*Yes >= 5 and 100*(1-Yes) >= 5 and 100*Yes2 >= 5 and 100*(1-Yes2) >= 5 ):
      boole = True
    
    return boole

def z_statistic(p1,p2):
   #at .05 alpha level test statistic is 1.96
   boole = False
   p_hat= (p1*100+p2*100)/200
   z= abs((p1-p2)/(( p_hat*(1-p_hat) *(1/100+1/100))**.5))
   if (z>1.96):
    boole = True


   return boole

print(run3(Yes=run(), Yes2=run2()))

