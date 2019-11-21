import numpy as np
import matplotlib.pyplot as plt
import random
import math
import csv
from process_data import process
from main import run
from main2 import run2


#Yes = run()
#Yes2= run2()

def run3(Yes, Yes2):

   #Classifier 1's incorrect outputs
   No= 100-Yes
   #Classifier 2's incorrect outputs
   No2= 100-Yes2

   Mcnemar_stat= (Yes/No2 -No/Yes2)**2 / (Yes/No2 + No/Yes2)

   # assume alpha level of .05
   return Mcnemar_stat

print(run3(Yes=run(), Yes2=run2()))

