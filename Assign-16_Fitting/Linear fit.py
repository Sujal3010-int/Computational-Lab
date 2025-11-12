#Sujal Sinha,2311191
from Fun import *
import numpy

x=[2,3,5,8,12]
y=[10,15,25,40,60]
print("The value at 6.7 for the data provided by interpolation is:", interpolation(6.7,x,y))
"""
The value at 6.7 for the data provided by interpolation is: 33.5

"""

x= [2.5,3.5,5,6,7.5,10,12.5,15,17.5,20.5]
y=[13,11,8.5,8.2,7,6.2,5.2,4.8,4.6,4.3]
sigma=[1 for i in range(len(x))]
log_y= []
log_x=[]
for i in range(0,len(y),1):
    log_1= numpy.log10(x[i])
    log= numpy.log10(y[i])
    log_y.append(log)
    log_x.append(log_1)
print("For Exponential method")
linear_fit(x,log_y,sigma)
print("For Power Rule")
linear_fit(log_x,log_y,sigma)
"""
For Exponential method
The error in a1 and a2 are respectively: 0.3911208151382824 0.002911208151382824
The Pearson's correlation coefficient is : 0.576242688806576
For Power Rule
The error in a1 and a2 are respectively: 1.0843187787790953 1.1829249225377425
The Pearson's correlation coefficient is : 0.7750435352872259
"""