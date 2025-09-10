#Sujal Sinha,2311191
#Assignment 8 by Fixed Point Method
from Fun import *
import math as m
def f1(x_1,x_2,x_3):# x1 =sqrt(37-x2)
    return m.sqrt(37-x_2)
def f2(x_1,x_2,x_3):#x2=sqrt(x1-5)
    val= m.sqrt(x_1-5)
    return m.sqrt(val) 
def f3(x_1,x_2,x_3):#x3= 3-x1-x2
    return 3-x_1-x_2
def rad(x1,x2,x3):
    return m.sqrt((x1)**2+(x2)**2+(x3)**2)

#Fixed Point
def Fixed_point(x1,x2,x3,max,tol=10**(-6)):#x1,x2,x3 are inital guesses
    for i in range (max):
        xa=f1(x1,x2,x3)
        xb=f2(x1,x2,x3)
        xc=f3(x1,x2,x3)
        xa1=abs(xa-x1)
        xb2=abs(xb-x2)
        xc3=abs(xc-x3)
        z= rad(xa1,xb2,xc3)
        y=rad(xa,xb,xc)
        x1,x2,x3=xa,xb,xc
        if z/y < tol:
            break
    return xa,xb,xc,i+1
A1,A2,A3,I=Fixed_point(6,0.5,-4.5,100)
print("For Fixed Point Method:\n")
print("The value of x1:",A1)
print("The value of x2:",A2)
print("The value of x3:",A3)
print("The value of iterations required:",I)
"""Output:
For Fixed Point Method:

The value of x1: 6.0
The value of x2: 0.9999999075265302
The value of x3: -3.9999996301061724
The value of iterations required: 8
"""