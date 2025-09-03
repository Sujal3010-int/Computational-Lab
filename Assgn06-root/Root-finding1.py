#Sujal Sinha,2311191
#Root-finding-Regula Falsi and Bisection
import numpy as np
import math as m
def f(x):
    return np.log(x/2)-m.sin(5*x/2)
#Bisection
def Bisection(a,b,epsilon,delta,max_iter):
    beta = 1.2
    if f(a)*f(b)>0:
        if abs(f(a))< abs(f(b)):
            a = a - beta*(b-a)
        if abs(f(a))> abs(f(b)):
            b = b - beta*(b-a)
    if f(a)*f(b)<0:
        if abs(b-a)<epsilon:
            if f(a)<delta or f(b)<delta:
                return a
    for i in range(max_iter):
        c= (a+b)/2
        if f(c)*f(a)<0:
            b = c
        if f(c)*f(b)<0:
            a = c
        if abs(b-a)<epsilon:
            if f(a)<delta or f(b)< delta :
                z=a
    return z

k=Bisection(1.5,3,10**(-6),10**(-6),10000)
print("The root for the equation by bisection method is:", k)
#Regula Falsi
def regula_falsi(a,b,epsilon,delta,max_iter):
    if a < b:
        pass
    if a > b:
        a,b = b,a
    if f(a)*f(b)<0:
        if abs(b-a)<epsilon:
            if (f(a) or f(b))<delta:
                return a
    for i in range(max_iter):
        c = b - (((b-a)*f(b))/(f(b)-f(a)))
        if f(a)*f(c)<0:
            if abs(b-c)<epsilon:
                return c
            b = c
            a = a
        if f(b)*f(c)<0:
            if abs(a-c)<epsilon:
                return c
            a = c
            b = b
    return None

l = regula_falsi(1.5,3,10**(-6),10**(-6),1000)
print("The root of the equation by regula falsi is ", l)

"""Output:
The root for the equation by bisection method is: 2.6231403354363083
The root of the equation by regula falsi is  2.6231403358555436
"""

