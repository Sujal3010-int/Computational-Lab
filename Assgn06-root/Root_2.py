#Sujal Sinha,2311191
#Root_2_Bracketing
import math as m
def f(x):
    return -x- m.cos(x)

def Interval(a,b,max_iter):
    for i in range(max_iter):
        beta = 0.5
        if f(a)*f(b)>0:
            if abs(f(a))< abs(f(b)):
                a = a - beta*(b-a)
            if abs(f(a))> abs(f(b)):
                b = b - beta*(b-a)
        if f(a)*f(b)<0:#Done
            break
    return a,b
a,b=Interval(2,4,50)
print("The bracketing interval with possible root is :[", a ,"," , b, "]")
"""Output:
The bracketing interval with possible root is :[ -2.75 , 0.625 ]
"""