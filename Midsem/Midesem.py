#Sujal Sinha, 2311191
from Fun import *
import numpy as np
#Question 1
for j in range(5000,20000,1000):
    a= RNG.lcg(j)
    b= RNG.lcg(j)
    count=0
    for i in range (0,len(a)):
        z= (b[i])*(b[i])
        if ((a[i])*(a[i]))/2 + z < 1:
            count= count+1
    print("The number of points inside for",j, "total number of points:", count)
    k= count/len(a)
    Area = k*2 #Multiplied by 2 as it is the area of rectange that it is inside and Area of ellipse/ Area of rectangle= No. of point inside ellipse/ Total No. of points
    print("Area of the entire ellipse", Area*4)
"""
The number of points inside for 5000 total number of points: 4035
Area of the entire ellipse 6.456
The number of points inside for 6000 total number of points: 4856
Area of the entire ellipse 6.474666666666667
The number of points inside for 7000 total number of points: 5661
Area of the entire ellipse 6.469714285714286
The number of points inside for 8000 total number of points: 6475
Area of the entire ellipse 6.475
The number of points inside for 9000 total number of points: 7277
Area of the entire ellipse 6.468444444444445
The number of points inside for 10000 total number of points: 8082
Area of the entire ellipse 6.4656
The number of points inside for 11000 total number of points: 8893
Area of the entire ellipse 6.467636363636363
The number of points inside for 12000 total number of points: 9731
Area of the entire ellipse 6.487333333333333
The number of points inside for 13000 total number of points: 10529
Area of the entire ellipse 6.479384615384616
The number of points inside for 14000 total number of points: 11361
Area of the entire ellipse 6.492
The number of points inside for 15000 total number of points: 12183
Area of the entire ellipse 6.4976
The number of points inside for 16000 total number of points: 12974
Area of the entire ellipse 6.487
The number of points inside for 17000 total number of points: 13779
Area of the entire ellipse 6.484235294117647
The number of points inside for 18000 total number of points: 14604
Area of the entire ellipse 6.490666666666667
The number of points inside for 19000 total number of points: 15419
Area of the entire ellipse 6.492210526315789
"""
#Question 2
def f(x):
    return (x-5)*np.exp(x)+5

def f1(x):
    return np.exp(x)+(x-5)*np.exp(x)

x= newton_raphson(f,f1,5)
print(x)
#As b = K_bT thus we can directly use it in the x equation to get the value of b 
b1= (6.626*10**(-34)*3*10**(8))/(1.381*10**(-23))
b=b1/x
print(b)
"""
x= 4.96511423174643
b= 0.002899010330737035
"""
#Question 3

M= (read_matrix('asgn0_matA'))
Z= inverse_matrix(M)
for i in range (0,5,1):
    for j in range (0,5,1):
        k= Z[i][j] *1000
        r=np.floor(k)
        if k-r <0.5:
            k = r
        else:
            k= np.ceil(k)
        o= k/1000
        Z[i][j] = float(o)
print(Z)
"""
[[-122.767, -178.034, -251.529, 29.951, -6.402]
 [-24.04, -34.324, -48.909, 5.901, -1.427] 
 [1.609, 2.344, 3.026, -0.316, -0.032]
 [-3.151, -1.543, -5.227, 1.263, -1.59]
 [0.185, -0.679, -0.633, -0.166, 1.0]
"""
#Question 4
B= (read_matrix('asgn0_matB'))
C= (read_matrix("asgn0_vecC"))
print(Jacobi_iterative(B,C))
"""
[1.499999769658066, -0.5, 2.0, -2.5000002584806746, 1.0, -0.9999998602994417]
"""







