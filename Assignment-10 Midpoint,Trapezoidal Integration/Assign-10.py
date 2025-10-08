#Sujal Sinha,2311191
#Assignment-10
import math as m
import numpy as np
from Fun import *
def f1(x):
    return 1/x
def f2(x):
    return x*m.cos(x)
def f3(x):
    return x*np.arctan(x)
N=[4,8,15,20]
print("For midpoint")
for i in range(0,4,1):
    print("For N number ", N[i])
    m_1= midpoint(1,2,f1,N[i])
    print("For 1st function", m_1)
    m_2 = midpoint(0,np.pi/2,f2,N[i])
    print("For 2nd function", m_2)
    m_3 = midpoint(0,1,f3,N[i])
    print("For 3rd function", m_3)
#Trapezoidal

print("For Trapezoidal")
for i in range(0,4,1):
    print("For N number ", N[i])
    k_1= Trapezoidal(1,2,f1,N[i])
    print("For 1st function", k_1)
    k_2 = Trapezoidal(0,np.pi/2,f2,N[i])
    print("For 2nd function", k_2)
    k_3 = Trapezoidal(0,1,f3,N[i])
    print("For 3rd function", k_3)
"""
Function	Exact Result	Midpoint N=4	Midpoint N=8	Midpoint N=15	Midpoint N=20	Trap N=4	Trap N=8	Trap N=15	   Trap N=20
1/x	          0.69314718	0.691219891	    0.692660554	    0.693008426	    0.693069098	   0.563095238	0.629538517	 0.659516758   0.667982869
x*cos(x)	  0.570796327	0.587447917	    0.574934273	    0.571971659	    0.571457287	   0.449085235	0.536202815	 0.560422255   0.564876824
x*arctan(x)	  0.285398163	0.282046049	    0.284561019	    0.285160103	    0.28526426	   0.133595347	0.198673797	 0.236332088   0.247986644
"""
"""
For midpoint
For N number  4
For 1st function 0.6912198912198912
For 2nd function 0.5874479167573121
For 3rd function 0.2820460493571144
For N number  8
For 1st function 0.6926605540432034
For 2nd function 0.5749342733821311
For 3rd function 0.2845610193056679
For N number  15
For 1st function 0.6930084263712958
For 2nd function 0.5719716590967575
For 3rd function 0.28516010270349235
For N number  20
For 1st function 0.6930690982255869
For 2nd function 0.5714572867152204
For 3rd function 0.28526426016144524
For Trapezoidal
For N number  4
For 1st function 0.5630952380952381
For 2nd function 0.44908523487295693
For 3rd function 0.13359534651990018
For N number  8
For 1st function 0.629538517038517
For 2nd function 0.5362028149251045
For 3rd function 0.19867379680888494
For N number  15
For 1st function 0.659516758381053
For 2nd function 0.5604222549632509
For 3rd function 0.2363320879406729
For N number  20
For 1st function 0.6679828689721813
For 2nd function 0.5648768242652932
For 3rd function 0.24798664384725727
"""
