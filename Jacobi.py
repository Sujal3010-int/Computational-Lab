#Sujal Sinha, 2311191
#Jordan
import math
from Fun import *

a= read_matrix('asgn0_matA')
b= read_matrix('asgn0_vecC')
n,iter=jacobi_iterative(a,b,0.000001)
n_n=[]
print("The non-rounded off result", n,"The iteration needed: ", iter)#Non-rounded off result
for i in range (0,4,1):
    s= math.floor(n[i]) + 0.5
    z=0
    if n[i] > (s):
        z = math.ceil(n[i])
    if n[i] < (s):
        z = math.floor(n[i])
    n_n.append(z)
print("The rounded off result", n_n)#Rounded off result
"""Output:
The non-rounded off result [2.980232238769531e-07, 1.0, 0.9999997019767761, 1.0000002980232239] The iteration needed:  43
The rounded off result [0, 1, 1, 1]
 """
 