#Sujal Sinha,2311191
#Cholensky Decomposition
from Fun import *

a= read_matrix('asgn0_matA')
b= read_matrix('asgn0_vecC')
c=cholesky_decomposition(a)
c_t= Matrix_Multiplication.transpose(c)
print("The lower Triangular matrix is: ", c)
print("The upper Triangular matrix is: ", c_t)
print("Solution of the equation:", fwd_bck_sub(b,c,c_t)) #The discrepence in the solution is occuring due to the rounding errors.
"""Output:
The lower Triangular matrix is:  [[2.0, 0.0, 0.0, 0.0], [0.5, 1.6583123951777, 0.0, 0.0], [0.5, -0.753778361444409, 1.087114613009218, 0.0], [0.5, 0.45226701686664544, 0.08362420100070905, 1.2403473458920844]]
The upper Triangular matrix is:  [[2.0, 0.5, 0.5, 0.5], [0.0, 1.6583123951777, -0.753778361444409, 0.45226701686664544], [0.0, 0.0, 1.087114613009218, 0.08362420100070905], [0.0, 0.0, 0.0, 1.2403473458920844]]
Solution of the equation: [0.967544818005012, 0.9774985876498932, 0.5324470642591002, 0.6198750760709587]
"""