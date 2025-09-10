#Sujal Sinha,2311191
#Assignment 8 by Newton Raphson Method
import math as m
from Fun import *
def rad(x1,x2,x3):
    return m.sqrt((x1)**2+(x2)**2+(x3)**2)

# Residual system (f = 0)
def fn(x1, x2, x3):
    r1 = x1 - m.sqrt(37 - x2)
    r2 = x2 - m.sqrt(x1 - 5)
    r3 = x1 + x2 + x3 - 3
    return [r1, r2, r3]

# Jacobian matrix
def Jacobian(x1, x2, x3):
    J = [
        [1,  1/(2*m.sqrt(37 - x2)), 0],
        [-1/(2*m.sqrt(x1 - 5)), 1, 0],
        [1, 1, 1]
    ]
    return J

# Newton–Raphson iteration using LU decomposisiton
def Newton_Raphson(x1, x2, x3, max_iter=100, tol=1e-6):
    for itr in range(0, max_iter):
        F = fn(x1, x2, x3)
        J = Jacobian(x1, x2, x3)
        U, L = Upper_Lower_Matrix(J)
        U_t = Matrix_Multiplication.transpose(U)
        n = len(U)
        b = [[F[i]] for i in range(n)]
        delt= fwd_bck_sub(b, L, U_t)        
        x1_new = x1 - delt[0]
        x2_new = x2 - delt[1]
        x3_new = x3 - delt[2]
        if rad((x1_new - x1), (x2_new - x2), (x3_new - x3))/rad(x1_new,x2_new,x3_new) < tol:
            return x1_new, x2_new, x3_new, itr
        x1, x2, x3 = x1_new, x2_new, x3_new
    return x1, x2, x3, max_iter

B1, B2, B3, iters = Newton_Raphson(6, 0.5, -4.5, 100)
print("Newton-Raphson Method:")
print("x1 =", B1)
print("x2 =", B2)
print("x3 =", B3)
print("Iterations =", iters)
"""Output:
Newton-Raphson Method:
x1 = 6.0
x2 = 0.9999985126151462
x3 = -3.999998512615146
Iterations = 7
"""
