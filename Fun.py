import numpy as np
import math 
class Matrix_Multiplication():
    def __init__(self):
        pass
    def read_matrix(filename):
        with open(filename,'r') as f:
            matrix=[]
            for line in f:
                row=[float(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix
    
    def mat_mult(A,B):
        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):#Ran 3 nested loop
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    def transpose(matrix):#Use a transpose to change the  dimension of C and then take the matrix multiplication defined above for the result
        return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    
    def dot(A,B):
        for i in range(len(A)):
            C= C+ A[i][0]*B[i][0]
            i=i+1
        return C

class Complex:
    def __init__(self,r,i):
        self.r=r
        self.i=i

    def display(self):
        print(self.r ," +j", self.i)

    def add(c1,c2):
        real= c1.r + c2.r
        imag= c1.i + c2.i
        z= Complex(real,imag)
        return z
    def mul(c1,c2):
        real = c1.r*c2.r - c1.i*c2.i
        imag = c1.r*c2.i + c1.i*c2.r
        z= Complex(real,imag)
        return z
    def mod(c1):
        mod= np.sqrt((c1.r)**2 +(c1.i)**2)
        return mod
    def arg_deg(c1):
        arg= np.arctan(c1.i/ c1.r)#It is in radians
        arg = arg *(180/3.1415)
        return arg
class RNG:
    def __init__(self):
        pass
    def lcg(n,a=1103515245,c=12345,m=32768):
        list=[]
        z=0.1
        for _ in range (n):
            z = (a*z +c) % m
            z= z/m
            list.append(z)
        return list
    def generate(z,a=1103515245,c=12345,m=32768):
        z = (a*z +c) % m
        return z/(m - 1)
    
class LCGrandom:
    def __init__(self, seed=1):
        self.a = 1103515245
        self.c = 12345
        self.m = 32768
        self.x = seed

    def generate(self):
        self.x = (self.a * self.x + self.c) % self.m
        return self.x / (self.m - 1)

def read_matrix(filename):
        with open(filename,'r') as f:
            matrix=[]
            for line in f:
                row=[float(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix
def gauss_jordan(matrix):
    n = len(matrix)   # number of rows

    for i in range(n):
        # Find the row with the largest value in the column
        max = i
        for k in range(i+1, n):
            val_k = matrix[k][i] if matrix[k][i] >= 0 else -matrix[k][i]
            val_max = matrix[max][i] if matrix[max][i] >= 0 else -matrix[max][i]
            if val_k > val_max:
                max = k

        # Swap the current row with the row having the largest pivot
        if i != max:
            matrix[i], matrix[max] = matrix[max], matrix[i]

        # Make the pivot element equal to 1
        pivot = matrix[i][i]
        if pivot == 0:
            print("Matrix is singular and cannot be solved")
        for j in range(i, n+1):  # The diagonal element as 1
            matrix[i][j] = matrix[i][j] / pivot

        # Eliminate all other entries in the current column
        for k in range(n):
            if k != i:
                factor = matrix[k][i]
                for j in range(i, n+1):
                    matrix[k][j] = matrix[k][j] - factor * matrix[i][j]

    # The solution will be in the last column
    solution = []
    for i in range(n):
        solution.append(matrix[i][n])
    return solution

def LUdecomp(A):#For a output in the same matrix
    n= len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        
        for k in range(i, n):
            s_val = 0
            for j in range(i):
                s_val += L[i][j] * U[j][k]
            U[i][k] = A[i][k] - s_val

        
        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0  # Diagonal elements of L are 1 as per doolittle method
            else:
                s_val = 0
                for j in range(i):
                    s_val += L[k][j] * U[j][i]
                L[k][i] = (A[k][i] - s_val) / U[i][i]
    for i in range(0,n,1):
        for j in range(0,n,1):
            if i<j:
                A[i][j]=U[i][j]
            else:
                A[i][j]=L[i][j]

    return A
def Upper_Lower_Matrix(A):
    n= len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]


    for i in range(n):
        
        for k in range(i, n):
            s_val = 0
            for j in range(i):
                s_val += L[i][j] * U[j][k]
            U[i][k] = int(A[i][k] - s_val)

        
        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0  # Diagonal elements of L are 1 as per doolittle method
            else:
                s_val = 0
                for j in range(i):
                    s_val += L[k][j] * U[j][i]
                L[k][i] = int((A[k][i] - s_val) / U[i][i])
    return U,L
def jacobi_iterative(A, b, precision):
   
    n = len(A)
    x_k = [0.0] * n  # Initial guess x^(0) = (0, 0, 0, 0)
    x_n = [0.0] * n
    iteration_count = 0

    while True:
        iteration_count += 1
        for i in range(n):
            sum_val = 0.0
            for j in range(n):
                if i != j:
                    sum_val += A[i][j] * x_k[j]
            if A[i][i] == 0:
                print("Main diagonal element is zero. So the division with it will be not allowed")
                return None, 0
            x_n[i] = (b[i][0] - sum_val) / A[i][i]
            
        diff_norm = 0.0
        for i in range(n):
            diff_norm += (x_n[i] - x_k[i])**2
        
        if math.sqrt(diff_norm) < precision:
            break
        
        x_k = list(x_n)
        
    return x_n, iteration_count

def cholesky_decomposition(A):
    n = len(A)
    L = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):  # only lower triangular part
            s = 0.0
            for k in range(j):
                s += L[i][k] * L[j][k]

            if i == j:  
                val = A[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = val ** 0.5
            else:  
                L[i][j] = (A[i][j] - s) / L[j][j]

    return L
def fwd_bck_sub(b,c,c_t):
    n=len(c)
    y = [0.0] * n
    for i in range(n):
        s = 0
        for j in range(i):
            s += c[i][j] * y[j]
        y[i] = b[i][0] - s
    # Backward substitution 
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += c_t[i][j] * x[j]
        x[i] = (y[i] - s) / c_t[i][i]
    return x







   
