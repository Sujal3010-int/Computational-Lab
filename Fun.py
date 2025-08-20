import numpy as np
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





   
