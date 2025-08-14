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
    
class Lcg_random():
    def _init_(self):
        self.a = 1103515245
        self.m= 32768
        self.c = 12345
        self.x= 1

    def gen(self):
        self.x=((1103515245 * self.x + 12345)) % 32768
        return self.x/(32768 - 1)

def read_matrix(filename):
        with open(filename,'r') as f:
            matrix=[]
            for line in f:
                row=[float(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix








   
