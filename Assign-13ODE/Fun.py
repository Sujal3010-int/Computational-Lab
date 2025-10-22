import numpy as np
import matplotlib.pyplot as plt
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
    
class Plots():
    def __init__(self):
        pass

    def plot(self,x, y,title,xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(x, y, marker='o', color='b')
        plt.grid(True)
        plt.show()
        #plt.savefig(f"{title}.png")

    def line_plot(self,x, y,title,xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y)
        plt.grid(True)
        plt.show()
        #plt.savefig(f"{title}.png")

    def hist(self, data, title, xlabel, ylabel, bins):
        plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

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

def jacobi(A, B, itr):
    n = len(A)
    x=[0.0 for _ in range(itr) ]
    for _ in range(itr):
        xnew=[0]*n
        for i in range(n):
            sum = 0
            for j in range(n):
                if i!= j:
                   sum = sum +  A[i][j]*x[j]
            xnew[i] = (B[i][0] - sum)/A[i][i]
        x= xnew
    return x

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
                    print("Matrix is not positive definite")
                    break
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
        y[i] = (b[i][0] - s)/c[i][i]
    # Backward substitution 
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += c_t[i][j] * x[j]
        x[i] = (y[i] - s) / c_t[i][i]
    return x

def Jacobi_iterative(A,b):
    e = 10**(-6) # precision
    n = len(A)
    x=[0 for _ in range(2*n)]
    for p in range(1000):
        m = 0
        for i in range(n):
            sum1=0
            for j in range(n):
                if i!=j:
                    sum1 += A[i][j]*x[j]
            x[i+n] = (b[i][0] - sum1)/ A[i][i]
        for i in range(n):
            m+=(x[i+n] - x[i])**2
        if math.sqrt(m)<e:
            print("No. of iterations for convergence",p)
            break
        for i in range(n):
            x[i],x[i+n] = x[i+n],x[i]
    return x[n:]

def gauss_seidel_nodiag(A,B):
    n = len(A)
    x = [0.0] * n
    for itr in range(100):
        y = x.copy()
        for i in range(n):
            s = 0
            for j in range(i):
                s +=  A[i][j] * y[j]
            a = 0
            for j in range(i + 1, n):
                a += A[i][j] * x[j]
            y[i] = (B[i][0] - s - a) / A[i][i]

        e = 0
        for i in range(n):
            if abs(y[i] - x[i]) > e:
                e = abs(y[i] - x[i])
        if e < 10e-6:
            return y
        x = y
def inverse_matrix(A):
    
    n = 5
    M = [row[:] for row in A]
    LU = LUdecomp(M)

    inv = []
    for col in range(n):
        e = [0.0] * n
        e[col] = 1.0

        y = [0.0] * n
        for i in range(n):
            s = sum(LU[i][j] * y[j] for j in range(i))
            y[i] = e[i] - s   

        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(LU[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - s) / LU[i][i]

        inv.append(x)

    return [list(col) for col in zip(*inv)]
def Bracketing_Interval(f,a,b,max_iter):
    for i in range(max_iter):
        beta = 0.5
        if f(a)*f(b)>0:
            if abs(f(a))< abs(f(b)):
                a = a - beta*(b-a)
            if abs(f(a))> abs(f(b)):
                b = b - beta*(b-a)
        if f(a)*f(b)<0:
            break
    return a,b

def bisection(f, a, b, tol=1e-6):
    iters = 0
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iters += 1
    return (a + b) / 2, iters


def regula_falsi(f, a, b, tol=1e-6):
    iters = 0
    while True:
        fa, fb = f(a), f(b)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        
        if abs(fc) < tol:
            break
            
        if f(a) * fc < 0:
            b = c
        else:
            a = c
        iters += 1
    return c, iters

def newton_raphson(f, f_prime, x0, tol=1e-4):
    x = x0
    iters = 0
    while True:
        fx = f(x)
        if abs(fx) < tol:
            break
        x = x - fx / f_prime(x)
        iters += 1
    return x

def Fixed_point(f1,f2,f3,rad,x1,x2,x3,max,tol=10**(-6)):#x1,x2,x3 are inital guesses #rad is the modulus of the point
    for i in range (max):
        xa=f1(x1,x2,x3)
        xb=f2(x1,x2,x3)
        xc=f3(x1,x2,x3)
        xa1=abs(xa-x1)
        xb2=abs(xb-x2)
        xc3=abs(xc-x3)
        z= rad(xa1,xb2,xc3)
        y=rad(xa,xb,xc)
        x1,x2,x3=xa,xb,xc
        if z/y < tol:
            break
    return xa,xb,xc,i+1
def Trapezoidal(a,b,f,N):
    h= (b-a)/N
    x=[]
    for i in range(0,N,1):
        x1= a+ (i)*h
        x.append(x1)
    sum=0
    for i in range(N-1,0,-1):
        T= (h/2)*(f(x[i-1])+f(x[i]))
        sum = sum +T
    return sum
def midpoint(a,b,f,N):
    h= (b-a)/N
    x=[]
    for i in range(0,N,1):
        j= ((a+i*h)+(a+(i+1)*h))/2
        x.append(j)
    sum=0
    for i in range(0,N,1):
        m= h*f(x[i])
        sum = sum + m
    return sum
def Simpson(a,b,f,N):
    if N%2==1:
        N= N+1
    h= (b-a)/(N)
    sum_e= f(a)+f(b)
    s=0
    for i in range(1,N):
        if i%2==0:
            s += 2*(f(a+i*h))
        else:
            s += 4*(f(a+i*h))
    s =sum_e+s
    return (h/3)*(s)
def MonteCarlo(a, b, f, N):
    m = RNG.lcg(N)
    s = 0.0
    s2 = 0.0
    for r in m:
        x = a + (b - a) * r  
        fx = f(x)
        s += fx
        s2 += fx * fx

    mean = s / N
    var = s2 / N - mean**2
    F = (b - a) * mean
    return F, var
def Guass_Quadrature(n,a,b,f):
    x,w=np.polynomial.legendre.leggauss(n)
    tot=0
    z= (b-a)/2
    r= (b+a)/2
    for i in range(len(x)):
        m= x[i]*z + r
        fm=f(m)
        weight=w[i]
        tot+= fm*weight
    return z*tot
def euler(f, x0, y0, x_end, h, analytical):
    n = int((x_end - x0) / h)
    x = x0
    y = y0
    x1=[]
    y1=[]
    z1=[]

    for i in range(n + 1):
        print(f"{x:.2f}\t{y:.6f}\t{analytical(x):.6f}")
        y = y + h * f(x, y) 
        z=analytical(x) 
        x = x + h
        x1.append(x)
        y1.append(y)
        z1.append(z)
    return(x1,y1,z1)

def predictor_corrector(f, x0, y0, x_end, h, analytical):
    n = int((x_end - x0) / h)
    x = x0
    y = y0
    x1=[]
    y1=[]
    z1=[]


    for i in range(n + 1):
        print(f"{x:.2f}\t{y:.6f}\t{analytical(x):.6f}")
        y_pred = y + h * f(x, y)
        y = y + (h/2) * (f(x, y) + f(x + h, y_pred))
        z=analytical(x)
        x = x + h
        x1.append(x)
        y1.append(y)
        z1.append(z)
    return (x1,y1,z1)