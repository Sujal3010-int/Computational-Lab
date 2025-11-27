import numpy as np
import matplotlib.pyplot as plt
import math
class Matrix_Multiplication:
    def __init__(self):
        pass

    @staticmethod
    def read_matrix(filename):
        with open(filename, 'r') as f:
            matrix = []
            for line in f:
                row = [float(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix

    @staticmethod
    def mat_mult(A, B):
        if len(A) == 0 or len(B) == 0:
            raise ValueError("Empty matrices are not allowed")

        if len(A[0]) != len(B):
            raise ValueError("Incompatible dimensions for matrix multiplication")

        result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                s = 0.0
                for k in range(len(B)):
                    s += A[i][k] * B[k][j]
                result[i][j] = s
        return result

    @staticmethod
    def transpose(matrix):
        if not matrix:
            return []
        return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    @staticmethod
    def dot(A, B):
        if len(A) != len(B):
            raise ValueError("Vectors must have the same length")
        C = 0.0
        for i in range(len(A)):
            C += A[i][0] * B[i][0]
        return C


class Complex:
    def __init__(self, r, i):
        self.r = r
        self.i = i

    def display(self):
        print(self.r, "+j", self.i)

    @staticmethod
    def add(c1, c2):
        real = c1.r + c2.r
        imag = c1.i + c2.i
        return Complex(real, imag)

    @staticmethod
    def mul(c1, c2):
        real = c1.r * c2.r - c1.i * c2.i
        imag = c1.r * c2.i + c1.i * c2.r
        return Complex(real, imag)

    @staticmethod
    def mod(c1):
        return np.sqrt(c1.r**2 + c1.i**2)

    @staticmethod
    def arg_deg(c1):
        arg = np.arctan2(c1.i, c1.r)  # radians
        return np.degrees(arg)


class RNG:
    def __init__(self):
        pass

    @staticmethod
    def lcg(n, a=1103515245, c=12345, m=32768, seed=0.1):
        nums = []
        z = seed
        for _ in range(n):
            z = (a * z + c) % m
            nums.append(z / m)
        return nums

    @staticmethod
    def generate(z, a=1103515245, c=12345, m=32768):
        z = (a * z + c) % m
        return z / (m - 1)


class Plots:
    def __init__(self):
        pass

    def plot(self, x, y, title, xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(x, y, marker='o')
        plt.grid(True)
        plt.show()
        # plt.savefig(f"{title}.png")

    def line_plot(self, x, y, title, xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y)
        plt.grid(True)
        plt.show()
        # plt.savefig(f"{title}.png")

    def hist(self, data, title, xlabel, ylabel, bins):
        plt.hist(data, bins=bins, edgecolor='black')
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
    with open(filename, 'r') as f:
        matrix = []
        for line in f:
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return matrix


def gauss_jordan(matrix):
    n = len(matrix)   

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            val_k = abs(matrix[k][i])
            val_max = abs(matrix[max_row][i])
            if val_k > val_max:
                max_row = k

        if i != max_row:
            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]

        pivot = matrix[i][i]
        if pivot == 0:
            raise ValueError("Matrix is singular and cannot be solved")

        for j in range(i, n + 1):  
            matrix[i][j] = matrix[i][j] / pivot

        for k in range(n):
            if k != i:
                factor = matrix[k][i]
                for j in range(i, n + 1):
                    matrix[k][j] = matrix[k][j] - factor * matrix[i][j]

    solution = [matrix[i][n] for i in range(n)]
    return solution

def inverse_gauss_jordan(A): #inverse using Gauss Jordan
    n = len(A)
    M = [row[:] for row in A]

    aug = [M[i] + [1 if i == j else 0 for j in range(n)] for i in range(n)]

    for i in range(n):

        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k

        aug[i], aug[max_row] = aug[max_row], aug[i]

        pivot = aug[i][i]
        if pivot == 0:
            raise ValueError("Matrix is singular and cannot be inverted")

        for j in range(2 * n):
            aug[i][j] /= pivot

        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]

    inverse = [row[n:] for row in aug]
    return inverse

def LUdecomp(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for k in range(i, n):
            s_val = 0.0
            for j in range(i):
                s_val += L[i][j] * U[j][k]
            U[i][k] = A[i][k] - s_val

        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                s_val = 0.0
                for j in range(i):
                    s_val += L[k][j] * U[j][i]
                L[k][i] = (A[k][i] - s_val) / U[i][i]

    for i in range(n):
        for j in range(n):
            if i < j:
                A[i][j] = U[i][j]
            else:
                A[i][j] = L[i][j]

    return A


def Upper_Lower_Matrix(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for k in range(i, n):
            s_val = 0.0
            for j in range(i):
                s_val += L[i][j] * U[j][k]
            U[i][k] = A[i][k] - s_val

        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                s_val = 0.0
                for j in range(i):
                    s_val += L[k][j] * U[j][i]
                L[k][i] = (A[k][i] - s_val) / U[i][i]

    return U, L


def jacobi(A, B, itr):
    n = len(A)
    x = [0.0 for _ in range(n)]
    for _ in range(itr):
        xnew = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i != j:
                    s += A[i][j] * x[j]
            xnew[i] = (B[i][0] - s) / A[i][i]
        x = xnew
    return x


def cholesky_decomposition(A):
    n = len(A)
    L = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):  
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

def cholesky_complex(A):
    n = len(A)
    L = [[0+0j for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):

            s = 0+0j
            for k in range(j):
                s += L[i][k] * L[j][k].conjugate()

            if i == j:
                val = A[i][i] - s
                if val.real <= 0 or abs(val.imag) > 1e-12:
                    raise ValueError("Matrix is not Hermitian positive definite.")
                L[i][j] = (val)**0.5  
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]

    return L

def fwd_bck_sub(b, c, c_t):
    n = len(c)
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += c[i][j] * y[j]
        y[i] = (b[i][0] - s) / c[i][i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += c_t[i][j] * x[j]
        x[i] = (y[i] - s) / c_t[i][i]
    return x


def Jacobi_iterative(A, b):
    e = 10 ** (-6)  
    n = len(A)
    x = [0.0 for _ in range(2 * n)]
    for p in range(1000):
        m = 0.0
        for i in range(n):
            sum1 = 0.0
            for j in range(n):
                if i != j:
                    sum1 += A[i][j] * x[j]
            x[i + n] = (b[i][0] - sum1) / A[i][i]

        for i in range(n):
            m += (x[i + n] - x[i]) ** 2

        if math.sqrt(m) < e:
            print("No. of iterations for convergence", p)
            break

        for i in range(n):
            x[i], x[i + n] = x[i + n], x[i]

    return x[n:]

def jacobi_solve(A, B, tol=1e-6, max_iter=1000):#Solve by making a matrix with n dimension and all coefficients in A and after = in B
    n = len(A)
    x = [0.0] * n         
    x_new = [0.0] * n

    for it in range(max_iter):
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i != j:
                    s += A[i][j] * x[j]
            x_new[i] = (B[i][0] - s) / A[i][i]

        err = max(abs(x_new[i] - x[i]) for i in range(n))

        for i in range(n):
            x[i] = x_new[i]

        if err < tol:
            return x, it + 1

    return x, max_iter


def gauss_seidel_nodiag(A, B):
    n = len(A)
    x = [0.0] * n
    for itr in range(100):
        y = x.copy()
        for i in range(n):
            s = 0.0
            for j in range(i):
                s += A[i][j] * y[j]
            a = 0.0
            for j in range(i + 1, n):
                a += A[i][j] * x[j]
            y[i] = (B[i][0] - s - a) / A[i][i]

        e = 0.0
        for i in range(n):
            if abs(y[i] - x[i]) > e:
                e = abs(y[i] - x[i])
        if e < 1e-6:
            return y
        x = y
    return x


def inverse_matrix(A):
    n = len(A)
    M = [row[:] for row in A]  
    LU = LUdecomp(M)

    inv = []
    for col in range(n):
        e = [0.0] * n
        e[col] = 1.0

        y = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(i):
                s += LU[i][j] * y[j]
            y[i] = e[i] - s

        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = 0.0
            for j in range(i + 1, n):
                s += LU[i][j] * x[j]
            x[i] = (y[i] - s) / LU[i][i]

        inv.append(x)

    return [list(col) for col in zip(*inv)]


def Bracketing_Interval(f, a, b, max_iter):
    for _ in range(max_iter):
        beta = 0.5
        if f(a) * f(b) > 0:
            if abs(f(a)) < abs(f(b)):
                a = a - beta * (b - a)
            elif abs(f(a)) > abs(f(b)):
                b = b - beta * (b - a)
        if f(a) * f(b) < 0:
            break
    return a, b


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


def newton_raphson(f, f_prime, x0, tol=1e-4, max_iter=1000):
    x = x0
    iters = 0
    while iters < max_iter:
        fx = f(x)
        if abs(fx) < tol:
            break
        fpx = f_prime(x)
        if fpx == 0:
            raise ZeroDivisionError("Derivative is zero; Newton-Raphson fails.")
        x = x - fx / fpx
        iters += 1
    return x


def Fixed_point(f1, f2, f3, rad, x1, x2, x3, max_iter, tol=10 ** (-6)):
    for i in range(max_iter):
        xa = f1(x1, x2, x3)
        xb = f2(x1, x2, x3)
        xc = f3(x1, x2, x3)

        xa1 = abs(xa - x1)
        xb2 = abs(xb - x2)
        xc3 = abs(xc - x3)

        z = rad(xa1, xb2, xc3)
        y = rad(xa, xb, xc)

        x1, x2, x3 = xa, xb, xc

        if y != 0 and z / y < tol:
            break
    return xa, xb, xc, i + 1


def Trapezoidal(a, b, f, N):
    h = (b - a) / N
    s = 0.5 * (f(a) + f(b))
    for i in range(1, N):
        s += f(a + i * h)
    return h * s


def midpoint(a, b, f, N):
    h = (b - a) / N
    x = []
    for i in range(0, N, 1):
        j = ((a + i * h) + (a + (i + 1) * h)) / 2
        x.append(j)
    s = 0.0
    for i in range(0, N, 1):
        s += h * f(x[i])
    return s


def Simpson(a, b, f, N):
    if N % 2 == 1:
        N = N + 1
    h = (b - a) / N
    s = f(a) + f(b)
    for i in range(1, N):
        if i % 2 == 0:
            s += 2 * f(a + i * h)
        else:
            s += 4 * f(a + i * h)
    return (h / 3) * s


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
    var = s2 / N - mean ** 2
    F = (b - a) * mean
    return F, var


def Guass_Quadrature(n, a, b, f):
    x, w = np.polynomial.legendre.leggauss(n)
    tot = 0.0
    z = (b - a) / 2.0
    r = (b + a) / 2.0
    for i in range(len(x)):
        m = x[i] * z + r
        fm = f(m)
        weight = w[i]
        tot += fm * weight
    return z * tot


def euler(f, x0, y0, x_end, h, analytical):
    n = int((x_end - x0) / h)
    x = x0
    y = y0
    xs = []
    ys = []
    zs = []
    for _ in range(n + 1):
        print(f"{x:.2f}\t{y:.6f}\t{analytical(x):.6f}")
        xs.append(x)
        ys.append(y)
        zs.append(analytical(x))
        y = y + h * f(x, y)
        x = x + h
    return xs, ys, zs


def predictor_corrector(f, x0, y0, x_end, h, analytical):
    n = int((x_end - x0) / h)
    x = x0
    y = y0
    xs = []
    ys = []
    zs = []
    for _ in range(n + 1):
        print(f"{x:.2f}\t{y:.6f}\t{analytical(x):.6f}")
        xs.append(x)
        ys.append(y)
        zs.append(analytical(x))
        y_pred = y + h * f(x, y)
        y = y + (h / 2.0) * (f(x, y) + f(x + h, y_pred))
        x = x + h
    return xs, ys, zs


def RK4(f, x0, y0, x_end, h):
    x = x0
    y = y0
    xs = [x0]
    ys = [y0]
    while x < x_end - 1e-12:
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        x = x + h
        xs.append(x)
        ys.append(y)
    return xs, ys


def RK4_vector(F, t0, y0_vector, t_end, h):
    t = t0
    y = np.array(y0_vector, dtype=float)

    t_values = [t0]
    y_values = [y.copy()]

    while t < t_end:
        h_actual = min(h, t_end - t)

        k1 = h_actual * F(t, y)
        k2 = h_actual * F(t + h_actual / 2, y + k1 / 2)
        k3 = h_actual * F(t + h_actual / 2, y + k2 / 2)
        k4 = h_actual * F(t + h_actual, y + k3)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t = t + h_actual

        t_values.append(t)
        y_values.append(y.copy())

        if h_actual != h:
            break

    return np.array(t_values), np.array(y_values)


def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def RK4_step(f, y, t, h):
    k1 = h * f(y, t)
    k2 = h * f(y + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(y + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(y + k3, t + h)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def linear_fit(x, y, sigma):
    n = len(x)
    S_y = 0.0
    S = 0.0
    S_x = 0.0
    S_xy = 0.0
    S_xx = 0.0
    S_yy = 0.0
    for i in range(n):
        w = 1.0 / (sigma[i] * sigma[i])
        S_y += y[i] * w
        S += w
        S_x += x[i] * w
        S_xy += x[i] * y[i] * w
        S_xx += x[i] * x[i] * w
        S_yy += y[i] * y[i] * w
    delta = (S * S_xx) - (S_x * S_x)

    a_1 = (S_xx * S_y - S_x * S_xy) / delta
    a_2 = (S_xy * S - S_x * S_y) / delta

    sig_a1_sqr = S_xx / delta
    sig_a2_sqr = S / delta
    r_sqr = (S_xy * S_xy) / (S_xx * S_yy)

    print("The error in a1 and a2 are respectively:", sig_a1_sqr, sig_a2_sqr)
    print("The Pearson's correlation coefficient is :", r_sqr)

    return a_1, a_2, sig_a1_sqr, sig_a2_sqr, r_sqr


def interpolation(x, x1, y1):
    n = len(x1)
    total = 0.0
    for i in range(n):
        prod = 1.0
        for k in range(n):
            if i != k:
                prod *= (x - x1[k]) / (x1[i] - x1[k])
        total += prod * y1[i]
    return total

def poly_fit(x, y, deg):
    n = deg + 1
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    B = [0.0 for _ in range(n)]

    for i in range(n):
        for k in range(n):
            p = 0.0
            for j in range(len(x)):
                p += x[j] ** (i + k)
            A[i][k] = p

        q = 0.0
        for j in range(len(x)):
            q += y[j] * (x[j] ** i)
        B[i] = q

    augmented = [A[i] + [B[i]] for i in range(n)]

    a = gauss_jordan(augmented)

    return a
def heat(self,nt,nx,x0,X,t0,T,Yx0):
        dx = (X-x0)/(nx-1)
        dt = (T-t0)/(nt-1)
        b= dt/(dx)**2
        V0 = []
        T0 = []
        X0 = []
        for i in range(nx):
            X0.append(x0 + i*dx)
        for t in range(nt):
            T0.append(t0 + t*dt)
        for i in range(nx):
            V0.append([Yx0(x0+i*dx)])
        A = [[0 for i in range(nx)] for i in range(nx)]  
        res = [V0]
        for i in range(nx):
            for j in range(nx):
                if  i==j :
                    A[i][j] = 1 - 2*b
                if i == j-1 or i == j+1:
                    A[i][j] = b
        for k in range(0,nt):
            r = Matrix_Multiplication.mat_mult(A,res[k])
            res.append(r)
        return res, T0, X0
