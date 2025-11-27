#Sujal Sinha,2311191
from lib1 import *
import matplotlib.pyplot as plt
#Q1
def q1():
    N = 5000
    s = 20000

    left = N
    right = 0

    left_cou = []
    times = []

    ran = RNG.lcg(s, seed=0.12345)

    for t in range(s):

        r = ran[t]
        particle = int(r * N) # We multiplied it by N as we need it in range of 0 to 5000 and r by lcg is less than 1  

        if particle < left:
            left -= 1
            right += 1
        else:
            right -= 1
            left += 1

        left_cou.append(left)
        times.append(t)

    plt.figure()
    plt.plot(times, left_cou, label="Particles on LEFT")
    plt.xlabel("Time step")
    plt.ylabel("Left-side particle count")
    plt.title("Random Exchange of Particles Between Two Halves")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nFinal left-side count:", left)
    print("Expected equilibrium: 2500")

q1()
"""
Output: 
Final left-side count: 2504
Expected equilibrium: 2500
"""
#Q2
def q2():

    A = [
        [4, -1, 0, -1, 0, 0],
        [-1, 4, -1, 0, -1, 0],
        [0, -1, 4, 0, 0, -1],
        [-1, 0, 0, 4, -1, 0],
        [0, -1, 0, -1, 4, -1],
        [0, 0, -1, 0, -1, 4],
    ]

    B = [
        [2],
        [1],
        [2],
        [2],
        [1],
        [2],
    ]

    x = gauss_seidel_nodiag(A, B)

    print("Solution by gauss seidel")
    for i, xi in enumerate(x, start=1):
        print(f"x{i} = {xi:.4f}")
q2()
"""
Output: 
Solution by gauss seidel
x1 = 1.0000
x2 = 1.0000
x3 = 1.0000
x4 = 1.0000
x5 = 1.0000
x6 = 1.0000
"""
#Q3
def q3():
    F0 = 2.5

    def f(x):
        return F0 - x * np.exp(x)

    def f_prime(x):
        #The derivative of F(x)= -(1+x)e**x
        return -(1 + x) * np.exp(x)

    x0 = 0.6 #This is our inital guess
    root = newton_raphson(f, f_prime, x0, tol=1e-6)
    print(f"Extension of spring (root of F(x)=0) = {root:.6f}")
q3()
"""
Output:
Extension of spring (root of F(x)=0) = 0.958586
"""

#Q4
def q4(): # We are using Simpson method for integration 
    #We are going to integrate and then find the com of the system as Com= Integration(a to b)(x*lambda)dx\Integration(a to b)(lambda)dx
    def lambd(x):
        return x ** 2

    def lamb1(x):
        return x ** 3  # x * λ(x)

    a= 0.0
    b= 2.0
    N = 1000  

    mass = Simpson(a, b, lambd, N)
    mom = Simpson(a, b, lamb1, N)

    x_cm = mom / mass
    print(f"Center of mass = {x_cm:.4f}")
q4()
"""
Output:
Center of mass = 1.5000
"""
# Q5
#By manual calculation we get y_max= 5m
v0 = 10.0
g = 10.0
gamma = 0.02

def F(t, Y):
    y, v = Y
    dy_dt = v
    dv_dt = -gamma * v - g
    return np.array([dy_dt, dv_dt])

t0 = 0.0
t_end = 10.0  # Any arbitrary time greater than its time of flight works here 
h = 0.001

t_values, Y_values = RK4_vector(F, t0, [0.0, v0], t_end, h)
y_values = Y_values[:, 0]
v_values = Y_values[:, 1]
a=0
max=0
for i in range(len(y_values)):
    if max < y_values[i]:
        max= y_values[i]
        a=i
    else: 
        pass
y_max = Y_values[a]

print(f"Maximum height", y_max[0])
plt.figure()
plt.plot(y_values, v_values)
plt.xlabel("Height y")
plt.ylabel("Velocity v")
plt.title("Velocity vs height with air resistance")
plt.grid(True)
plt.show()
"""
Output:
Maximum height 4.934317509223537
"""
#Q6
def Y_x(x):
     return 20*abs(np.sin(np.pi*x))

result, T0, X0 = heat(Y_x,5001,21,0,2,0,4,Y_x)
time_int = [0,10,20,50,100,200,500,999]
for i in time_int:
     plt.plot(X0,result[i],label=f'Time = {T0[i]:.2f} s')
plt.xlabel('Length of rod (x)')
plt.ylabel('Temperature (T)')
plt.legend()
plt.show()

#Q7
Z= read_matrix('Esem.txt')
xs=[]
ys=[]
for i  in range(len(Z)):
    xs.append(Z[i][0])
    ys.append(Z[i][1])
coeffs = poly_fit(xs, ys, deg=4)

print("Quartic polynomial coefficients:")
for i, c in enumerate(coeffs):
    print(f"a{i} = {c:.6f}")
"""
Output:
Quartic polynomial coefficients:
a0 = 0.254630
a1 = -1.193759
a2 = -0.457255
a3 = -0.802565
a4 = 0.013239
"""





    

