#Sujal Sinha,2311191
import matplotlib.pyplot as plt
import math as m
import numpy as np
from Fun import *
def f(y,x):
    return (x+y)**2
def y(x):
    return (m.tan(x+m.pi/4)-x)


x1,y1,z1=(RK4(f,0,1,m.pi/5,0.1,y))
print("x value:", x1)
print("y value:", y1)
print("Analytical value",z1)

x2,y2,z2=(RK4(f,0,1,m.pi/5,0.25,y))
print("x value:", x2)
print("y value:", y2)
print("Analytical value",z2)

x3,y3,z3=(RK4(f,0,1,m.pi/5,0.45,y))
print("x value:", x3)
print("y value:", y3)
print("Analytical value",z3)


plt.figure(figsize=(5,4))
plt.plot(x1, y1, 'k-', label='For h=0.1')
plt.plot(x2, y2, 'r--', label='For h=0.25')
plt.plot(x1, z1, 'g', label='Analytical for h=0.1')
plt.plot(x2, z2, 'b-', label='Analytical for h=0.25')
plt.plot(x3, y3, 'p-', label='For h=0.45')
plt.plot(x3, z3, 'b:', label='Analytical for h=0.45')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of h')
plt.legend()
plt.grid(True)
plt.show()
"""
x value: [0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6, 0.7]
y value: [1, 1.1230489138367843, 1.308496167191276, 1.5957541602334353, 2.0648996869578835, 2.907820425151802, 4.7278968165905155, 10.853932075720177]
Analytical value [0.9999999999999999, 1.123048880449865, 1.3084976471214003, 1.595765122854009, 2.064962756722603, 2.9082234423358275, 4.7318552234587274, 10.981373800310225]
x value: [0, 0.25, 0.5, 0.75]
y value: [1, 1.43555804125693, 2.8972272051287176, 17.885430277813693]
Analytical value [0.9999999999999999, 1.4357964171683395, 2.9082234423358275, 27.4882528501416]
x value: [0, 0.45, 0.9]
y value: [1, 2.3890595350788653, 109.96022593101979]
Analytical value [0.9999999999999999, 2.4188840280163877, -9.587629546481706]
"""

#Question 2
mu = 0.15
k = 1.0
m = 1.0
omega_squared = k / m # omega^2 = 1.0

def F2(t, Y):
    
    x, v = Y[0], Y[1]
    
    dxdt = v
    dvdt = -omega_squared * x - mu * v
    
    return np.array([dxdt, dvdt])

def calculate_energy(X, V, k, m):
    return 0.5 * m * (V**2) + 0.5 * k * (X**2)

t0 = 0.0
t_end = 40.0
h = 0.01 
initial_state = [1.0, 0.0]
T, Y_matrix = RK4_vector(F2, t0, initial_state, t_end, h)
X = Y_matrix[:, 0]
V = Y_matrix[:, 1]
E = calculate_energy(X, V, k, m)
plt.tight_layout()
plt.show()
plt.figure(figsize=(5,4))
plt.plot(T, E, 'b:', label='Energy')
plt.plot(T, V, 'g.', label='Velcoity')
plt.plot(T, X, 'r--', label='Displacemnt')
plt.xlabel('')
plt.ylabel('Time')
plt.title('Damped Harmonic Oscillator')
plt.legend()
plt.grid(True)
plt.show()
