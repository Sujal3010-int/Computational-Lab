#Sujal Sinha,2311191
import numpy as np
import matplotlib.pyplot as plt
from Fun import *
#Question no.1

alpha = 0.01
Ta = 20.0
L = 10.0
T0 = 40.0
TL = 200.0
h = 0.01 

def F(x, y):
    T = y[0]
    Tp = y[1]
    return np.array([Tp, alpha*(T - Ta)])

def shoot(slope):
    x_vals, y_vals = RK4_vector(F, 0.0, [T0, slope], L, h)
    return y_vals[-1, 0]  

s1 = 0.0
s2 = 10.0
T1 = shoot(s1)
T2 = shoot(s2)

for i in range(20):
    s_new = s1 + (s2 - s1) * (TL - T1) / (T2 - T1)
    T_new = shoot(s_new)

    if abs(T_new - TL) < 1e-6:
        slope = s_new
        break

    if (T1 - TL) * (T_new - TL) < 0:
        s2, T2 = s_new, T_new
    else:
        s1, T1 = s_new, T_new


x_vals, y_vals = RK4_vector(F, 0.0, [T0, slope], L, h)
T_vals = y_vals[:, 0]

target_T = 100.0
x_100 = None

for i in range(1, len(T_vals)):
    if T_vals[i-1] < target_T <= T_vals[i]:
        x1, x2 = x_vals[i-1], x_vals[i]
        T1, T2 = T_vals[i-1], T_vals[i]
        x_100 = x1 + (target_T - T1) * (x2 - x1) / (T2 - T1)
        print("The x at which temperature is 100^oC:", x_100)
        break

plt.plot(x_vals, T_vals, label='T(x)')
plt.xlabel('x (m)')
plt.ylabel('Temperature (°C)')
plt.title("Temperature Distribution by Shooting Mehtod")
plt.grid(True)
plt.legend()
plt.show()

print(f"Final slope T'(0) = {slope:.6f}")
print(f"T(0) = {T_vals[0]:.2f} °C,  T(L) = {T_vals[-1]:.2f} °C (target {TL})")

#Question 2

L = 2.0          
dx = 0.02       
dt = 0.3 * dx**2 
r = dt / dx**2

t_final = 0.8    

x = [0.0]
while x[-1] < L:
    x.append(x[-1] + dx)
x = np.array(x)
n = len(x)

u = [0.0] * n
mid = n // 2
u[mid] = 300.0  
u = np.array(u)

n_steps = int(t_final / dt)

for step in range(n_steps):
    u_new = u.copy()
    for i in range(1, n - 1):          
        u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
    u = u_new.copy()

plt.plot(x, u, label=f'Final at t={t_final}s')
plt.xlabel('x (m)')
plt.ylabel('Temperature (°C)')
plt.title('Heat Equation')
plt.grid(True)
plt.legend()
plt.show()

