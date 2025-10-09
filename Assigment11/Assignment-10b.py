#Sujal Sinha,2311191
from Fun import *
import numpy as n
from lib import Plots
#For 1st problem
#N=290 for midpoint and N=7 for simpson for 1/x

def f(x):
    return 1/x
g_1=midpoint(1,2,f,290)
print("The result for midpoint of 1/x", g_1)
k_1=Simpson(1,2,f,20)
print("The result for Simpson of 1/x", k_1)
def f1(x):
    return x*n.cos(x)
w_1= midpoint(0,n.pi/2,f1,610)#N=610 for xcos(x) by midpoint 
print("The result for mispoint for xcos(x)", w_1 )
l_1= Simpson(0,n.pi/2,f1,22)#N=22 for xcos(x) for Simpson1/3 method
print("The result for simpson for xcos(x)", l_1)
def f2(x):
    return (np.sin(x))**2
p,q,w=[],[],[]
for N in range(1000, 25001, 1000):
    es, sigma = MonteCarlo(-1, 1, f2, N)
    w.append(N)
    p.append(es)
    q.append(sigma)
    print(f"N={N:5d}  Estimate={es:.6f}  Sigma={sigma:.6f} ")
h=Plots.line_plot(p,w,p,"Plot of N v/s F", "N","F")
y=Plots.line_plot(q,w,q,"Plot of N v/s Sigma", "N","Sigma")
"""

The result for midpoint of 1/x 0.6931468089794606
The result for Simpson of 1/x 0.6931473746651162
The result for mispoint for xcos(x) 0.5707970370864707
The result for simpson for xcos(x) 0.570796987316687
N= 1000  Estimate=0.560924  Sigma=0.051421
N= 2000  Estimate=0.552794  Sigma=0.051115
N= 3000  Estimate=0.552370  Sigma=0.051011 
N= 4000  Estimate=0.562076  Sigma=0.051177 
N= 5000  Estimate=0.562160  Sigma=0.050988 
N= 6000  Estimate=0.562756  Sigma=0.050981 
N= 7000  Estimate=0.561821  Sigma=0.050792 
N= 8000  Estimate=0.560738  Sigma=0.051003 
N= 9000  Estimate=0.562035  Sigma=0.050919 
N=10000  Estimate=0.559326  Sigma=0.050777 
N=11000  Estimate=0.559154  Sigma=0.051001 
N=12000  Estimate=0.555925  Sigma=0.050786 
N=13000  Estimate=0.554612  Sigma=0.050672 
N=14000  Estimate=0.554722  Sigma=0.050576 
N=15000  Estimate=0.554049  Sigma=0.050475 
N=16000  Estimate=0.554972  Sigma=0.050487 
N=17000  Estimate=0.555108  Sigma=0.050358 
N=18000  Estimate=0.553868  Sigma=0.050386 
N=19000  Estimate=0.554302  Sigma=0.050464 
N=20000  Estimate=0.553299  Sigma=0.050486 
N=21000  Estimate=0.552749  Sigma=0.050450 
N=22000  Estimate=0.552311  Sigma=0.050423 
N=23000  Estimate=0.551583  Sigma=0.050423 
N=24000  Estimate=0.551824  Sigma=0.050357 
N=25000  Estimate=0.551271  Sigma=0.050287
"""
    


