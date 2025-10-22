#Sujal Sinha, 2311191
import math
import matplotlib.pyplot as plt
from Fun import *
def f1(x, y):
    return y - x**2

def analy1(x):
    return x**2 + 2*x + 2 - 2*math.exp(x)

def f2(x, y):
    return (x + y)**2

def analy2(x):
    return math.tan(x + math.pi/4) - x

print("EULER’S METHOD ")
print("For 1st Function")
x1,y1,z1=euler(f1, 0.0, 0.0, 2.0, 0.1, analy1)
print("For 2nd Function")
x2,y2,z2=euler(f2, 0.0, 1.0, math.pi/5, 0.1, analy2)
print("PREDICTOR–CORRECTOR ")
print("For 1st Function")
x3,y3,z3=predictor_corrector(f1, 0.0, 0.0, 2.0, 0.1, analy1)
print("For 2nd function")
x4,y4,z4=predictor_corrector(f2, 0.0, 1.0, math.pi/5, 0.1, analy2)
#Plot code for 2nd function
plt.figure(figsize=(8,4))
plt.plot(x2, z2, 'k-', label='Analytical')
plt.plot(x2, y2, 'r--', label='Forward Euler')
plt.plot(x4, y4, 'b:', label='Predictor-Corrector')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Methods on 2nd given function')
plt.legend()
plt.grid(True)
plt.show()
"""
EULER’S METHOD 
H       Euler's         Analytical
For 1st Function
0.00    0.000000        0.000000
0.10    0.000000        -0.000342
0.20    -0.001000       -0.002806
0.30    -0.005100       -0.009718
0.40    -0.014610       -0.023649
0.50    -0.032071       -0.047443
0.60    -0.060278       -0.084238
0.70    -0.102306       -0.137505
0.80    -0.161537       -0.211082
0.90    -0.241690       -0.309206
1.00    -0.346859       -0.436564
1.10    -0.481545       -0.598332
1.20    -0.650700       -0.800234
1.30    -0.859770       -1.048593
1.40    -1.114747       -1.350400
1.50    -1.422221       -1.713378
1.60    -1.789443       -2.146065
1.70    -2.224388       -2.657895
1.80    -2.735826       -3.259295
1.90    -3.333409       -3.961789
2.00    -4.027750       -4.778112
For 2nd function
0.00    1.000000        1.000000
0.10    1.100000        1.123049
0.20    1.244000        1.308498
0.30    1.452514        1.595765
0.40    1.759644        2.064963
0.50    2.226050        2.908223
0.60    2.969185        4.731855
PREDICTOR–CORRECTOR
H       Predic-Co's      Analytical
For 1st Function
0.00     0.000000        0.000000
0.10    -0.000500       -0.000342
0.20    -0.003103       -0.002806
0.30    -0.010128       -0.009718
0.40    -0.024142       -0.023649
0.50    -0.047977       -0.047443
0.60    -0.084764       -0.084238
0.70    -0.137964       -0.137505
0.80    -0.211401       -0.211082
0.90    -0.309298       -0.309206
1.00    -0.436324       -0.436564
1.10    -0.597638       -0.598332
1.20    -0.798940       -0.800234
1.30    -1.046529       -1.048593
1.40    -1.347364       -1.350400
1.50    -1.709137       -1.713378
1.60    -2.140347       -2.146065
1.70    -2.650383       -2.657895
1.80    -3.249624       -3.259295
1.90    -3.949534       -3.961789
2.00    -4.762785       -4.778112
For 2nd function
0.00    1.000000        1.000000
0.10    1.122000        1.123049
0.20    1.304905        1.308498
0.30    1.585839        1.595765
0.40    2.037784        2.064963
0.50    2.825415        2.908223
0.60    4.404946        4.731855
"""
