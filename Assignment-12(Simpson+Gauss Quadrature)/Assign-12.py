#Sujal Sinha,2311191
import numpy as np
from Fun import *
#Question 1
def f(x):
    return x**2/(1+x**4)
for i in range(1,16,1):
    print('For N:', i)
    print(Guass_Quadrature(i,-1,1,f))#We get the 9th deciaml accurate for N=14
#Question 2
def f1(x):
    return np.sqrt(1+x**4)
for i in range(1,20):
    print("For N:", i)
    print("By Simpson Method", Simpson(0,1,f1,i))
    print("From Gaussian Quadrature", Guass_Quadrature(i,0,1,f1))
    i=i+1
#The Simpson gives us the value for N=16
#The Gauss Quadrature gives us the value for N=7
"""
For N: 1
0.0
For N: 2
0.6
For N: 3
0.4901960784313727
For N: 4
0.4816354816354814
For N: 5
0.48843235239681676
For N: 6
0.4876044866757159
For N: 7
0.48743128787975926
For N: 8
0.48750193975790385
For N: 9
0.48749736654274756
For N: 10
0.4874948523469875
For N: 11
0.48749552176648453
For N: 12
0.4874955197503811
For N: 13
0.48749548854467284
For N: 14
0.4874954942585569
For N: 15
0.4874954947005132
For 2nd question
For N: 1
By Simpson Method 1.0895531979984592
From Gaussian Quadrature 1.0307764064044151
For N: 2
By Simpson Method 1.0895531979984592
From Gaussian Quadrature 1.0893307728183257
For N: 3
By Simpson Method 1.0894134331936443
From Gaussian Quadrature 1.0894588167747246
For N: 4
By Simpson Method 1.0894134331936443
From Gaussian Quadrature 1.0894243604064495
For N: 5
By Simpson Method 1.0894287579268167
From Gaussian Quadrature 1.089429887547431
For N: 6
By Simpson Method 1.0894287579268167
From Gaussian Quadrature 1.0894293765988807
For N: 7
By Simpson Method 1.0894292989850167
From Gaussian Quadrature 1.0894294156095166
For N: 8
By Simpson Method 1.0894292989850167
From Gaussian Quadrature 1.0894294131091893
For N: 9
By Simpson Method 1.0894293839024571
From Gaussian Quadrature 1.0894294132263513
For N: 10
By Simpson Method 1.0894293839024571
From Gaussian Quadrature 1.089429413225362
For N: 11
By Simpson Method 1.089429403535486
From Gaussian Quadrature 1.0894294132247317
For N: 12
By Simpson Method 1.089429403535486
From Gaussian Quadrature 1.0894294132248323
For N: 13
By Simpson Method 1.0894294094133299
From Gaussian Quadrature 1.0894294132248215
For N: 14
By Simpson Method 1.0894294094133299
From Gaussian Quadrature 1.0894294132248223
For N: 15
By Simpson Method 1.0894294115232381
From Gaussian Quadrature 1.0894294132248226
For N: 16
By Simpson Method 1.0894294115232381
From Gaussian Quadrature 1.0894294132248221
For N: 17
By Simpson Method 1.0894294123885135
From Gaussian Quadrature 1.0894294132248223
For N: 18
By Simpson Method 1.0894294123885135
From Gaussian Quadrature 1.0894294132248226
For N: 19
By Simpson Method 1.0894294127815232
From Gaussian Quadrature 1.0894294132248226
"""




