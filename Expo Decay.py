from Fun import *
import matplotlib.pyplot as plt
import numpy as np
a= RNG.lcg(5000)
z=0
k=[]
for i in range(len(a)):
    z= float(-np.log(a[i]))
    k.append(z)
avg=0
sum=0
for i in range(1000):
    sum= sum + k[4000+i]
    i=i+1
avg= sum/1000
print("The average of the last 1000 terms are", avg)

# Plotting a basic histogram
plt.hist(k, bins=40, color='skyblue', edgecolor='black')
# Display the plot
plt.show()