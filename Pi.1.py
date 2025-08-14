from Fun import LCGrandom
import numpy as np
import matplotlib.pyplot as plt
s=Lcg_random()

# define x and y , as a python list
def Plot(x, y, title='Sample Plot', xlabel='X-axis Label', ylabel='Y-axis Label', file_name='sample_plot.png'):
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()
j=[]
pi=[]
for i in range(20,20020,20):
    ntotal=0
    n1=0
    j.append(i)
    while ntotal < i:
        xt=s.gen()
        yt=s.gen()
        r=np.sqrt(xt**2 +yt**2)
        if r < 1:
            n1= n1 + 1
        ntotal = ntotal +1

    tpi =4*n1/ntotal
    pi.append(tpi)
avg=0
sum=0
for i in range(len(pi)):
    sum= sum + pi[i]
    i=i+1
avg= sum/len(pi)
print("The average pi value is:", avg)
Plot(j,pi)

