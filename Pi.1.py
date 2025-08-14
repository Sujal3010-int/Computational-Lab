from Fun import LCGrandom
import numpy as np
import matplotlib.pyplot as plt
s=LCGrandom()

# define x and y , as a python list
def Plot(x, y, title='Sample Plot', xlabel='X-axis Label', ylabel='Y-axis Label', file_name='sample_plot.png'):
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()
lsti=[]
lstpi=[]
for i in range(20,20020,20):
    ntotal=0
    n1=0
    lsti.append(i)
    while ntotal < i:
        xt=s.generate()
        yt=s.generate()
        r=np.sqrt(xt**2 +yt**2)
        if r < 1:
            n1= n1 + 1
        ntotal = ntotal +1

    tpi =4*n1/ntotal
    lstpi.append(tpi)
avg=0
sum=0
for i in range(len(lstpi)):
    sum= sum + lstpi[i]
    i=i+1
avg= sum/len(lstpi)
print("The average pi value is:", avg)
Plot(lsti,lstpi)
