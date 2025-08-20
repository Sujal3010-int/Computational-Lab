from Fun import LCGrandom
import numpy as np
# define x and y , as a python list
def Plot(x, y, title='Sample Plot', xlabel='X-axis Label', ylabel='Y-axis Label', file_name='sample_plot.png'):
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()

s = LCGrandom(seed=42)

li = []
pi = []

for i in range(20, 20020, 20):  
    ntotal = 0
    n1 = 0
    li.append(i)

    for _ in range(i):
        x = s.generate()
        y = s.generate()
        r = np.sqrt(x * x + y * y)  # distance from origin
        if r < 1:
            n1 += 1
        ntotal += 1

    t = 4 * n1 / ntotal
    pi.append(t)


sum_val = 0
for val in lstpi:
    sum_val += val
avg = sum_val / len(lstpi)

print("The average pi value is:", avg)
print("First 10 estimates:", lstpi[:10])
Plot(lsti,lstpi)

