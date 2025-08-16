from Fun import LCGrandom
import numpy as np

s = LCGrandom(seed=42)

lsti = []
lstpi = []

for i in range(20, 20020, 20):  
    ntotal = 0
    n1 = 0
    lsti.append(i)

    for _ in range(i):
        xt = s.generate()
        yt = s.generate()
        r = np.sqrt(xt * xt + yt * yt)  # distance from origin
        if r < 1:
            n1 += 1
        ntotal += 1

    tpi = 4 * n1 / ntotal
    lstpi.append(tpi)


sum_val = 0
for val in lstpi:
    sum_val += val
avg = sum_val / len(lstpi)

print("The average pi value is:", avg)
print("First 10 estimates:", lstpi[:10])
