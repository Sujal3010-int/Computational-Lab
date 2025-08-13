#Sujal Sinha, 2311191
import matplotlib.pyplot as plt

# Parameters needed to be set
n = 1000
x0 = 0.1
c_values = [3.1, 3.5, 3.8, 4.0, 4.1]  # chosen c values
k_values = [5, 10, 15, 20]

# Function to generate iterative sequence
def iterative(c,n):
        x=0.1
        list=[]
        for _ in range (n):
            x = (c*x)*(1-x)
            list.append(x)
        return list

# Plotting xi vs xi+k for various c and k
fig, axes = plt.subplots(len(c_values), len(k_values), figsize=(15, 10), sharex=False, sharey=False)

for ci, c in enumerate(c_values):
    x = iterative(c, 200)
    for ki, k in enumerate(k_values):
        ax = axes[ci, ki]
        ax.scatter(x[:-k], x[k:], s=1, alpha=0.5)
        ax.set_title(f"c={c}, k={k}")
        ax.set_xlabel(r"$x_i$")
        ax.set_ylabel(r"$x_{i+k}$")

plt.tight_layout()
plt.show()
