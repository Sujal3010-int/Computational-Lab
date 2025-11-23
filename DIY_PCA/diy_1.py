from Fun import *
import math
import matplotlib.pyplot as plt

def jacobi_eigen(A):
    n = len(A)
    
    M = []
    for i in range(n):
        M.append(A[i][:])   

    V = [[1 if i==j else 0 for j in range(n)] for i in range(n)]  

    for _ in range(100):
        p, q = 0, 1
        max_val = 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(M[i][j]) > max_val:
                    max_val = abs(M[i][j])
                    p, q = i, j

        if max_val < 1e-10:
            break

        if M[p][p] == M[q][q]:
            angle = math.pi / 4
        else:
            angle = 0.5 * math.atan(2*M[p][q] / (M[p][p] - M[q][q]))

        c = math.cos(angle)
        s = math.sin(angle)

        Mp = [row[:] for row in M]
        for i in range(n):
            M[i][p] = c*Mp[i][p] - s*Mp[i][q]
            M[i][q] = s*Mp[i][p] + c*Mp[i][q]

        for j in range(n):
            M[p][j] = M[j][p]
            M[q][j] = M[j][q]

        M[p][q] = M[q][p] = 0

        Vp = [row[:] for row in V]
        for i in range(n):
            V[i][p] = c*Vp[i][p] - s*Vp[i][q]
            V[i][q] = s*Vp[i][p] + c*Vp[i][q]

    eigenvalues = [M[i][i] for i in range(n)]
    return eigenvalues, V


def SVD(A):
    At = Matrix_Multiplication.transpose(A)
    ATA = Matrix_Multiplication.mat_mult(At, A)
    eigenvalues, V = jacobi_eigen(ATA)
    S = [math.sqrt(abs(ev)) for ev in eigenvalues]
    return S, V


def PCA_using_SVD(X):
    rows = len(X)
    cols = len(X[0])

    means = []
    for j in range(cols):
        s = sum(X[i][j] for i in range(rows))
        means.append(s / rows)

    Xc = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(X[i][j] - means[j])
        Xc.append(row)

    S, V = SVD(Xc)

    variances = [s*s for s in S]
    principal_components = V

    return variances, principal_components, means, Xc


data = [
    [1, 3],
    [4, 6],
    [6, 9],
    [7, 10]
]

variances, components, means, Xc = PCA_using_SVD(data)

print("COLUMN MEANS:")
print(means)

print("\nVARIANCES:")
print(variances)

print("\nPRINCIPAL COMPONENTS:")
for row in components:
    print(row)

X = np.array(data)
Xc = np.array(Xc)
V = np.array(components)

PC1 = V[:,0]
PC2 = V[:,1]

scale = 1.5


plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], color="orange")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("Before Centering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(Xc[:,0], Xc[:,1], color="blue")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("After Centering")
plt.xlabel("X1 (centered)")
plt.ylabel("X2 (centered)")
plt.grid(True)
plt.show()


plt.figure(figsize=(7,7))

plt.scatter(Xc[:,0], Xc[:,1], color="green")

plt.arrow(0, 0, PC1[0]*scale, PC1[1]*scale,
          head_width=0.1, color="red", length_includes_head=True, label="PC1")

plt.arrow(0, 0, PC2[0]*scale, PC2[1]*scale,
          head_width=0.1, color="purple", length_includes_head=True, label="PC2")

plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

plt.title("PCA Directions (PC1 & PC2)")
plt.xlabel("X1 (centered)")
plt.ylabel("X2 (centered)")
plt.grid(True)
plt.show()
