#Sujal Sinha,2311191
#Assignment-9 Laguerre
import math

def laguerre(co, x0, tol=1e-6,kel=1e-6, max_iter=50):
    n = len(co) - 1

    def poly(x):
        v = co[0]
        for c in co[1:]:
            v = v * x + c
        return v

    def dpoly(x):
        v = co[0] * n
        for i in range(1, len(co)-1):
            v = v * x + co[i] * (n - i)
        return v

    def ddpoly(x):
        v = co[0] * n * (n - 1)
        for i in range(1, len(co)-2):
            v = v * x + co[i] * (n - i) * (n - i - 1)
        return v

    x = x0
    for j in range(max_iter):
        p, dp, ddp = poly(x), dpoly(x), ddpoly(x)
        if abs(p) < tol:
            return x
        G = dp / p
        H = G * G - ddp / p
        sqr = math.sqrt((n - 1) * (n * H - G * G))
        if abs(G + sqr) > abs(G - sqr):
            a = n / (G + sqr)
        else:
            a = n / (G - sqr)
        x_new = x - a
        if abs(x_new - x) < tol and poly(x)<kel:
            return x
        x = x_new
        j=j+1
    return x  

def deflate(co, root):
    n = len(co)
    new_coeffs = [co[0]]   
    for i in range(1, n-1):
        value = co[i] + new_coeffs[-1] * root
        new_coeffs.append(value)
    return new_coeffs


def find_all_roots(co):
    roots = []
    c = co[:]
    n = len(c) - 1
    g = [0.0] * n  
    for m in g:
        r = laguerre(c, m)
        roots.append(r)
        c = deflate(c, r)
    return roots

P1 = [1, -1, -7, 1, 6]  # x^4 - x^3 - 7x^2 + x + 6
P2 = [1, 0, -5, 0, 4]   # x^4 - 5x^2 + 4
P3 = [2, 0, -19.5, 0.5, 13.5, -4.5]  # 2x^5 - 19.5x^3 + 0.5x^2 + 13.5x - 4.5

print("Roots of Polynomail 1:", find_all_roots(P1))
print("Roots of Polynomial 2:", find_all_roots(P2))
print("Roots of Polynomail 3:", find_all_roots(P3))
"""
Output:
Roots of Polynomail 1: [-1.0, 0.9999999999929006, -1.9999999999971605, 3.0000000000042597]
Roots of Polynomial 2: [1.0, -0.999999996112025, 1.9999999990280064, -2.0000000029159812]
Roots of Polynomail 3: [0.4999218679385703, 0.5000781286894971, -0.9999999955487126, 2.999999999465806, -3.000000000545161]
"""