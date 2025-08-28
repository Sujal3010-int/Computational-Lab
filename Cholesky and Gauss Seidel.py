#Sujal Sinha,2311191
#Cholesky and Gauss-Seidel
from Fun import *
a= read_matrix('asgn0_matA')
b= read_matrix('asgn0_vecC')
n=len(a)
for i in range (0,n,1):
    for j in range (0,n,1):
        if a[i][j]==a[j][i]:
            pass
        else:
            print('Matrix is not symmetric')
            break
print('Matrix is symmetric. So, we can move ahead with cholesky decomposition')
c=cholesky_decomposition(a)
print('The lower triangular matrix as per Cholesky is:', c)
c_t= Matrix_Multiplication.transpose(c)
print("The solution as per Cholesky method is:", fwd_bck_sub(b,c,c_t)) #The discrepence in the solution is occuring due to the rounding errors.
#Gauss-seidel
d=gauss_seidel_nodiag(a,b)
print('The soltuion as per gauss-seidel is:', d)
"""Output:
Matrix is symmetric. So, we can move ahead with cholesky decomposition
The lower triangular matrix as per Cholesky is: [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.5, 1.9364916731037085, 0.0, 0.0, 0.0, 0.0], [0.0, -0.5163977794943222, 1.9321835661585918, 0.0, 0.0, 0.0], [-0.5, -0.12909944487358055, -0.034503277967117704, 1.9318754766140744, 0.0, 0.0], [0.0, -0.5163977794943222, -0.13801311186847082, -0.5546053999849018, 1.8457244010396843, 0.0], [0.0, 0.0, -0.5175491695067657, -0.009243423333081693, -0.5832696492049564, 1.841698654119145]]
The solution as per Cholesky method is: [2.3952732291017056, 2.851731112485338, 2.7502422748269755, 2.7293618039214844, 3.388425599805255, 3.322115807214588]
The soltuion as per gauss-seidel is: [0.9999997530614102, 0.9999997892247294, 0.9999999100460266, 0.9999998509593769, 0.9999998727858708, 0.9999999457079743]
"""