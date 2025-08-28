#Sujal Sinha,2311191
# Assignment5
from Fun import *
a= read_matrix('asgn0_matA')
b= read_matrix('asgn0_vecC')
print(' Solution according to jacobi method', jacobi(a,b,1000))
print('Solution according to Gauss method', gauss_seidel_nodiag(a,b))
"""Output:
Solution according to jacobi method [2.5608772550406793, -3.124867350548284, 13.12373540856031, -1.2990449239476476, 1.2519985850725148]
Solution according to Gauss method [2.560876495148605, -3.1248664748088935, 13.123735229777056, -1.2990444474269947, 1.251998268942399]
"""