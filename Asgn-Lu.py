#Sujal Sinha, 2311191
from Fun import *
A= read_matrix('asgn0_matA')
print("The original matrix", A)
#Following Doolittle method
#l[i][i]=1
U,L= Upper_Lower_Matrix(A)
print("The upper Triangular matrix is ", U)
print("The lower Triangular matrix is ", L)
S= Matrix_Multiplication.mat_mult(L,U)
print("The multiplication of L and U yields(Verification)",S)
"""
Output:
The original matrix [[1.0, 2.0, 4.0], [3.0, 8.0, 14.0], [2.0, 6.0, 13.0]]
The upper Triangular matrix is  [[1.0, 2.0, 4.0], [0.0, 2.0, 2.0], [0.0, 0.0, 3.0]]
The lower Triangular matrix is  [[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [2.0, 1.0, 1.0]]
The multiplication of L and U yields [[1.0, 2.0, 4.0], [3.0, 8.0, 14.0], [2.0, 6.0, 13.0]]
"""

