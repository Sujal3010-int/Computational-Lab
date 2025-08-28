#Sujal Sinha,2311191
# Assignment5
from Fun import *
#a and b matrices have been swithced when entering the values with row1 to 4 and 3 to 5 
a= read_matrix('asgn0_matA')
b= read_matrix('asgn0_vecC')
print(' Solution according to jacobi method', jacobi(a,b,1000))
print('Solution according to Gauss method', gauss_seidel_nodiag(a,b))
"""Output:
Solution according to jacobi method [2.97916519278387, 2.215599575521754, 0.211284046692607, 0.15231694375663238, 5.715033604527767]
Solution according to Gauss method [2.9791647917923516, 2.2155999742491046, 0.2112839156849176, 0.15231719966776006, 5.715033463883198]
"""

