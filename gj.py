from Fun import*
a= read_matrix('asgn0_matB')
print(a)
b= read_matrix('asgn0_vecC')
print(b)
aug=[]
n=len(a)
for i in range(n):
        aug.append(a[i] + b[i])   
        print(aug)
solution= gauss_jordan(aug)
print('Solution of the given matrix is ',solution)
#Solution of the given matrix is  [-1.76181704399786, 0.8962280338740123, 4.051931404116157, -1.6171308025395417, 2.041913538501913, 0.15183248715593547]