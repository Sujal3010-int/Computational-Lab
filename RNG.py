import matplotlib.pyplot as plt 
class RnG:
    def __init__(self):
        pass
    def iterative(c,n):
        x=0.1
        list=[]
        for _ in range (n):
            x = (c*x)*(1-x)
            list.append(x)
        return list
def Plot(x, y , title='Sample Plot', xlabel='X-axis Label', ylabel='Y-axis Label',file_name='sample_plot.png'):
    plt.scatter(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)  
    plt.savefig(file_name)
    plt.show()

c= [3.2,3.6,3.9,1.2,2.8]
k=[5,10,15,20]
for z in range (len(c)):
    print("The list of random nummbers are for c=", c[z], "is", RnG.iterative(c[z],1000))
    z= z+1
x=[]
z=[]
y=[]
z= (RnG.iterative(4.8,500))
x.append(RnG.iterative(4.8,200))
for j in range(200):
    y.append(z[5+j])
    j=j+1
Plot(x, y , title='For c=4.8', xlabel='X-axis Label', ylabel='Y-axis Label',file_name='sample_plot.png')


#Output 
