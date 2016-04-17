import numpy as np
import matplotlib.pyplot as plt
import random



# cost function 
def cost(a,b,c,d, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        sigmai2=c*x**2+d
        totalError += .5*(y - (a * x + b)) ** 2/float(sigmai2)+.5*np.log(sigmai2)       
    return totalError 


#  gradient desecent parameters update
def step_gradient(a_current,b_current, c_current,d_current,points,learningRate):
    a_gradient=0
    b_gradient=0
    c_gradient=0
    d_gradient=0
    for i in range(0,len(points)): 
        xi = points[i, 0]
        yi = points[i, 1]
        sigmai2=c_current*xi**2+d_current
        a_gradient += -xi*(yi-a_current*xi-b_current)/float(sigmai2)
        b_gradient += -(yi-a_current*xi-b_current)/float(sigmai2) 
        c_gradient += .5*(xi**2/sigmai2)-.5*(xi**2)*(yi-a_current*xi-b_current)**2/(sigmai2)**2
        d_gradient += .5*(1/sigmai2)-.5*(yi-a_current*xi-b_current)**2/(sigmai2)**2
 
    new_a = a_current - (learningRate * a_gradient)
    new_b = b_current - (learningRate * b_gradient)
    new_c = c_current - (learningRate * c_gradient)
    new_d = d_current - (learningRate * d_gradient)

    # limit c, d to [0, 5]
    new_c=min(new_c,5)
    new_c=max(new_c,0)   
    new_d=min(new_d,5)
    new_d=max(new_d,0)
    
    return [new_a, new_b, new_c,new_d]
 



############################################################################## 
##############################################################################
# linear fitting in signal-dependent noise


#  load data from csv file
path='../data/'
filename='data_1_1.csv'
points = np.genfromtxt(path+filename, delimiter=",")
points=points[1:,:]
X=points[:,0]
Y=points[:,1]

# plot y vs x
plt.plot(X,Y,'b+')


learning_rate = 0.001 # learning rate
num_iterations=3000

# initialize parameters
a = random.random()
b = random.random()
c = random.random()
d = random.random()
initial_cost= cost(a,b,c,d, points);

print "Initial values: a = {0}, b = {1}, c={2},d2={3}, cost = {4}".format(a, b, c,d, initial_cost)
print "Running..."


# gradient descent running
for i in range(num_iterations):
    a, b,c,d = step_gradient(a, b,c,d, np.array(points), learning_rate)
    

final_cost= cost(a,b,c,d, points);
print "After {0} iterations a = {1}, b = {2}, sigma_u={3}, sigma_v^2={4},cost = {5}".format(num_iterations, a, b,c,d, final_cost)
plt.plot(X,a*X+b,'r')
plt.xlabel("X")
plt.ylabel('Y')
plt.savefig("xyplot.eps", format="eps")

# varinace of estimated a and b
temp_a=0
temp_b=0
for i in range(0,len(points)): 
    xi = points[i, 0]
    sigmai2=c*xi**2+d
    temp_a+= xi**2/sigmai2
    temp_b+= 1/sigmai2
var_a=1/temp_a
var_b=1/temp_b
print "variance of a {0} and b {1} " .format(var_a, var_b)



