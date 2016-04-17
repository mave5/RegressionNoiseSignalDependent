from numpy import *
import numpy as np
from numpy import linalg as LA

# y = ax + b + x u + v
# a is slope, b is y-intercept, u,v are Gaussian noise


# cost function 
def cost(a,b,c,d, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        sigmai2=c*x**2+d
        totalError += .5*(y - (a * x + b)) ** 2/float(sigmai2)+.5*log(sigmai2)       
    return totalError 


#  numerical gradient
def numerical_gradient(a_current,b_current, c_current,d_current,points):
    delta=1e-6    
    a_gradient = (cost(a_current+delta,b_current,c_current,d_current,points)-cost(a_current,b_current,c_current,d_current,points))/delta
    b_gradient = (cost(a_current,b_current+delta,c_current,d_current,points)-cost(a_current,b_current,c_current,d_current,points))/delta
    c_gradient = (cost(a_current,b_current,c_current+delta,d_current,points)-cost(a_current,b_current,c_current,d_current,points))/delta
    d_gradient = (cost(a_current,b_current,c_current,d_current+delta,points)-cost(a_current,b_current,c_current,d_current,points))/delta
 
    return array([a_gradient, b_gradient, c_gradient,d_gradient])



#  analytical gradient
def analytical_gradient(a,b, c,d,points):
    a_gradient=0
    b_gradient=0
    c_gradient=0
    d_gradient=0
    for i in range(0,len(points)): 
        xi = points[i, 0]
        yi = points[i, 1]
        sigmai2=c*xi**2+d
        a_gradient += -xi*(yi-a*xi-b)/float(sigmai2)
        b_gradient += -(yi-a*xi-b)/float(sigmai2) 
        c_gradient += .5*(xi**2/sigmai2)-.5*(xi**2)*(yi-a*xi-b)**2/(sigmai2)**2
        d_gradient += .5*(1/sigmai2)-.5*(yi-a*xi-b)**2/(sigmai2)**2
 
    return array([a_gradient, b_gradient, c_gradient,d_gradient])


# data generation function
def generate_data(a,b,su,sv,N):
    X=np.zeros(N)
    Y=np.zeros(N)
    points=np.zeros((N,2))
    random.seed()
    for k in xrange(N):
        X[k]=random.uniform(2,10)
        u = np.random.normal(0, su, 1)
        v = np.random.normal(0, sv, 1)
        e=X[k]*u+v
        Y[k]= a* X[k]+b+e
    points[:,0]=X
    points[:,1]=Y
    return points




#######################################################
# gradient checking


# generate some sample data
XY=generate_data(2,5,1,1,100)

# randomly set parameters
a=random.uniform(-10,10)
b=random.uniform(-10,10)
c=random.uniform(1,10)
d=random.uniform(1,10)

# compute numerical gradient
num_grad=numerical_gradient(a,b,c,d,XY)

# compute analytical gradient
grad=analytical_gradient(a,b,c,d,XY)


# error between numerical and analytical gradients
diff=LA.norm(num_grad-grad)/LA.norm(num_grad+grad)

print "difference:", diff

print "numerical gradients:",  num_grad
print "analytical gradients:", grad



