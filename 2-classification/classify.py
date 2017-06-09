from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


reg_factor = 0.1 #only for the second part of the homework

#returns a numpy array
def load_data():
    return np.loadtxt('../data/ex2data2.txt', delimiter = ',')

#write the hypothesis function here
def hypothesis(theta,X):
    z = np.matmul(X,theta)
    return 1/(1+np.exp(-z))

#write the cost function here
def cost(theta,X,Y):
    theta = np.expand_dims(theta,1)
    h = hypothesis(theta,X)
    cost = -Y*np.log(h) - (1-Y)*np.log(1-h)

    regularization = reg_factor * np.mean(theta*theta)/2 #only for the second half of the assignment

    finalCost = np.mean(cost) + regularization
    return finalCost

#write the gradient function here
def gradient(theta, X,Y):
    theta = np.expand_dims(theta,1)
    h = hypothesis(theta,X)
    err = h-Y
    grad = np.matmul(X.T,err)/X.shape[0] + reg_factor/X.shape[0]*theta
    return grad

def plot_points(points,ax):
    x = points[:,0]
    y = points[:,1]
    label = points[:,2]
    ax.scatter(x,y,label,c=plt.cm.coolwarm(label))

def mapFeatures(x,y, order):
    colList = list()
    for i in range(1,order):
       for j in range(i+1):
           colList.append(np.power(x,j)*np.power(y,i-j))
    return np.array(colList).T 

if __name__ == '__main__':
    
    data = load_data()
    print(data.shape)
    # data = np.array([[1,1,1],[1,0,1],[0,1,0],[0,0,0],[0.5,1,1],[0.5,0,1],[0.5,1,0],[0.5,0,0],[0,0,1]])

    num_samples = data.shape[0] #I calculate this beforehand for convenience
    print(num_samples)

    #use matplotlib.pyplot to make a scatterplot of the data
    #I'll give you this one as a freeby since matplotlib is a bit obtuse
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    plot_points(data,ax) 

    
    #Then split the data such that the first two columns
    #are one matrix, and the third is it's own array
    X = data[:,:2]
    Y = data[:,2:]
    
    #this is for the second half where we add features

    X = mapFeatures(X[:,0],X[:,1],6)
    print(X.shape)

    #again prepend a column of ones to the front (of X)

    ones = np.ones((num_samples,1))
    X = np.concatenate((ones,X),axis=1)

    #create an array of three theta
    #this time filling it with zeroes.
    theta = np.zeros((X.shape[1]))
    
    #This should output 0.693 if everything is correct
    print(cost(theta,X,Y))

    thetas = opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(X,Y))
    theta= np.expand_dims(thetas[0],1)
    
    print(hypothesis(np.array([[1,45,85]]),theta))
    #the probability here should be 0.776

    #plot a surface. I'll give you this one for free too because...
    #again, matplotlib is super obtuse
    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    dummies = np.stack((x.flatten(), y.flatten()),axis=1)
    mappedDummies = mapFeatures(dummies[:,0],dummies[:,1],6)
    mappedDummies = np.concatenate((np.ones((100*100,1)),mappedDummies),axis=1)

    surf = ax.plot_surface(x,y,np.reshape(hypothesis(theta,mappedDummies),(100,100)),cmap=plt.cm.coolwarm)
    plt.show()

    #PART 2:

    #now try to use this for ex2data2. You might notice it won't work.
    #that's because you need more features. Try creating features by multiplying
    #x and y, i.e adding features like xy, x^2, and y^2, etc, up to the 6th power
    #you can do this before adding the ones since it's not necessary to map them
    #and since you have to do some fancy combinatorics to allocate powers among
    #more than 2 things properly.

    #BONUS ROUND:

    #apply regularization to your cost function and gradient, and see how it affects
    #your model for ex2data2

