from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt



#returns a numpy array
def load_data():
    return np.loadtxt('../data/ex2data1.txt', delimiter = ',')

def hypothesis(theta,X):
    #insert code here
    return 0.5

def cost(theta,X,Y):
    #insert code here
    return 0

def gradient(theta, X,Y):
    #insert code here
    return 0

def plot_points(points,ax):
    x = points[:,0]
    y = points[:,1]
    label = points[:,2]
    ax.scatter(x,y,label,c=plt.cm.coolwarm(label))

def mapFeatures(x,y, order):
    #insert code here
    return np.array(x,y).T

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
    X=None
    Y=None
    #again prepend a column of ones to the front (of X)

    #create an array of three theta
    #this time filling it with zeroes.
    theta = [0,0,0]         #cost function should give 0.693
    # theta = [-24,0.2,0.2]   #cost function should give 0.218
    
    #This should output 0.693 for zeroeif everything is correct
    print(cost(theta,X,Y))
    
    #instead of doing gradient descent on our own, we're going to start
    #using other people's tools. We use scipy's fmin_tnc to optimize our
    #function, but it needs a function, and a gradient function, which 
    #you need to write above.

    thetas = opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(X,Y))
    theta= np.expand_dims(thetas[0],1)
    
    print(hypothesis(np.array([[1,45,85]]),theta))
    #the probability here should be 0.776

    #plot a surface. I'll give you this one for free too because...
    #again, matplotlib is super obtuse
    xmin,xmax,ymin,ymax = 0,100,0,100
    x,y = np.meshgrid(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100))

    dummies = np.stack((x.flatten(), y.flatten()),axis=1)
    mappedDummies = mapFeatures(dummies[:,0],dummies[:,1],6) #ONLY FOR PART 2
    mappedDummies = np.concatenate((np.ones((100*100,1)),mappedDummies),axis=1)

    surf = ax.plot_surface(x,y,np.reshape(hypothesis(theta,dummies),(100,100)),cmap=plt.cm.coolwarm)
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

