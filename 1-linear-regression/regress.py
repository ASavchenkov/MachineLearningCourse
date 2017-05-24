import numpy as np
import matplotlib.pyplot as plt

#returns a numpy array
def load_data():
    return np.loadtxt('../data/ex1data1.txt', delimiter = ',')


if __name__ == '__main__':
    
    data = load_data()

    #use matplotlib.pyplot to make a scatterplot of the data
    x =  data[:,0]
    y = data[:,1]
    plt.scatter(x,y)

    #prepend a column of ones to the front
    #(the "this exists" feature that is always one)
    #because it is always one and is multiplied against,
    #most times, the feature is omitted from data,
    #and models simply add a "bias"
    #term to avoid the unnecessary calculation. It makes
    #code slightly more complicated though
    #use numpy.concatenate to accomplish this
    #you should now have an array with shape [97,3]
    
    num_samples = data.shape[0] #I calculate this beforehand for convenience
    print(num_samples)

    ones = np.ones((num_samples,1))
    data = np.concatenate((ones,data),axis=1)
    
    #Then split the data such that the first two columns
    #are one matrix, and the third is it's own array
    features = data[:,:2]
    target = data[:,2:]

    #create an array of two weights (filled with ones) and
    weights = np.zeros((2,1))
    
    for i in range(500):
        #calculate the hypothesis for each sample
        #(you will want a shape of (2,1) for np.matmul to work)

        hypothesiss = np.matmul(features,weights)

        #calculate the cost function per data sample
        cost = np.power(hypothesiss-target,2)/num_samples/2

        #then sum it up to get the total cost
        #this number will vary based off of the weights
        total_cost = np.sum(cost)
        print(total_cost)

        #calculate the gradient (remember you want to sum along the samples
        #but not along the features.
        #You should have 2 gradients in the end, one for each weight)
        learning_rate = 0.005
        grad = -learning_rate/num_samples * np.sum((hypothesiss-target) * features,axis=0)

        #now alter the weights by the gradient, and wrap this all in a big for loop.
        #in a nice for loop. You should see the total cost go down over each iteration.
        weights = weights+grad
    
    #make some dummy samples and plot their hypothesiss along with
    #the original data. It should form a line

    dummies = np.stack((np.ones((10)),np.linspace(0,30,10)),axis=1)
    hypothesiss = np.matmul(dummies,weights)
    plt.scatter(dummies[:,1],hypothesiss[:,0])
    plt.show()
    

