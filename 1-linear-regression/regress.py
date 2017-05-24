import numpy as np
import matplotlib.pyplot as plt

#returns a numpy array
def load_data():
    return np.loadtxt('../data/ex1data1.txt', delimiter = ',')


if __name__ == '__main__':
    
    data = load_data()

    #use matplotlib.pyplot to make a scatterplot of the data

    #prepend a column of ones to the front
    #(the "this exists" feature that is always one)
    #because it is always one and is multiplied against,
    #most times, the feature is omitted from data,
    #and models simply add a "bias"
    #term to avoid the unnecessary calculation. It makes
    #code slightly more complicated though
    #use numpy.concatenate to accomplish this
    #you should now have an array with shape [97,3]

    #Then split the data such that the first two columns
    #are one matrix, and the third is it's own array

    #create an array of two weights (filled with ones) and
        
    #calculate the hypothesis for each sample
    #(you will want a shape of (2,1) for np.matmul to work)


    #calculate the cost function per data sample

    #then sum it up to get the total cost
    #this number will vary based off of the weights

    #calculate the gradient (remember you want to sum along the samples
    #but not along the features.
    #You should have 2 gradients in the end, one for each weight)

    #now alter the weights by the gradient, and wrap this all in a big for loop.
    #in a nice for loop. You should see the total cost go down over each iteration.
    
    #make some dummy samples and plot their hypothesiss along with
    #the original data. It should form a line

    

