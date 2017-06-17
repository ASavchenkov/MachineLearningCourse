import struct
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

reg_factor = 0. #only for the second part of the homework

def read_images(filename):
    with open(filename,'rb') as f:
        #magic that reads binary format 
        magic_number, num_images, dim1, dim2  = struct.unpack('>IIII',f.read(16))
    
        return np.fromstring(f.read(),dtype=np.uint8).reshape((num_images,dim1*dim2))

def read_labels(filename):
    with open(filename,'rb') as f:
        magic_number, num_labels = struct.unpack('>II',f.read(8))
        return np.fromstring(f.read(),dtype=np.uint8)

#write the hypothesis function here
def hypothesis(theta,X):
    z = np.matmul(X,theta)
    return 1/(1+np.exp(-z))

#write the cost function here
#scipy's minimize flattens arrays for whatever reason,
#so you have to reshape them
def cost(theta,X,Y):
    theta = np.reshape(theta,(785,10))
    h = hypothesis(theta,X)
    cost = -Y*np.log(h) - (1-Y)*np.log(1-h)

    regularization = reg_factor * np.mean(theta*theta)/2 #only for the second half of the assignment

    finalCost = np.mean(cost) + regularization
    print(finalCost)
    return finalCost

#write the gradient function here
def gradient(theta, X,Y):
    theta = np.reshape(theta,(785,10))
    h = hypothesis(theta,X)
    err = h-Y
    grad = np.matmul(X.T,err)/X.shape[0] + reg_factor/X.shape[0]*theta
    return grad

#only takes integers > 0
def one_hot_encode(arr):
    one_hot = np.zeros((arr.shape[0],np.max(arr)+1))
    one_hot[np.arange(arr.shape[0]),arr] = 1
    return one_hot

def predict_all(theta,X,Y,Y_int):
    predictions = hypothesis(np.reshape(theta,(785,10)),X)
    predictions_int = np.argmax(predictions,axis=1)
    num_correct = np.sum(np.equal(predictions_int,Y_int))
    print('accuracy:',float(num_correct)/float(Y_int.shape[0])*100)

    print(cost(theta,X,Y))


if __name__ == '__main__':
    
    X = read_images('../data/train-images-idx3-ubyte')
    #normalize the data to be between 0 and 1
    X = X/255
    Y_int = read_labels('../data/train-labels-idx1-ubyte')
    Y = one_hot_encode(Y_int)

    num_samples = X.shape[0] #I calculate this beforehand for convenience

    #again prepend a column of ones to the front (of X)

    ones = np.ones((num_samples,1))
    X = np.concatenate((ones,X),axis=1)

    #create an array of three theta
    #this time filling it with zeroes.
    theta = np.zeros((X.shape[1],Y.shape[1]))
    theta = theta.flatten()

    predict_all(theta,X,Y,Y_int) 

    fmin = minimize(fun=cost, x0=theta, args=(X,Y), method='TNC', jac=gradient)#, options={'maxiter':40})
    theta = fmin.x
    predict_all(theta,X,Y,Y_int)
    print(cost(theta,X,Y))


    




