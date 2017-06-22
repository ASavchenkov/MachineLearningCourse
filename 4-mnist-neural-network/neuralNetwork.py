import struct
import numpy as np
from scipy.optimize import minimize

reg_factor = 0.0 #not necessary this time. I believe you know how to do regularization
LEARNING_RATE=0.5

def read_images(filename):
    with open(filename,'rb') as f:
        #magic that reads binary format 
        magic_number, num_images, dim1, dim2  = struct.unpack('>IIII',f.read(16))
    
        return np.fromstring(f.read(),dtype=np.uint8).reshape((num_images,dim1*dim2))

def read_labels(filename):
    with open(filename,'rb') as f:
        magic_number, num_labels = struct.unpack('>II',f.read(8))
        return np.fromstring(f.read(),dtype=np.uint8)

#only takes integers > 0
def one_hot_encode(arr):
    one_hot = np.zeros((arr.shape[0],10))
    one_hot[np.arange(arr.shape[0]),arr] = 1
    return one_hot

#functions that take the parameters and flatten/unflatten them
def pack_parameters(thetas,biases):
    params = thetas + biases
    return np.concatenate([p.flatten() for p in params])

def unpack_parameters(params,shape_groups):
    unpacked_groups = list()
    curIdx = 0
    for group in shape_groups:
        unpacked = list()
        for shape in group:
            flat = params[curIdx:curIdx+np.prod(shape)]
            curIdx+=np.prod(shape)
            unpacked.append(np.reshape(flat,shape))
        unpacked_groups.append(unpacked)
    return unpacked_groups

def stats(arr):
    print(np.mean(arr),np.var(arr))

def sigmoid(z):
    return 1/(1+np.exp(-z))

#YOU NEED TO WRITE THIS----------------------------------
def sigmoid_grad(x):
    return x

#YOU NEED TO WRITE THIS----------------------------------
#outputs a tuple z,a for gradient descent
def nn_layer_forward(theta,bias,x):
    return None


#YOU NEED TO WRITE THIS----------------------------------
#this should give not just the hypothesis
#but also the z and a in each layer.
#(we need these for backpropagation)
#(remember that X is a sort of "a0",
#so you should have one extra "a")
def forward_neural_network(thetas, biases, X):
    return None

#YOU NEED TO WRITE THIS----------------------------------
#the cost function, should take params, unpack them based on the
#shapes of your parameters, then calculate the cost
def cost(params,shapes,X,Y):
    thetas, biases = unpack_parameters(params,shapes)
    return 0

    #also print out the percent correct (defined below)
   
#YOU NEED TO WRITE THIS----------------------------------
#this is the tough one. I would suggest doing the math manually.
#if you want to look at a generic solution though feel free to look
#at the solution branch
def gradient(params, shapes, X, Y):
    thetas, biases = unpack_parameters(params,shapes)
    return np.zeros_like(params)

def calculate_correct(h,Y):
    h_int = np.argmax(h,axis=1)
    Y_int = np.argmax(Y,axis=1)
    num_correct = np.sum(np.equal(h_int,Y_int))
    print('accuracy:',float(num_correct)/float(Y_int.shape[0])*100)


#this code is a freebie. Creates the weights and biases in the shapes you want
def create_parameters(layerShapes):
    thetas = list()
    biases = list()
    theta_shapes = list()
    bias_shapes = list()
    for shape in layerShapes:
        thetas.append(np.random.normal(0,1/np.sqrt(shape[0]),shape))
        biases.append(np.random.normal(0,1/np.sqrt(shape[0]),shape[1]))
        theta_shapes.append(thetas[-1].shape)
        bias_shapes.append(biases[-1].shape)


    return thetas, biases, theta_shapes, bias_shapes
        


if __name__ == '__main__':
    
    np.set_printoptions(linewidth=99999) #to make printing large arrays easier

    X = read_images('../data/train-images-idx3-ubyte')
    X = X.astype(float)/255
    Y_int = read_labels('../data/train-labels-idx1-ubyte')
    Y = one_hot_encode(Y_int)

    num_samples = X.shape[0] #I calculate this beforehand for convenience
    
    #example generated layers
    thetas, biases, theta_shapes, bias_shapes = create_parameters([(784,100),(100,10)])
    
    
    packed = pack_parameters(thetas,biases)
    #make sure your gradient is working ok first.
    gradient(packed,(theta_shapes,bias_shapes),X,Y)


    #should max out around 99 percent.
    #BFGS isn't the best way to optimize neural networks though, so it could "crash" if it gets a "nan"
    #somewhere. If it seems to be training fine, and fails suddenly, just give it another go.
    fmin = minimize(fun=cost, x0=packed, args=((theta_shapes,bias_shapes), X,Y), method='L-BFGS-B', jac=gradient)


    




