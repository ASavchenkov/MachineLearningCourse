import struct
import numpy as np
from scipy.optimize import minimize

reg_factor = 0.0 #only for the second part of the homework
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

def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))

#outputs a tuple z,a for gradient descent
def nn_layer_forward(theta,bias,x):
    z = np.matmul(x,theta)
    a = sigmoid(z)
    return z,a

#this should give not just the hypothesis
#but also the z and a in each layer.
#(we need these for backpropagation)
#(remember that X is a sort of "a0",
#so you should have one extra "a")
def forward_neural_network(thetas, biases, X):
     
    Zs = list()
    As = list()
    cur_a = X
    As.append(cur_a)
    for theta,bias in zip(thetas,biases):
        z,a = nn_layer_forward(theta,bias,cur_a)
        Zs.append(z)
        As.append(a)
        cur_a = a

    return Zs,As #the last element of As is the "hypothesis"

#the cost function, should take params, unpack them based on the
#shapes of your parameters, then calculate the cost
def cost(params,shapes,X,Y):
    thetas, biases = unpack_parameters(params,shapes)

    Zs,As = forward_neural_network(thetas,biases,X)
    h = As[-1] #we only care about the last value in the cost function
    cost = -Y*np.log(h) - (1-Y)*np.log(1-h)

    regularization = reg_factor * np.mean(params*params)/2
    #the flattened parameters are also convenient for calculating regularization

    finalCost = np.mean(cost) + regularization
    print(finalCost)
    calculate_correct(h,Y)
    return finalCost
   
#this is the tough one. I would suggest doing the math manually.
#if you want to look at a generic solution though feel free to look
#at the solution branch
def gradient(params, shapes, X, Y):
    thetas, biases = unpack_parameters(params,shapes)
    Zs,As = forward_neural_network(thetas,biases,X)
    #since we're going backwards it makes iterating easier
    Zs,As,thetas,biases = [list(reversed(l)) for l in (Zs,As,thetas,biases)]
    deltas = list()

    #calculate gradient at last layer using the cost function gradient
    cur_delta = (As[0]-Y)*sigmoid_grad(Zs[0])
    deltas.append(cur_delta)
    theta_grads, bias_grads = list(), list()

    for z,theta,bias in zip(Zs[1:],thetas[:-1], biases[:-1]):
        cur_delta = np.matmul(cur_delta, theta.T) * sigmoid_grad(z)
        deltas.append(cur_delta)
    
    for delta, a in zip(deltas,As[1:]):
        theta_grads.append(np.matmul(delta.T,a).T/delta.shape[0])
        bias_grads.append(np.sum(delta,axis=0)/delta.shape[0])
    theta_grads, bias_grads = [list(reversed(l)) for l in (theta_grads,bias_grads)]
    packed_grads = pack_parameters(theta_grads,bias_grads)
    packed_grads = packed_grads*LEARNING_RATE
    return packed_grads

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



    fmin = minimize(fun=cost, x0=packed, args=((theta_shapes,bias_shapes), X,Y), method='L-BFGS-B', jac=gradient)


    




