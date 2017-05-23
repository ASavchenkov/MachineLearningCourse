import struct
import numpy as np


def read_images(filename):
    with open(filename,'rb') as f:
        #magic that reads binary format 
        magic_number, num_images, dim1, dim2  = struct.unpack('>IIII',f.read(16))
    
        return np.fromstring(f.read(),dtype=np.uint8).reshape((num_images,dim1,dim2))

def read_labels(filename):
    with open(filename,'rb') as f:
        magic_number, num_labels = struct.unpack('>II',f.read(8))
        return np.fromstring(f.read(),dtype=np.uint8)

if __name__=="__main__":
    train_images = read_images('../data/train-images-idx3-ubyte')
    train_labels = read_labels('../data/train-labels-idx1-ubyte')
    
    print(train_images.shape,train_labels.shape)
