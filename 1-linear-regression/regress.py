import numpy as np
import matplotlib.pyplot as plt

#returns a numpy array
def load_data():
    return np.loadtxt('../data/ex1data1.txt', delimiter = ',')
    
if __name__ == '__main__':
    
    data = load_data()
