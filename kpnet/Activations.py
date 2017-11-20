import numpy as np

def sigmoid(x):
    """Sigmoid function.
       @param x: input matrix/vector.   
       @return: elementwise sigmoid"""
    return 1/(1+np.exp(-x))

def sigmoid_dash(x):
    """Sigmoid derivative function.
       @param x: input matrix/vector.   
       @return: elementwise sigmoid derivative"""
    return sigmoid(x)*(1-sigmoid(x))    

def relu(x):
    """Rectified linear activation function.
       @param x: input matrix/vector.
       @return: elementwise relu"""
    return np.maximum(x,0)

def relu_dash(x):
    """Relu derivative function.
       @param x: input matrix/vector.   
       @return: elementwise relu derivative"""
    x[x<=0]=0
    x[x>0]=1
    return x

def softmax(x):
    """Softmax function.
       @param x: input matrix/vector.   
       @return: columnwise softmax"""
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis=0)
    return x_exp/x_sum