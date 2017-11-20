from kpnet import Layer
import numpy as np
from kpnet.Activations import *

class NeuralNet():
    """this is the main neural net class"""
    def __init__(self):
        """Constructor of the Neural Network class.
        """  
        self.layers = list()
        self.parameters = dict()
        self.forward_parameters = dict()
        self.gradiants = dict()

    def add(self, layer):
        """Adds a new layer in the structure of the neural network.
        """
        self.layers.append(layer) 

    def compile(self):
        """Creates and initilizes the weights randomly with Xavier initilizetion.
           Creats empty placeholders(dict) for gradients 
        """
        for i in range(1,len(self.layers)):
            self.parameters['W'+str(i)] = np.random.normal(0,
                                                      1.0/self.layers[i-1].number_of_neurons,
                                                      (self.layers[i].number_of_neurons, self.layers[i-1].number_of_neurons)
                                                      )
            self.parameters['b'+str(i)] = np.zeros((self.layers[i].number_of_neurons,1))

    def _init_gradiants(self,m):        
        """Initilizes the gradiants dictionary
           @param m: number of input examples
        """ 
        for i in range(1,len(self.layers)):
            self.gradiants['dW'+str(i)] = np.zeros((self.layers[i].number_of_neurons, self.layers[i-1].number_of_neurons))
            self.gradiants['db'+str(i)] = np.zeros((self.layers[i].number_of_neurons,1))
            self.gradiants['dZ'+str(i)] = np.zeros((self.layers[i].number_of_neurons,m))


    def _activation(self, layer,z):
        """Applies correct activation on Z.
        """
        if layer.layer_activation=='relu':
            return relu(z)
        elif layer.layer_activation=='sigmoid':
            return sigmoid(z)
        elif layer.layer_activation=='softmax':
            return softmax(z)       

    def _reverse_activation(self, layer,z):
        """Applies correct activation derivative on Z.
        """
        if layer.layer_activation=='relu':
            return relu_dash(z)
        elif layer.layer_activation=='sigmoid':
            return sigmoid_dash(z)
        

    def forward_feed(self, X):
        """Feeds the input X through the network.
           Also sets the paramaters required for backprop.

           @param X: numpy input array/matrix as input vector/matrix.
           @return: return final output
        """
        self.forward_parameters['A0'] = X
        for i in range(1,len(self.layers)):
            self.forward_parameters['Z'+str(i)]= np.dot(self.parameters['W'+str(i)],self.forward_parameters['A'+str(i-1)])+self.parameters['b'+str(i)]
            self.forward_parameters['A'+str(i)]= self._activation(self.layers[i],self.forward_parameters['Z'+str(i)]) 
        return  self.forward_parameters['A'+str(len(self.layers)-1)]

    def _cost(self, y_hat):
        """Calculates the cost using corss entorpy loss
           @param y_hat: expected output
           @return: cost 
        """

        # total examples
        m =  y_hat.shape[1]
        y = self.forward_parameters['A'+str(len(self.layers)-1)]
        y_t = y*y_hat

        y_t[y_t==0]=1 #will be removed by log

        # log loss/ cross entropy loss
        y_t = (1/m)*np.sum(-np.log(y_t),keepdims=True)
        return float(y_t)

    def _backpropogation(self,y,y_hat):
        """ Does one pass of backpropogation
            @param y: output from forward feed
            @param y_hat: expected output
            @return: gradient
        """
        m = y_hat.shape[1]
        self._init_gradiants(m)
        self.gradiants['dZ'+str(len(self.layers)-1)] = y - y_hat 
        for i in reversed(range(1,len(self.layers))):
            self.gradiants['dW'+str(i)] = (1.0/m)*np.dot(self.gradiants['dZ'+str(i)],self.forward_parameters['A'+str(i-1)].T)
            self.gradiants['db'+str(i)] = (1.0/m)*np.sum(self.gradiants['dZ'+str(i)],axis=1,keepdims=True)    
            # Last step not required for input layer.
            if i!=1:
                self.gradiants['dZ'+str(i-1)] = np.dot(self.parameters['W'+str(i)].T,self.gradiants['dZ'+str(i)])*(self._reverse_activation(self.layers[i-1],self.forward_parameters['Z'+str(i-1)]))


    def _update_weights(self, learning_rate):
        """Update the weights from the gradiants. Single weight update.
           @param learning_rate: decides the rate of learning.
        """
        for i in range(1,len(self.layers)):
            self.parameters['W'+str(i)] = self.parameters['W'+str(i)] - learning_rate*self.gradiants['dW'+str(i)]
            self.parameters['b'+str(i)] = self.parameters['b'+str(i)] - learning_rate*self.gradiants['db'+str(i)]

    def train(self,x,y_hat,learning_rate=0.05,epoch=1000,print_cost=False,store_cost_after_iterations=100):
        """Train function, uses the above all method to train the model
           @param x: input matrix/vector
           @param y: expected output(1-hot vectors)
           @param learning_rate: Rate of learning algorithm
           @param epoch: number of epochs
           @param print_cost: print cost while training
           @return: list of cost after every 100 epochs
        """
        costs=[]
        for i in range(0,epoch):
            self.forward_feed(x)
            cost = self._cost(y_hat)
            if i%store_cost_after_iterations==0 :
                costs.append(cost)
                if print_cost:
                    print(cost)
            y = self.forward_parameters['A'+str(len(self.layers)-1)]
            self._backpropogation(y,y_hat)
            self._update_weights(learning_rate)
        costs.append(cost)
        if print_cost:
            print(cost)
        return costs            