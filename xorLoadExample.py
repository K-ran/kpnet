from kpnet import NeuralNet 
from kpnet import Layer 
import numpy as np
import matplotlib.pyplot as plt

# Sample input: xor input
x = np.array([[0,0,1,1],[0,1,0,1]])

# Sample out
y = np.array([[0, 1, 1, 0],[1, 0, 0, 1]])

# Create NeuralNet object
network = NeuralNet()

# Define the required layers
input_layer = Layer(number_of_neurons=2,layer_type='input')
hidden_layer1 = Layer(number_of_neurons=8)
hidden_layer2 = Layer(number_of_neurons=4)
output1 = Layer(number_of_neurons=2,layer_type='output')

# Put them all together
network.add(input_layer)
network.add(hidden_layer1)
network.add(hidden_layer2)
network.add(output1)
network.compile()

network.load('xornetwork.h5')
print(network.forward_feed(x))
# print(np.round(network.forward_feed(x)))
# plt.plot(np.arange(0,10000,10000/len(cost)),cost)
# plt.show()



