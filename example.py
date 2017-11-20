from kpnet import NeuralNet 
from kpnet import Layer 
import numpy as np
import matplotlib.pyplot as plt

#Sample input: xor input
x = np.array([[0,0,1,1],[0,1,0,1]])

#Sample out
y = np.array([[0, 1, 1, 0],[1, 0, 0, 1]])

network = NeuralNet()

input_layer = Layer(number_of_neurons=2,layer_type='input')
hidden_layer1 = Layer(number_of_neurons=8)
hidden_layer2 = Layer(number_of_neurons=4)
output1 = Layer(number_of_neurons=2,layer_type='output')

network.add(input_layer)
network.add(hidden_layer1)
network.add(hidden_layer2)
network.add(output1)
network.compile()

cost = network.train(x=x,y_hat=y,print_cost=True,learning_rate=0.08,epoch=10000,store_cost_after_iterations=500)
np.round(network.forward_feed(x))
plt.plot(np.arange(0,10000,10000/len(cost)),cost)
plt.show()



