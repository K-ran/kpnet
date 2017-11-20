import numpy as np

class Layer(object):
	"""This class represents a layer of the Neural Network."""
	def __init__(self, number_of_neurons,layer_type='hidden',layer_activation='relu'
				,dropout=False,keep_prob=0.75):
		"""Constructor of Layer class"""
		self.number_of_neurons = number_of_neurons
		self.layer_type = layer_type

		if layer_type =='output':
			self.layer_activation ='softmax'
		else:	
			self.layer_activation = layer_activation	

		self.dropout = dropout
