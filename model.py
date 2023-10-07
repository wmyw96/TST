import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class RegressionNN(nn.Module):
	'''
		A class to implement standard relu nn for regression

		...
		Attributes
		----------
		use_input_dropout : bool
			whether to use dropout (True) or not (False) in the input layer
		input_dropout : nn.module
			pytorch module of input dropout
		relu_stack : nn.module
			pytorch module to implement relu neural network

		Methods
		----------
		__init__(x, is_training=False)
			Initialize the module
		forward()
			Implementation of forwards pass
	'''
	def __init__(self, d, depth, width, input_dropout=False, dropout_rate=0.0):
		'''
			Parameters
			----------
			d : int
				input dimension
			depth : int
				the number of hidden layers of relu network
			width : int
				the number of units in each hidden layer of relu network
			input_dropout : bool, optional
				whether to use input dropout in the input layer (True)
			dropout_rate: float, optional
				the dropout rate for the input dropout
		'''
		super(RegressionNN, self).__init__()
		self.use_input_dropout = input_dropout
		self.input_dropout = nn.Dropout(p=dropout_rate)

		relu_nn = [('linear1', nn.Linear(d, width)), ('relu1', nn.ReLU())]
		for i in range(depth - 1):
			relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
			relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))
		relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))

		self.relu_stack = nn.Sequential(
			OrderedDict(relu_nn)
		)

	def forward(self, x, is_training=False):
		'''
			Parameters
			----------
			x : torch.tensor
				the (n x p) matrix of the input
			is_training : bool
				whether the forward pass is used in the training (True) or not,
				used for dropout module

			Returns
			----------
			pred : torch.tensor
				(n, 1) matrix of the prediction
		'''
		if self.use_input_dropout and is_training:
			x = self.input_dropout(x)
		pred = self.relu_stack(x)
		return pred

