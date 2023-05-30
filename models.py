import torch
import torch.nn as nn
import torch.nn.parallel

import math
import numpy as np

class Generator(nn.Module):
	def __init__(self, nz, nf=1000, num_hidden_layers=10):
		super(Generator, self).__init__()

		self.layers = nn.ModuleList()

		num_hidden = np.linspace(nz, nf*1.5, num_hidden_layers)

		h_prev = nz
		for h in num_hidden:
			h = int(h)
			self.layers.append(nn.Sequential(
									nn.Linear(h_prev, h, bias=True),
									nn.LeakyReLU(),
									nn.BatchNorm1d(h))
            					)
			h_prev = h

		# add an out layer to reach the desired number of features 
		if h != nf:
			self.layers.append(nn.Sequential(
									nn.Linear(h, nf, bias=True),
									)
            					)
		self.num_layers = len(self.layers)

	def forward(self, x):
		for l in range(self.num_layers):
			#print(z.shape)
			x = self.layers[l](x)
		return x


class Generator_with_activation(nn.Module):
	def __init__(self, nz, nf=1000, num_hidden_layers=10):
		super(Generator_with_activation, self).__init__()

		self.layers = nn.ModuleList()

		num_hidden = np.linspace(nz, nf*1.5, num_hidden_layers)

		h_prev = nz
		for h in num_hidden:
			h = int(h)
			self.layers.append(nn.Sequential(
									nn.Linear(h_prev, h, bias=True),
									nn.LeakyReLU(),
									nn.BatchNorm1d(h))
            					)
			h_prev = h

		# add an out layer to reach the desired number of features 
		if h != nf:
			self.layers.append(nn.Sequential(
									nn.Linear(h, nf, bias=True),
									nn.Tanh())
            					)
		self.num_layers = len(self.layers)

	def forward(self, x):
		for l in range(self.num_layers):
			#print(z.shape)
			x = self.layers[l](x)
		return x




class Discriminator(nn.Module):
	def __init__(self, nz, nf=1000, num_hidden_layers=10):
		super(Discriminator, self).__init__()

		self.layers = nn.ModuleList()
		
		num_hidden = np.linspace(nz, nf*1.5, num_hidden_layers)

		h_prev = nz
		for h in num_hidden:
			h = int(h)
			self.layers.insert(0, nn.Sequential(
									nn.Linear(h, h_prev, bias=True),
									nn.LeakyReLU(),)
									#nn.BatchNorm1d(h_prev))
            					)
			h_prev = h

		# add an out layer to reach the desired number of features 
		if h != nf:
			self.layers.insert(0, nn.Sequential(
									nn.Linear(nf, h, bias=True),
									nn.LeakyReLU(),)
									#nn.BatchNorm1d(h))
            					)

		self.layers.append(nn.Sequential(
									nn.Linear(nz, 1, bias=True),
									#nn.Tanh()
									#nn.Sigmoid()
									)
            					)
		self.num_layers = len(self.layers)


	def forward(self, x):
		for l in range(self.num_layers):
			x = self.layers[l](x)
		return x
