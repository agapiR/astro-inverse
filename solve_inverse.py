"""
Function that allows solving linear inverse problems
"""

import time
import os
import os.path as osp
import sys
import io
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable


# GLO training requires restriction of latent code in the l2 unit ball 
def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    z_l2_norm = torch.norm(z, p=2, dim=0).detach()
    if z_l2_norm.item() > 1:
        z = z.div(z_l2_norm.expand_as(z))
    return z


def solve_inverse(model, measurements, transform, latent_size, device, max_epochs=10000, lr=1e-3, 
					optm="SGD", criterion="mse", regularization_lambda=0, verbose=True):

	model.eval()

	# initialize learnable latent code
	latent_z =torch.randn(latent_size, device=device, requires_grad=True)
	# project initial latent code onto unit ball
	if regularization_lambda==0:
		latent_z.data = project_l2_ball(latent_z.data)

	if optm=="SGD":
		optimizer = optim.SGD([latent_z], lr=lr, momentum=0.9)
	elif optm=="adam":
		optimizer = optim.Adam([latent_z], lr=lr)
	else:
		print("Optimizer option not supported.")
		return

	if criterion=="mse":
		criterion = nn.MSELoss() 
	#elif criterion=="l1":
	else:
		criterion = nn.MSELoss()

	for epoch in range(max_epochs):
		# get reconstruction
		output = model(latent_z.unsqueeze(0))

		# calculate loss
		output_after_transform = torch.matmul(transform,output.squeeze(0))
		loss = criterion(output_after_transform, measurements) 
		if regularization_lambda!=0:
			regularization_term = regularization_lambda*torch.norm(latent_z, p=2, dim=0).pow(2).unsqueeze(0)
			loss = loss + regularization_term

		# logging
		if (epoch+1)%100==0 and verbose:
			print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, max_epochs, loss.item()))

		# run optimization step
		optimizer.zero_grad()  # zero the gradient buffer
		loss.backward()
		optimizer.step()

		if regularization_lambda==0:
			latent_z.data = project_l2_ball(latent_z.data)

	return latent_z