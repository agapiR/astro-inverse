#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

from models import *

import matplotlib
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Used: " + str(device))

# set random seed
manual_seed = 17 #42
random.seed(manual_seed)
torch.manual_seed(manual_seed)


# In[4]:


## Data and Experiment Config

# Set Result Directory
# this folder's name should describe the configuration
result_folder = './GLO'
if not os.path.exists(result_folder):
    os.mkdir(result_folder)

# Load Data
data_folder = "./Data/Dataset_5_35"
training_data_complete = np.load(osp.join(data_folder, "spectra_complete_training.npy")).astype(np.float32) #"log_training.npy"
NLA_max = 205 
training_data = training_data_complete[:,:NLA_max] #manually delete very high wavelengths
TRAINING_DATA = training_data.shape[0]
FEATURE_SIZE = training_data.shape[1]
print("#Signal ", TRAINING_DATA)
print("#Measurements ", FEATURE_SIZE)


# Hyperparameters (could receive as arguments along with data/res directory)
LATENT_SIZE = 50

BATCH_SIZE = 100 #32,64,128
NUM_EPOCHS = 10000

LOG_EVERY = 10   #error logging
SAVE_EVERY = 100  #checkpoints and model saving

LR_MODEL = 0.5
LR_LATENT = 0.05


# In[5]:


## Model Definition

# ~~toy code without dataloaders

# Initialize model
d = 6 #num layers
netG = Generator(nz=LATENT_SIZE, nf=FEATURE_SIZE, num_hidden_layers=d).to(device)
model_size = sum(p.numel() for p in netG.parameters())
print("Generative Model size (total): {:.3f}M".format(model_size/1e6))


## Training Config


# GLO training requires projection of latent code in the l2 unit ball 
def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    z_l2_norm = torch.norm(z, p=2, dim=0).detach()
    if z_l2_norm.item() > 1:
        z = z.div(z_l2_norm.expand_as(z))
    return z

# placeholder variable for learnable latent code
Zin = torch.ones(BATCH_SIZE, LATENT_SIZE, device=device, requires_grad=True)
latent_codes = torch.randn(TRAINING_DATA, LATENT_SIZE, device=device)
# project onto unit ball (see GLO paper)
for idx in range(TRAINING_DATA):
    latent_codes[idx] = project_l2_ball(latent_codes[idx])

# Setup Adam optimizers for both model parameters and latent codes 
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
optimizer = optim.Adam([
        {'params': netG.parameters(), 'lr': LR_MODEL, 'weight_decay': 0.001}, 
        {'params': Zin, 'lr': LR_LATENT, 'weight_decay': 0.001}
    ])

# Learning rate decay
decayRate = 0.5
step = 2000 #decay every 2000 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step, gamma=decayRate)

# Training Loss
criterion = nn.MSELoss(reduction='sum')

# Print info
print("Model Architecture:")
print(netG)


# In[6]:


## Training

# Preliminaries : 
# 1) training requires restriction of latent codes in the l2 unit ball 
def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    z_l2_norm = torch.norm(z, p=2, dim=0).detach()
    if z_l2_norm.item() > 1:
        z = z.div(z_l2_norm.expand_as(z))
    return z


# Training Loop

# Lists to keep track of progress
losses = []

print("Starting Training Loop...")
netG.train()
# For each epoch
for epoch in range(NUM_EPOCHS):
    # Manually build batches for current epoch. For SDSS data toy code without dataloaders.
    ids = np.asarray(np.random.permutation(np.arange(TRAINING_DATA)))
    num_batches = TRAINING_DATA//BATCH_SIZE
    batches = np.split(ids, num_batches)
    # initialize loss accumulator 
    err = 0.0
    # For each batch
    for i, ids in enumerate(batches):   
        netG.zero_grad()
        # Format batch
        data = torch.tensor(training_data[ids], device=device)
        b_size = data.size(0)
        # get latent codes
        z = latent_codes[ids]
        # load latent codes in variable
        Zin.data = z
        # get instance reconstruction
        output = netG(Zin)
        # calculate loss
        loss = criterion(output,data)
        # run optimization step
        optimizer.zero_grad()  # zero the gradient buffer
        loss.backward()
        optimizer.step()
        
        # get the updated latent codes
        new_latent_codes = Zin.data
        #restrict into ball
        for idx in range(BATCH_SIZE):
            new_latent_codes[idx] = project_l2_ball(new_latent_codes[idx])
        latent_codes[ids] = new_latent_codes
        
        err += loss.item()
    
    # Learning rate decay
    lr_scheduler.step()

    # Reduce error by taking the average
    err = err/TRAINING_DATA
        
    # Save Losses for plotting later
    losses.append(err)
        
    # Output training stats
    if ((epoch+1) % LOG_EVERY == 0):
        print('Epoch[%d/%d]\tLoss: %.4f' % (epoch+1, NUM_EPOCHS, err))

    # Checkpoint generator
    if ((epoch+1) % SAVE_EVERY == 0):
        # save learned model so far
        torch.save(netG.state_dict(), osp.join(result_folder,"learned_generator_%d.pth")%(epoch+1))
        
# save the learned model and stats at the end of training 
torch.save(netG.state_dict(), osp.join(result_folder,"learned_model.pth"))
torch.save(latent_codes, osp.join(result_folder,"learned_latent_codes.pth"))
np.save(osp.join(result_folder,"losses.npy"), np.asarray(losses))


