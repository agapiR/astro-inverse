#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
manual_seed = 13
random.seed(manual_seed)
torch.manual_seed(manual_seed)


# In[2]:


## Data and Experiment Config

# Set Result Directory
# this folder's name should describe the configuration
result_folder = './logWGAN'
if not os.path.exists(result_folder):
    os.mkdir(result_folder)

# Load Data
data_folder = "./Data/Dataset_5_35"
training_data_complete = np.load(osp.join(data_folder, "log_training.npy")).astype(np.float32) #"spectra_complete_training.npy"
NLA_max = 205 
training_data = training_data_complete[:,:NLA_max] #manually delete very high wavelengths
TRAINING_DATA = training_data.shape[0]
FEATURE_SIZE = training_data.shape[1]
print("#Signal ", TRAINING_DATA)
print("#Measurements ", FEATURE_SIZE)


# Hyperparameters (could receive as arguments along with data/res directory)
LATENT_SIZE = 50

BATCH_SIZE = 100 #32,64,128
NUM_EPOCHS = 15000
# For WGAN, update the Discriminator more often
D_ITERS = 5

LOG_EVERY = 10   #error logging
SAVE_EVERY = 100  #checkpoints and model saving

LR_D = 0.001
LR_G = 0.001


# In[3]:


## Model Definition

# ~~toy code without dataloaders

# Initialize model
d = 6 #num layers
netG = Generator(nz=LATENT_SIZE, nf=FEATURE_SIZE, num_hidden_layers=d).to(device)
netD = Discriminator(nz=LATENT_SIZE, nf=FEATURE_SIZE, num_hidden_layers=d).to(device)
model_size = sum(p.numel() for p in netG.parameters())
print("Generative Model size (total): {:.3f}M".format(model_size/1e6))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#netG.apply(weights_init)
#netD.apply(weights_init)


## Training Config


# Establish convention for real and fake labels during training
real_label = -1
fake_label = 1
#D_label_smoothing = 0.005, for Vanilla GAN
print("Label convention:")
print("real: "+ str(real_label) + "  fake: " + str(fake_label))

# Setup RMSprop optimizers for both G and D
optimizerD = optim.RMSprop(netD.parameters(), lr=LR_D)
optimizerG = optim.RMSprop(netG.parameters(), lr=LR_G)

# Learning rate decay
decayRate = 0.5
step = 3000 #decay every 3000 epochs
lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer=optimizerD, step_size=step, gamma=decayRate)
lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer=optimizerG, step_size=step, gamma=decayRate)

# Wasserstein Loss
def Wasserstein_Loss(y_pred, y_true):
    return mean(y_true * y_pred)

# Vanilla GAN loss
#criterion = nn.BCEWithLogitsLoss(reduction='mean'),nn.SoftMarginLoss(reduction='mean'),nn.BCELoss(reduction='mean')
sigmoid_response = nn.Sigmoid() # for training output

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(30, LATENT_SIZE, device=device)

print("Architecture:")
print(netG)


# In[4]:


## Training


# Training Loop

# Lists to keep track of progress
G_losses = []
D_losses = []

print("Starting Training Loop...")
netG.train()
netD.train()
# For each epoch
for epoch in range(NUM_EPOCHS):
    # Manually build batches for current epoch. Toy code without dataloaders.
    ids = np.asarray(np.random.permutation(np.arange(TRAINING_DATA)))
    num_batches = TRAINING_DATA//BATCH_SIZE
    batches = np.split(ids, num_batches)
    # For each batch
    for i, ids in enumerate(batches):   
        ############################
        # (1) Update D network: maximize  -(fake label * Avg Score for Fake Batch + real label * Avg Score for Real Batch)
        ###########################

        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_batch = torch.tensor(training_data[ids], device=device)
        b_size = real_batch.size(0)
        #label_real = torch.full((b_size,), real_label, device=device)-D_label_smoothing*torch.randn((b_size,), device=device)
        # Forward pass real batch through D
        output_real = netD(real_batch).view(-1)
        # Calculate loss on all-real batch
        errD_real =  real_label*torch.mean(output_real)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # Update D
        optimizerD.step()
        # Weight clipping for Discriminator Network
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)
        #Get Discriminator's avg classification response for all-real batch
        D_x = sigmoid_response(output_real).mean().item()

        ## Train with all-fake batch
        netD.zero_grad()
        # Generate batch of latent vectors
        noise = torch.randn(b_size, LATENT_SIZE, device=device)
        # Generate fake batch with G
        fake_batch = netG(noise)
        #label_fake = torch.full((b_size,), fake_label, device=device)-D_label_smoothing*torch.randn((b_size,), device=device)
        # Classify all fake batch with D
        output_fake = netD(fake_batch.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = fake_label*torch.mean(output_fake)
        # Calculate the gradients for this batch
        errD_fake.backward()
        # Update D
        optimizerD.step()
        # Weight clipping for Discriminator Network
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)
        #Get Discriminator's avg classification response for all-fake batch
        D_G_z1 = sigmoid_response(output_fake).mean().item()
        # Get current error of D: Add the errors from the all-real and all-fake batches
        errD = errD_real.item() + errD_fake.item()

        if (i+1) % D_ITERS == 0:
            ############################
            # (2) Update G network: maximize fake label * Avg Score for Generated Batch
            ###########################
            netG.zero_grad()
            # Generate batch of latent vectors
            noise = torch.randn(b_size, LATENT_SIZE, device=device)
            # Generate fake image batch with G
            fake_batch = netG(noise)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output_G = netD(fake_batch).view(-1)
            # Calculate G's loss based on this output
            errG = -fake_label*torch.mean(output_G)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizerG.step()
            #Get Discriminator's avg classification response for newly generated batch
            D_G_z2 = sigmoid_response(output_G).mean().item()
            # Get current error of G
            errG = errG.item()
        
        
        # Output training stats, for last batch
        if ((epoch==0 or (epoch+1) % LOG_EVERY == 0)) and (i == num_batches-1):
            print('Epoch[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (epoch+1, NUM_EPOCHS, errD, errG, D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG)
            D_losses.append(errD)

    # Check how the generator is doing by saving G's output on fixed_noise
    # plus Checkpoint generator
    if ((epoch+1) % SAVE_EVERY == 0):
        with torch.no_grad():
            fake_batch_checkpoint = netG(fixed_noise).detach().cpu()
        # save checkpoints
        #torch.save(fake, osp.join(result_folder,"checkpoint_sample_%d.pth")%(epoch+1))
        # save learned model so far
        torch.save(netG.state_dict(), osp.join(result_folder,"learned_generator_%d.pth")%(epoch+1))
        
# save the learned model and stats at the end of training 
torch.save(netG.state_dict(), osp.join(result_folder,"learned_model.pth"))
np.save(osp.join(result_folder,"G_losses.npy"), np.asarray(G_losses))
np.save(osp.join(result_folder,"D_losses.npy"), np.asarray(D_losses))


# In[ ]:




