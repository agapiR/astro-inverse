import time
import os
import os.path as osp
import sys, argparse
import io
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from astroML.datasets import sdss_corrected_spectra

from models import *
from solve_inverse import *
from statistics import mean
from math import sqrt
from bisect import bisect


os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Used: " + str(device))

print(torch.get_num_threads())


### Config
parser = argparse.ArgumentParser(description='Qualitative evaluation')

parser.add_argument("-dir", action="store", dest="res_dir", default="./trained_models", help="Directory of models to be evaluated.", type=str)

parser.add_argument("-s", action="store", dest="random_seed", default="42", help="Random seed for initialization.", type=int)
parser.add_argument("-d", action="store", dest="data", default="log", help="Raw or Logarithmic data.", type=str)
parser.add_argument("-m", action="store", dest="model", default="glo", help="GLO or GAN model.", type=str)
parser.add_argument("-i", action="store", dest="iter", default="5000", help="Saved iteration of trained model.", type=int)
parser.add_argument("-z", action="store", dest="latent", default="50", help="Dimensionality of the Latent Space.", type=int)
parser.add_argument("-l", action="store", dest="layers", default="6", help="Number of layers of the deep generative model.", type=int)

parser.add_argument("-lambda", action="store", dest="regularization_lambda", default=0, help="Regularization Hyperparameter of the reconstruction procedure. If zero it uses projection instead.", type=float)
parser.add_argument("-lr", action="store", dest="lr", default="0.05", help="Learning Rate of the reconstruction procedure.", type=float)

parser.add_argument("-n", action="store", dest="sample_size", default="100", help="Size of sample used for evaluation.", type=int)
parser.add_argument("-v", action="store_true", dest="verbose", default=False, help="To print out helpful comments")

arg_results = parser.parse_args()

res_dir = arg_results.res_dir
DATA = arg_results.data
MODEL = arg_results.model
ITER_SAVED = int(arg_results.iter)
RANDOMSEED = int(arg_results.random_seed)
REG_LAMBDA = float(arg_results.regularization_lambda)
LR = float(arg_results.lr)
LATENT_SIZE = int(arg_results.latent)
d = int(arg_results.layers)
SAMPLE = int(arg_results.sample_size)
verbose = arg_results.verbose


# set random seed
random.seed(RANDOMSEED)
torch.manual_seed(RANDOMSEED)


## Data 

dataset_dir = "./Data/Dataset_5_35"

training_5_35 = np.load(osp.join(dataset_dir,"spectra_complete_training_5_35.npy"))
test_5_35 = np.load(osp.join(dataset_dir,"spectra_complete_testing_5_35.npy"))

training_complete = np.load(osp.join(dataset_dir,"spectra_complete_training.npy"))
test_complete = np.load(osp.join(dataset_dir,"spectra_complete_testing.npy"))

training_log = np.load(osp.join(dataset_dir,"log_training.npy"))
test_log = np.load(osp.join(dataset_dir,"log_test.npy"))

training_reduced = np.load(osp.join(dataset_dir,"spectra_reduced_training.npy"))
test_reduced = np.load(osp.join(dataset_dir,"spectra_reduced_testing.npy"))

wavelengths_complete_5_35 = np.load(osp.join(dataset_dir,"wavelenths_complete_5_35.npy"))
wavelengths_complete = np.load(osp.join(dataset_dir,"wavelenths_complete.npy"))
wavelengths_reduced = np.load(osp.join(dataset_dir,"wavelenths_reduced.npy"))


#manually delete very high wavelengths, hardcoded correction
NLA_max = 205
if DATA=="log":
	training = training_log[:,:NLA_max] #training_log for logGLO
	test = test_log[:,:NLA_max] #test_log for logGLO
else:
	training = training_complete[:,:NLA_max]
	test = test_complete[:,:NLA_max]

wavelengths = wavelengths_complete[:NLA_max]


TRAINING_DATA = training.shape[0]
TEST_DATA = test.shape[0]
FEATURE_SIZE = training.shape[1]
REDUCED_DATA = test_reduced.shape[0]
REDUCED_FEATURES = test_reduced.shape[1]
NLA = wavelengths.shape[0]
NLA_reduced = wavelengths_reduced.shape[0]



## Model
if DATA=="log":
		model = DATA + MODEL.upper()
else:
		model = MODEL.upper()
result_folder = osp.join(res_dir,model)


netG = Generator(nz=LATENT_SIZE, nf=FEATURE_SIZE, num_hidden_layers=d)
netG.load_state_dict(torch.load(osp.join(result_folder,"learned_generator_" + str(ITER_SAVED) + ".pth")))
netG.eval()



## Problem Setup

inverse_problem = "superresolution_real"
res_dir = osp.join(result_folder, model + "_quantitative_real")
if not osp.exists(res_dir):
    os.mkdir(res_dir)
    
## used to restrict the reconstruction into the 5-35 range ([:, idx_low:(idx_high+1)])
low_wv = 5
high_wv = 35
idx_low = bisect(wavelengths, low_wv)
idx_high = bisect(wavelengths, high_wv)

## Construct The A matrix for the real superresolution application
A_superres_real = torch.zeros((NLA_reduced, NLA), device=device)

for i,wv in enumerate(wavelengths_reduced):
    idx = bisect(wavelengths, wv)
    if idx>0 and idx<NLA:
        A_superres_real[i][idx-1] = 0.5
        A_superres_real[i][idx] = 0.5


## Chi^2 measure
def chi_2_measure(x, y, ax=0):
	sigma_x = 0.1*x 
	sigma_y = 0.1*y
	num = np.square(x - y)
	denom = np.square(sigma_x) + np.square(sigma_y)
	return np.divide(num, denom).sum(axis=ax)

## Mean Squared Error measure
def mse(x, y, ax=0):
	return np.square(x - y).mean(axis=ax)


## Solve the inverse problem for a large sample (quantitative evaluation with chi^2 measure)

ids = random.sample(range(0,REDUCED_DATA), SAMPLE)
rec_errors = []

for cnt,i in enumerate(ids):
	if verbose:
		print("Reconstruction of id " + str(i))
	#collect measurements
	y = torch.tensor(test_reduced[i], device=device)
	if DATA=="log":
		y= torch.log10(y)
	# real super-res application
	A = A_superres_real
	# solve inverse
	z = solve_inverse(netG, y, A, LATENT_SIZE, device, max_epochs=1000, lr=LR, 
	                  optm="adam", regularization_lambda=REG_LAMBDA, verbose=verbose)
	inv = netG(z.unsqueeze(0)).squeeze(0).detach()
	#get original signal 
	original = test[i]
	original_5_35 = test_5_35[i]
	#restrict the reconstruction into the 5 - 35 range
	inv = inv.numpy()
	inv_5_35 = inv[idx_low:(idx_high+1)]
	#compute recostruction error
	err = mse(original_5_35, inv_5_35)
	rec_errors.append(err)
    #save run: [ original | reconstruction ], stack to the previous save
	run_res = np.concatenate((original, inv))
	if cnt==0:
		runs = run_res
	else:
		runs = np.vstack((runs, run_res))
    
    
## Save Test Runs and Errors
run_config = "_lambda_" + str(REG_LAMBDA) + "_lr_" + str(LR)
np.save(osp.join(res_dir, "sample_runs" + run_config + ".npy"), runs)
np.save(osp.join(res_dir, "reconstruction_errors" + run_config + ".npy"), np.asarray(rec_errors))