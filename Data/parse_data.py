#!/usr/bin/env python

import sys 
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# manually delete first few rows, keep just the spectra data (delete NLA NM and wavelenghts)

#tr -d "\n\r" < bayes_library1_only_data.txt > bayes_library1_only_data_serial.txt
#tr -d "\n\r" < GANs_z_2_output_only_data.txt > GANs_z_2_output_only_data_serial.txt

#number of wavelenths
NLA = 223
NLA_reduced = 20
#number of spectra
NM = 10000




data = np.loadtxt('bayes_library1_only_data_serial.txt')
data = data.reshape([NM, NLA])
np.save('spectra_complete', data)




data_reduced = np.loadtxt('GANs_z_2_output_only_data_serial.txt')
data_reduced = data_reduced.reshape([NM, NLA_reduced])
np.save('spectra_reduced', data_reduced)


