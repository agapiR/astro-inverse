#!/usr/bin/env python

import sys 
import os
import numpy as np

value_list = []
fp = open("GLO_eval_log.txt")
line = fp.readline()
while line:
	if line.startswith("Reconstruction of id"):
		tokens = line.split()
		value = int(tokens[3])
		value_list.append(value)
	
	line = fp.readline()

id_array = np.array(value_list)
np.save("test_ids.npy", id_array)
	
fp.close()

