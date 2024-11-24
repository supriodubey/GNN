# List of constants and some common functions

import numpy as np
import torch
import random

# Random seeds
torch.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

# use GPUs if available

if torch.cuda.is_available():
    # print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# Reduced Hubble constant
hred = 0.7

# Box size in comoving kpc/h 
boxsize = 1.e6

# Validation and test size
valid_size, test_size = 0.15, 0.15

# Batch size
batch_size = 32
