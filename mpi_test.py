#import pandas as pd
#from torch import np # Torch wrapper for Numpy

import os

import torch
import torch.cuda
import torch.distributed as dist

dist.init_process_group(backend='mpi', world_size=4)
rank = dist.get_rank()
wsize = dist.get_world_size()
seed = 123
torch.manual_seed(seed)
print(('Hello from process {} (out of {})\n'.format(dist.get_rank(), dist.get_world_size())))


