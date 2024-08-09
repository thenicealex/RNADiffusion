import collections
from functools import partial
import os
from itertools import product
from math import exp
import pickle
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils.data import padding
from data.data_generator import RNADataset
from torch.utils.data import DataLoader

data_path = "/home/fkli/RNAdata/bpRNA_lasted.pkl"

if __name__ == "__main__":
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    for k in data.keys():
        print(data[k][0])
        break