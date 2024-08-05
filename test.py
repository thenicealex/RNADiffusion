import os
from itertools import product
from math import exp
import pickle
import numpy as np
import torch
from tqdm import tqdm
from utils.data import padding

data_path = "/home/fkli/Projects/RNADiffusion/dataset/dataset.pkl"
test_path = "/home/fkli/Projects/RNADiffusion/data/test"


if __name__ == "__main__":
    print("Loading data from {}".format(data_path))