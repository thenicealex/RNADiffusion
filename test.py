import os
from itertools import product
from math import exp
import pickle as cPickle
import numpy as np
import torch
from tqdm import tqdm

data_path = "/home/fkli/Projects/RNADiffusion/dataset/dataset.pkl"

test_path = "/home/fkli/Projects/RNADiffusion/data/test"


if __name__ == "__main__":
    with open(data_path, "rb") as f:
        datasets = cPickle.load(f, encoding="bytes")
    print(f"keys in dataset.pkl is {datasets.keys()}")
    for key in datasets.keys():
        print(datasets[key][0])