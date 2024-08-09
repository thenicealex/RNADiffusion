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

train_data_path = "/lustre/home/fkli/RNAdata/bpRNA_lasted/batching/train"


if __name__ == "__main__":
    train_dataset = RNADataset(train_data_path, upsampling=True)

    train_dataloader = DataLoader( train_dataset, batch_size=1, shuffle=True, collate_fn=train_dataset.collate_fn)
    for _, (set_max_len, name_tensor, raw_seq_list, length_tensor, contact_tensor, base_info_tensor, padded_encoded_seq_tensor) in enumerate(train_dataloader):
        print(set_max_len)
        print(name_tensor.shape)
        print(raw_seq_list)
        print(length_tensor.shape)
        print(contact_tensor.shape)
        print(base_info_tensor.shape)
        print(padded_encoded_seq_tensor.shape)
        break
    # for file in os.listdir(train_data_path):
    #     with open(os.path.join(train_data_path, file), "rb") as f:
    #         data = pickle.load(f)
    #     print(data)
    #     break
