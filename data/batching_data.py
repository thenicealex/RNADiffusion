# -*- coding: utf-8 -*-
import os
import time
import numpy
import pickle as cPickle
from os.path import join

from tqdm import tqdm

def load_data(path):
    with open(path, "rb") as f:
        load_data = cPickle.load(f)
    return load_data

if __name__=="__main__":
    train_data_path = "/home/fkli/RNAdata/bpRNA_lasted/binning/train"
    test_data_path = "/home/fkli/RNAdata/bpRNA_lasted/binning/test"
    val_data_path = "/home/fkli/RNAdata/bpRNA_lasted/binning/val"
    target_data_path = "/home/fkli/RNAdata/bpRNA_lasted/batching"
    path_list = [train_data_path,val_data_path, test_data_path]

    for dpath in path_list:
        for k in tqdm(range(0, len(os.listdir(dpath)))):
            c = dpath.split("/")[-1]
            source_path = os.path.join(dpath, os.listdir(dpath)[k])

            set_max_len = int(source_path.split("/")[-1].split("_")[-1].split(".")[0])

            if set_max_len == 80:
                set_batch = 128
            elif set_max_len == 160:
                set_batch = 64
            elif set_max_len > 160 and set_max_len <= 320:
                set_batch = 16
            elif set_max_len > 320 and set_max_len <= 640:
                set_batch = 4
            elif set_max_len > 640 and set_max_len <= 1280:
                set_batch = 2
            else:
                set_batch = 1
            data = load_data(source_path)
            seq_len = len(data)
            for i in range(0, seq_len, set_batch):
                if i + set_batch > seq_len:
                    batch_data = data[i:]
                else:
                    batch_data = data[i:i+set_batch]

                if not os.path.exists(join(target_data_path, c)):
                    os.makedirs(join(target_data_path, c))

                target_path = join(target_data_path, c, f"{i}_{i+set_batch}_{set_max_len}.pkl")
                with open(target_path, "wb") as f:
                    cPickle.dump(batch_data, f)
                print(f"write {target_path} done")