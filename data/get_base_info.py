import os
import pickle
import numpy as np
from math import exp
from tqdm import tqdm
from itertools import product
import sys
sys.path.append("/home/fkli/Projects/RNADiffusion")
from utils.data import padding, seq2encoding


def parse_fasta(seq_dataset):
    seq_name_list = [seq_data["name"] for seq_data in seq_dataset]
    seq_raw_list = [seq_data["seq_raw"] for seq_data in seq_dataset]
    seq_len_list = [len(seq_data["seq_raw"]) for seq_data in seq_dataset]

    return seq_name_list, seq_raw_list, seq_len_list


def get_base_info_matrix(seq_raw, data_length, set_length):
    seq_onehot_pad = padding(seq2encoding(seq_raw), set_length)
    perm = list(product(np.arange(4), np.arange(4)))
    kronecker_product = np.zeros((16, set_length, set_length))
    # Calculate the Kronecker product of the sequence
    for n, cord in enumerate(perm):
        i, j = cord
        kronecker_product[n, :data_length, :data_length] = np.matmul(
            seq_onehot_pad[:data_length, i].reshape(-1, 1),
            seq_onehot_pad[:data_length, j].reshape(1, -1),
        )
    pair_probability = np.zeros((1, set_length, set_length))
    pair_probability[0, :data_length, :data_length] = creatmat(
        seq_onehot_pad[:data_length, :]
    )
    return np.concatenate((kronecker_product, pair_probability), axis=0)


# Algorithm for generating RNA secondary structure pair probability
def creatmat(data):
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(list(data[i - add]), list(data[j + add]))
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(list(data[i + add]), list(data[j - add]))
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i], [j]] = coefficient
    return mat


def paired(x, y):
    if x == [1, 0, 0, 0] and y == [0, 1, 0, 0]:
        return 2
    elif x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
        return 3
    elif x == [0, 0, 0, 1] and y == [0, 1, 0, 0]:
        return 0.8
    elif x == [0, 1, 0, 0] and y == [1, 0, 0, 0]:
        return 2
    elif x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
        return 3
    elif x == [0, 1, 0, 0] and y == [0, 0, 0, 1]:
        return 0.8
    else:
        return 0


def Gaussian(x):
    return exp(-0.5 * (x * x))


if __name__ == "__main__":
    train_data_path = "/home/fkli/RNAdata/bpRNA_lasted/batching/train"
    test_data_path = "/home/fkli/RNAdata/bpRNA_lasted/batching/test"
    val_data_path = "/home/fkli/RNAdata/bpRNA_lasted/batching/val"
    target_data_path = "/home/fkli/RNAdata/bpRNA_lasted/binning"
    path_list = [train_data_path, test_data_path, val_data_path]

    for dpath in path_list:
        for k in tqdm(range(0, len(os.listdir(dpath)))):
            c = dpath.split("/")[-1]
            source_path = os.path.join(dpath, os.listdir(dpath)[k])
            print(f"source path is {source_path}")

            with open(source_path, "rb") as f:
                data = pickle.load(f, encoding="bytes")

            print(f"data nums is {len(data)}")

            _, seq_raw_list, seq_len_list = parse_fasta(data)
            seq_max_len = max(seq_len_list)
            set_max_len = (seq_max_len // 80 + int(seq_max_len % 80 != 0)) * 80
            for seq in tqdm(data):
                seq["base_info"] = np.array(1)
                # seq["base_info"] = get_base_info_matrix(
                #     seq["seq_raw"], seq["length"], set_max_len
                # )

            target_path = os.path.join(target_data_path, c, os.listdir(dpath)[k])
            print(f"target path is {target_path}\n")
            if not os.path.exists(os.path.join(target_data_path, c)):
                os.makedirs(os.path.join(target_data_path, c))

            with open(target_path, "wb") as f:
                pickle.dump(data, f)
