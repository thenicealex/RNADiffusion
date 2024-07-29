from itertools import product
from math import exp
import os
import pickle as cPickle
import numpy as np
from tqdm import tqdm

seq_to_onehot_dict = {
    "A": np.array([1, 0, 0, 0]),
    "U": np.array([0, 1, 0, 0]),  # T or U
    "C": np.array([0, 0, 1, 0]),
    "G": np.array([0, 0, 0, 1]),
    "N": np.array([0, 0, 0, 0]),
    "R": np.array([1, 0, 0, 1]),
    "Y": np.array([0, 1, 1, 0]),
    "K": np.array([0, 1, 0, 1]),
    "M": np.array([1, 0, 1, 0]),
    "S": np.array([0, 0, 1, 1]),
    "W": np.array([1, 1, 0, 0]),
    "B": np.array([0, 1, 1, 1]),
    "D": np.array([1, 1, 0, 1]),
    "H": np.array([1, 1, 1, 0]),
    "V": np.array([1, 0, 1, 1]),
    "_": np.array([0, 0, 0, 0]),
    "~": np.array([0, 0, 0, 0]),
    ".": np.array([0, 0, 0, 0]),
    "P": np.array([0, 0, 0, 0]),
    "I": np.array([0, 0, 0, 0]),
    "X": np.array([0, 0, 0, 0]),
}


def get_base_info_list(seq_dataset):
    _, seq_raw_list, seq_len_list = parse_fasta(seq_dataset)

    seq_max_len = max(seq_len_list)
    set_max_len = (seq_max_len // 80 + int(seq_max_len % 80 != 0)) * 80

    seq_encoding_list = [seq2encoding(seq) for seq in seq_raw_list]
    seq_encoding_pad_list = [padding(seq, set_max_len) for seq in seq_encoding_list]

    base_info_list = [
        get_base_info_matrix(x[0], x[1], set_max_len)
        for x in tqdm(
            zip(seq_encoding_pad_list, seq_len_list), total=len(seq_encoding_pad_list)
        )
    ]
    # base_info_list = torch.tensor(np.stack(base_info_list, axis=0)).float()
    return base_info_list


def parse_fasta(seq_dataset):
    seq_name_list = [seq_data["name"] for seq_data in seq_dataset]
    seq_raw_list = [seq_data["seq_raw"] for seq_data in seq_dataset]
    seq_len_list = [len(seq_data["seq_raw"]) for seq_data in seq_dataset]

    return seq_name_list, seq_raw_list, seq_len_list


def get_base_info_matrix(seq_onehot_pad, data_length, set_length):
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
    I = np.concatenate((kronecker_product, pair_probability), axis=0)

    return I


def seq2encoding(seq):
    encoding = list()
    for char in seq:
        encoding.append(seq_to_onehot_dict[char])
    return np.array(encoding)


def padding(data_array, maxlen):
    a, _ = data_array.shape
    # np.pad(array, ((before_1,after_1),……,(before_n,after_n),module)
    return np.pad(data_array, ((0, maxlen - a), (0, 0)), "constant")


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
    data_path = "/home/fkli/Projects/RNADiffusion/dataset"
    train_data_path = "/home/fkli/Projects/DiffRNA/datasets/batching/train"
    test_data_path = "/home/fkli/RNAdata/RNAcmap2/batching/test"
    val_data_path = "/home/fkli/Projects/DiffRNA/datasets/batching/val"
    target_data_path = "/home/fkli/RNAdata/RNAcmap2/datasets/test"
    path_list = [test_data_path]

    for dpath in path_list:
        for k in range(0, len(os.listdir(dpath))):
            c = dpath.split("/")[-1]
            source_path = os.path.join(dpath, os.listdir(dpath)[k])
            print(f"source path is {source_path}")

            with open(source_path, "rb") as f:
                data = cPickle.load(f, encoding="bytes")

            print(f"data nums is {len(data)}")

            mat = get_base_info_list(data)
            for j, seq in enumerate(data):
                seq["base_info"] = mat[j]

            target_path = os.path.join(target_data_path, c, os.listdir(dpath)[k])
            print(f"target path is {target_path}\n")
            if not os.path.exists(os.path.join(target_data_path, c)):
                os.makedirs(os.path.join(target_data_path, c))

            with open(target_path, "wb") as f:
                cPickle.dump(data, f)