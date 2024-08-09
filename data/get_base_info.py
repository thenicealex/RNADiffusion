import os
import pickle
import numpy as np
from math import exp
from tqdm import tqdm
from itertools import product
import sys
sys.path.append("/lustre/home/fkli/Projects/RNADiffusion")
from utils.data import padding, seq2encoding


def parse_fasta(seq_dataset):
    seq_names = [seq_data["name"] for seq_data in seq_dataset]
    seq_raws = [seq_data["seq_raw"] for seq_data in seq_dataset]
    seq_lengths = [len(seq_data["seq_raw"]) for seq_data in seq_dataset]
    return seq_names, seq_raws, seq_lengths


def get_base_info_matrix(seq_raw, data_length, set_length):
    seq_onehot_padded = padding(seq2encoding(seq_raw), set_length)
    kronecker_product = compute_kronecker_product(seq_onehot_padded, data_length, set_length)
    pair_probability = compute_pair_probability(seq_onehot_padded, data_length, set_length)
    return np.concatenate((kronecker_product, pair_probability), axis=0)


def compute_kronecker_product(seq_onehot_padded, data_length, set_length):
    perm = list(product(np.arange(4), np.arange(4)))
    kronecker_product = np.zeros((16, set_length, set_length))
    for n, (i, j) in enumerate(perm):
        kronecker_product[n, :data_length, :data_length] = np.matmul(
            seq_onehot_padded[:data_length, i].reshape(-1, 1),
            seq_onehot_padded[:data_length, j].reshape(1, -1),
        )
    return kronecker_product


def compute_pair_probability(seq_onehot_padded, data_length, set_length):
    pair_probability = np.zeros((1, set_length, set_length))
    pair_probability[0, :data_length, :data_length] = create_pair_probability_matrix(seq_onehot_padded[:data_length, :])
    return pair_probability


def create_pair_probability_matrix(data):
    matrix = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = calculate_coefficient(data, i, j)
            matrix[i, j] = coefficient
    return matrix


def calculate_coefficient(data, i, j):
    coefficient = 0
    for add in range(30):
        if i - add >= 0 and j + add < len(data):
            score = paired(data[i - add], data[j + add])
            if score == 0:
                break
            coefficient += score * gaussian(add)
        else:
            break
    if coefficient > 0:
        for add in range(1, 30):
            if i + add < len(data) and j - add >= 0:
                score = paired(data[i + add], data[j - add])
                if score == 0:
                    break
                coefficient += score * gaussian(add)
            else:
                break
    return coefficient


def paired(x, y):
    pair_scores = {
        ((1, 0, 0, 0), (0, 1, 0, 0)): 2,
        ((0, 0, 0, 1), (0, 0, 1, 0)): 3,
        ((0, 0, 0, 1), (0, 1, 0, 0)): 0.8,
        ((0, 1, 0, 0), (1, 0, 0, 0)): 2,
        ((0, 0, 1, 0), (0, 0, 0, 1)): 3,
        ((0, 1, 0, 0), (0, 0, 0, 1)): 0.8,
    }
    return pair_scores.get((tuple(x), tuple(y)), 0)


def gaussian(x):
    return exp(-0.5 * (x * x))


def process_data_paths(data_paths, target_data_path):
    for data_path in data_paths:
        process_data_path(data_path, target_data_path)


def process_data_path(data_path, target_data_path):
    for file_name in tqdm(os.listdir(data_path)):
        category = os.path.basename(data_path)
        source_path = os.path.join(data_path, file_name)
        print(f"Processing source path: {source_path}")

        with open(source_path, "rb") as file:
            data = pickle.load(file, encoding="bytes")

        print(f"Number of data entries: {len(data)}")

        _, _, seq_lengths = parse_fasta(data)
        max_seq_length = max(seq_lengths)
        set_length = ((max_seq_length // 80) + int(max_seq_length % 80 != 0)) * 80

        for seq in data:
            seq["base_info"] = get_base_info_matrix(seq["seq_raw"], seq["length"], set_length)

        target_dir = os.path.join(target_data_path, category)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, file_name)
        print(f"Saving to target path: {target_path}\n")

        with open(target_path, "wb") as file:
            pickle.dump(data, file)


if __name__ == "__main__":
    train_data_path = "/lustre/home/fkli/RNAdata/bpRNA_lasted/batching/train"
    test_data_path = "/lustre/home/fkli/RNAdata/bpRNA_lasted/batching/test"
    val_data_path = "/lustre/home/fkli/RNAdata/bpRNA_lasted/batching/val"
    target_data_path = "/lustre/home/fkli/RNAdata/bpRNA_lasted/data"
    data_paths = [train_data_path, test_data_path, val_data_path]

    process_data_paths(data_paths, target_data_path)