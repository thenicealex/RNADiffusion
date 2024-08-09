import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle

label_dict = {
    ".": np.array([1, 0, 0]),
    "(": np.array([0, 1, 0]),
    ")": np.array([0, 0, 1]),
}

seq_dict = {
    "A": np.array([1, 0, 0, 0]),
    "U": np.array([0, 1, 0, 0]),  # T or U
    "C": np.array([0, 0, 1, 0]),
    "G": np.array([0, 0, 0, 1]),
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
    "N": np.array([0, 0, 0, 0]),
    "_": np.array([0, 0, 0, 0]),
    "~": np.array([0, 0, 0, 0]),
    ".": np.array([0, 0, 0, 0]),
    "P": np.array([0, 0, 0, 0]),
    "I": np.array([0, 0, 0, 0]),
    "X": np.array([0, 0, 0, 0]),
}

char_dict = {0: "A", 1: "U", 2: "C", 3: "G"}
chars = (
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+{}|:"<>?1234567890'
)
itos = {i: c for i, c in enumerate(chars)}
stoi = {c: i for i, c in enumerate(chars)}
encode_name = lambda x: [stoi[c] for c in x]
decode_name = lambda x: "".join([itos[i] for i in x])


def padding(array, maxlen, constant_values=-1, axis=0):
    if array.ndim == 1:
        return np.pad(
            array, (0, maxlen - len(array)), "constant", constant_values=constant_values
        )

    pad_width = [(0, maxlen - array.shape[0]), (0, 0)]
    if axis == 1:
        pad_width[1] = (0, maxlen - array.shape[1])

    # np.pad(array, ((before_1,after_1),……,(before_n,after_n)),module)
    return np.pad(array, pad_width, "constant")


def seq2encoding(seq: str):
    return np.array([seq_dict[char] for char in seq])


def encoding2seq(arr):
    return "".join("N" if sum(row) == 0 else char_dict[np.argmax(row)] for row in arr)


def contact_map_masks(data_lens, matrix_rep):
    n_seq = len(data_lens)
    assert matrix_rep.shape[0] == n_seq
    
    # Convert data_lens to a NumPy array
    lengths = np.array([int(l.cpu().numpy()) for l in data_lens])
    
    for i, l in range(lengths):
        matrix_rep[i, :l, :l] = 1
        
    return matrix_rep


def load_data(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    if file_path.endswith(".js") or file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
    elif (
        file_path.endswith(".pkl")
        or file_path.endswith(".pickle")
        or file_path.endswith(".cPickle")
    ):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    return data


# return index of contact pairing, index start from 0
def get_pairings(data):
    rnadata1 = list(data.loc[:, 0].values)
    rnadata2 = list(data.loc[:, 4].values)
    rna_pairs = list(zip(rnadata1, rnadata2))
    rna_pairs = list(filter(lambda x: x[1] > 0, rna_pairs))
    rna_pairs = (np.array(rna_pairs) - 1).tolist()
    return rna_pairs


# generate .dbn format
def generate_label_dot_bracket(data):
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] <= 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return "".join(rnastructure)


# extract the pseudoknot index given the data
def extract_pseudoknot(data):
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]:
                print(i, j)
                break


def find_pseudoknot(data):
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    flag = False
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]:
                flag = True
                break
    return flag


def struct_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: label_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)


def plot__fig(
    name: str, data: list, sample_number: int, shortest_seq: int, longest_seq: int
):
    plot_data = np.array(data)
    plt.figure(figsize=(16, 10))
    n, bins, patches = plt.hist(
        plot_data, bins=10, rwidth=0.8, align="left", color="b", edgecolor="white"
    )
    for i in range(len(n)):
        plt.text(
            bins[i], n[i] * 1.02, int(n[i]), fontsize=12, horizontalalignment="center"
        )
    plt.title(
        f"{name} data distribution(Total number:{sample_number},shortest_seq:{shortest_seq},longest_seq:{longest_seq})"
    )
    plt.grid()
    # plt.legend()
    fig_name = f"{name}" + ".png"
    if not os.path.exists(fig_name):
        plt.savefig(fig_name)
    # plt.show()


def draw_dis(
    name: str, data: list, sample_number: int, shortest_seq: int, longest_seq: int
):
    # draw the squence length distribution
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.histplot(data, kde=False, color="b")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Sequence length", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.title(
        f"{name} data distribution(Total number:{sample_number},shortest_seq:{shortest_seq},longest_seq:{longest_seq})"
    )
    plt.savefig(f"{name}.png", dpi=200, bbox_inches="tight")
