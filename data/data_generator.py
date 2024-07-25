# -*- coding: utf-8 -*-
import os
from typing import List
import torch
import numpy as np
import pickle as cPickle
from random import shuffle
from torch.utils.data import Dataset

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


class RNADataset(Dataset):
    def __init__(self, data_root: List[str], upsampling: bool = False) -> None:
        if len(data_root) == 0:
            raise ValueError("data_root should be a list of data path")

        self.data_root = data_root
        self.upsampling = upsampling

        self.samples = []
        for root in self.data_root:
            print("Loading data from {}".format(root))
            self.samples += self.make_dataset(root)

        if self.upsampling:
            self.samples = self.upsampling_data()

    def make_dataset(self, directory: str) -> List[str]:
        instances = []
        directory = os.path.expanduser(directory)
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in sorted(fnames):
                if fname.endswith(".cPickle") or fname.endswith(".pkl"):
                    path = os.path.join(root, fname)
                    instances.append(path)
        return instances[:1]

    def load_data(self, path):
        with open(path, "rb") as f:
            load_data = cPickle.load(f)
        return load_data

    # for data balance, 4 times for 160~320 & 320~640
    def upsampling_data(self):
        augment_data_list = list()
        final_data_list = self.samples
        for data_path in final_data_list:
            with open(data_path, "rb") as f:
                load_data = cPickle.load(f)
            max_len = max([x["length"] for x in load_data])
            if max_len == 160:
                continue
            elif max_len == 320:
                augment_data_list.append(data_path)
            elif max_len == 640:
                augment_data_list.append(data_path)

        augment_data_list = list(
            np.random.choice(augment_data_list, 3 * len(augment_data_list))
        )
        final_data_list.extend(augment_data_list)
        shuffle(final_data_list)
        return final_data_list

    def __len__(self) -> int:
        "Denotes the total number of samples"
        return len(self.samples)

    def __getitem__(self, index: int):
        batch_data_path = self.samples[index]
        data = self.load_data(batch_data_path)
        seq_max_len = max([x["length"] for x in data])
        set_max_len = (seq_max_len // 80 + int(seq_max_len % 80 != 0)) * 80

        (
            data_name_list,
            data_length_list,
            contact_array,
            data_seq_raw_list,
            data_seq_encode_pad_array,
        ) = preprocess_data(data, set_max_len)

        contact = torch.tensor(contact_array).unsqueeze(1).long()
        data_length = torch.tensor(data_length_list).long()
        data_seq_encode_pad = torch.tensor(data_seq_encode_pad_array).float()

        return (
            set_max_len,
            data_name_list,
            data_length,
            contact,
            data_seq_raw_list,
            data_seq_encode_pad,
        )


def padding(data_array, maxlen, axis=0):
    a, b = data_array.shape
    if axis == 1:
        return np.pad(data_array, ((0, maxlen-a), (0, maxlen - b)), "constant")
    # np.pad(array, ((before_1,after_1),……,(before_n,after_n),module)
    return np.pad(data_array, ((0, maxlen - a), (0, 0)), "constant")


def seq_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: seq_dict[x.upper()], str_list))
    # need to stack
    return np.stack(encoding, axis=0)


def preprocess_data(data, set_max_len):
    shuffle(data)
    contact_list = [padding(item["contact"], set_max_len, 1) for item in data]
    data_seq_raw_list = [item["seq_raw"] for item in data]
    data_length_list = [item["length"] for item in data]
    data_name_list = [item["name"] for item in data]

    contact_array = np.stack(contact_list, axis=0)

    data_seq_encode_list = [seq_encoding(x) for x in data_seq_raw_list]
    data_seq_encode_pad_list = [padding(x, set_max_len) for x in data_seq_encode_list]
    data_seq_encode_pad_array = np.stack(data_seq_encode_pad_list, axis=0)

    return (
        data_name_list,
        data_length_list,
        contact_array,
        data_seq_raw_list,
        data_seq_encode_pad_array,
    )


def generate_token_batch(alphabet, seq_strs):
    batch_size = len(seq_strs)
    max_len = max(len(seq_str) for seq_str in seq_strs)
    tokens = torch.empty(
        (
            batch_size,
            max_len + int(alphabet.prepend_bos) + int(alphabet.append_eos),
        ),
        dtype=torch.int64,
    )
    tokens.fill_(alphabet.padding_idx)
    for i, seq_str in enumerate(seq_strs):
        if alphabet.prepend_bos:
            tokens[i, 0] = alphabet.cls_idx
        seq = torch.tensor([alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
        tokens[
            i,
            int(alphabet.prepend_bos) : len(seq_str) + int(alphabet.prepend_bos),
        ] = seq
        if alphabet.append_eos:
            tokens[i, len(seq_str) + int(alphabet.prepend_bos)] = alphabet.eos_idx
    return tokens


def get_data_id(args):
    return "{}_{}".format(args.dataset, args.seq_len)


def diff_collate_fn(batch):
    (
        contact,
        data_seq_raw_list,
        data_length,
        data_name_list,
        set_max_len,
        data_seq_encode_pad,
    ) = zip(*batch)
    if len(contact) == 1:
        contact = contact[0]
        data_seq_raw = data_seq_raw_list[0]
        data_length = data_length[0]
        data_name = data_name_list[0]
        set_max_len = set_max_len[0]
        data_seq_encode_pad = data_seq_encode_pad[0]

    else:
        set_max_len = (
            max(set_max_len) if isinstance(set_max_len, tuple) else set_max_len
        )

        contact_list = list()
        for item in contact:
            if item.shape[-1] < set_max_len:
                item = np.pad(
                    item,
                    (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]),
                    "constant",
                    0,
                )
                contact_list.append(item)
            else:
                contact_list.append(item)

        data_seq_encode_pad_list = list()
        for item in data_seq_encode_pad:
            if item.shape[-1] < set_max_len:
                item = np.pad(
                    item,
                    (0, set_max_len - item.shape[-1], 0, set_max_len - item.shape[-1]),
                    "constant",
                    0,
                )
                data_seq_encode_pad_list.append(item)
            else:
                data_seq_encode_pad_list.append(item)

        contact = torch.cat(contact_list, dim=0)
        data_seq_encode_pad = torch.cat(data_seq_encode_pad_list, dim=0)

        data_seq_raw = list()
        for item in data_seq_raw_list:
            data_seq_raw.extend(item)

        data_length = torch.cat(data_length, dim=0)

        data_name = list()
        for item in data_name_list:
            data_name.extend(item)

    # tokens = generate_token_batch_esm(data_seq_raw)

    return (
        contact,
        data_seq_raw,
        data_length,
        data_name,
        set_max_len,
        data_seq_encode_pad,
    )
