# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from typing import Union, List
from random import shuffle

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from utils.data import load_data, encode_name, padding, seq2encoding


class RNADataset(Dataset):
    def __init__(
        self, data_paths: Union[str, List[str]], upsampling: bool = False
    ) -> None:
        """
        Initialize the RNA Dataset.

        Args:
            data_paths (Union[str, List[str]]): A single path or a list of paths to data directories.
            upsampling (bool): Whether to perform upsampling on the data.
        """
        self.data_paths = [data_paths] if isinstance(data_paths, str) else data_paths
        self.upsampling = upsampling
        self.samples = self._load_samples()

    def _load_samples(self) -> List[str]:
        samples = []
        for path in self.data_paths:
            print(f"Loading data from {path}")
            samples += self._make_dataset(path)
        if self.upsampling:
            samples = self._upsample_data(samples)
        return samples

    def _make_dataset(self, directory: str) -> List[str]:
        instances = []
        directory = os.path.expanduser(directory)
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in sorted(fnames):
                if fname.endswith((".cPickle", ".pickle", ".pkl", ".js", ".json")):
                    instances.append(os.path.join(root, fname))
        return instances

    def _upsample_data(self, samples: List[str]) -> List[str]:
        augment_data_list = []
        for data_path in samples:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
            max_len = max(item["length"] for item in data)
            if max_len in {320, 640}:
                augment_data_list.append(data_path)

        augment_data_list = list(
            np.random.choice(augment_data_list, 3 * len(augment_data_list))
        )
        samples.extend(augment_data_list)
        shuffle(samples)
        return samples

    def preprocess(self, index):
        data_path = self.samples[index]
        data = load_data(data_path)
        max_len = max(item["length"] for item in data)
        set_max_len = (max_len // 80 + int(max_len % 80 != 0)) * 80

        contact_list = [item["contact"] for item in data]
        base_info_list = [item["base_info"] for item in data]
        raw_seq_list = [item["seq_raw"].replace("Y", "N") for item in data]
        length_array = np.array([item["length"] for item in data])

        padded_contact_array = [
            padding(np.array(item), set_max_len, axis=1) for item in contact_list
        ]
        contact_array = np.stack(padded_contact_array, axis=0)

        encoded_name_list = [encode_name(item["name"]) for item in data]
        max_name_len = max([len(item) for item in encoded_name_list])
        paddd_name_list = [
            padding(np.array(item), max_name_len) for item in encoded_name_list
        ]
        name_array = np.stack(paddd_name_list, axis=0)

        base_info_array = np.stack(base_info_list, axis=0)
        encoded_seq_list = [seq2encoding(seq) for seq in raw_seq_list]
        padded_encoded_seq_list = [
            padding(seq, set_max_len) for seq in encoded_seq_list
        ]
        padded_encoded_seq_array = np.stack(padded_encoded_seq_list, axis=0)

        return (
            set_max_len,
            name_array,
            raw_seq_list,
            length_array,
            contact_array,
            base_info_array,
            padded_encoded_seq_array,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):

        (
            set_max_len,
            name_array,
            raw_seq_list,
            length_array,
            contact_array,
            base_info_array,
            padded_encoded_seq_array,
        ) = self.preprocess(index)

        contact_tensor = torch.tensor(contact_array).unsqueeze(1).long()
        base_info_tensor = torch.tensor(base_info_array).float()
        name_tensor = torch.tensor(name_array)
        length_tensor = torch.tensor(length_array).long()
        padded_encoded_seq_tensor = torch.tensor(padded_encoded_seq_array).float()

        return (
            set_max_len,
            name_tensor,
            raw_seq_list,
            length_tensor,
            contact_tensor,
            base_info_tensor,
            padded_encoded_seq_tensor,
        )

    @staticmethod
    def collate_fn(batch):
        (
            set_max_len,
            name_tensor,
            raw_seq_list,
            length_tensor,
            contact_tensor,
            base_info_tensor,
            padded_encoded_seq_tensor,
        ) = zip(*batch)

        return (
            set_max_len[0],
            name_tensor[0],
            raw_seq_list[0],
            length_tensor[0],
            contact_tensor[0],
            base_info_tensor[0],
            padded_encoded_seq_tensor[0],
        )
