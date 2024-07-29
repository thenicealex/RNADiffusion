"""
~/RNAdata/bpRNA_lasted.pkl is a dictionary that contains the following keys:
1. train_data: training data
2. valid_data: validation data
3. test_data: test data

Each key corresponds to a list of tuples, where each tuple contains the following elements:
1. RNA sequence ID
2. RNA sequence name
3. RNA sequence raw
4. Interaction map
"""
import os
import pickle
import numpy as np
from tqdm import tqdm

def categorize_seq(seq_data, start, end):
    """
    Sequences were sorted by length and then binned with intervals of 80 nucleotides,
    ensuring that sequences within the same bucket had similar lengths.
    """
    categories = []
    seq_dict = {}

    print(f"start is {start}, end is {end}")
    for seq in tqdm(seq_data, total=len(seq_data)):
        # print(seq)
        length = len(seq[2])
        if length >= start and length <= end:
            seq_dict["name"] = seq[1]
            seq_dict["seq_raw"] = seq[2]
            seq_dict["length"] = len(seq[2])
            seq_dict["contact"] = get_contact_map(seq)
            categories.append(seq_dict)
            seq_dict = {}

    return categories

def is_one_dimensional(lst):
  """
  Determines if a list is one-dimensional.

  Args:
    lst: The list to check.

  Returns:
    True if the list is one-dimensional, False otherwise.
  """
  for item in lst:
    if isinstance(item, list):
      return False
  return True

def get_contact_map(seq_data):
    """
    The contact map has a size of L x L (where L is the length of the RNA sequence),
    and each point in the map can be classified into two categories:
    1 denotes pairing, and 0 denotes non-pairing.
    """
    interaction_map = seq_data[3]
    seq_length = len(seq_data[2])
    contact_map = np.zeros([seq_length, seq_length])
    if is_one_dimensional(interaction_map):
        for i, site in enumerate(interaction_map):
            if site != -1:
                contact_map[i][site] = 1
    else:
        for i, j in interaction_map:
            contact_map[i][j] = 1

    return contact_map


if __name__ == "__main__":
    dataset_file = "/home/fkli/RNAdata/bpRNA_lasted.pkl"
    save_path = "/home/fkli/RNAdata/bpRNA_lasted"

    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f, encoding="bytes")

    print(f"keys in dataset.pkl is {dataset.keys()}")
    for key in dataset.keys():
        print(f"\ndata nums in {key} is {len(dataset[key])}")
        
        c = key.split("_")[0]
    
        data = dataset[key]
        seq_max_len = max([len(data[2]) for data in data])
        print(f"seq_max_len is {seq_max_len}")
        step = 80
        set_max_len = (seq_max_len // 80 + int(seq_max_len % 80 != 0)) * 80
        print(f"set_max_len is {set_max_len}")

        print("start write file")
        for i in range(0, set_max_len, step):
            cate_list = categorize_seq(data, i + 1, i + 80)

            if len(cate_list) == 0:
                continue

            if not os.path.exists(f"{save_path}/binning/{c}"):
                os.makedirs(f"{save_path}/binning/{c}")

            store_file = f"{save_path}/binning/{c}/{c}_data_{i+1}_{i+80}.pkl"
            print(f"store_path is {store_file}\ndata nums is {len(cate_list)}")
            with open(store_file, "wb") as f:
                pickle.dump(cate_list, f)
        print("file write completed\n")