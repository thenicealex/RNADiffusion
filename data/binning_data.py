import os
import pickle
import numpy as np
from tqdm import tqdm

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
    Generates a contact map for an RNA sequence.

    Args:
        seq_data: A tuple containing RNA sequence information.

    Returns:
        A 2D numpy array representing the contact map.
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

def categorize_sequences(seq_data, start, end):
    """
    Categorizes RNA sequences into bins based on their length.

    Args:
        seq_data: List of tuples containing RNA sequence information.
        start: Start of the length interval.
        end: End of the length interval.

    Returns:
        A list of dictionaries containing categorized sequence information.
    """
    categories = []
    
    for seq in tqdm(seq_data, total=len(seq_data)):
        seq_length = len(seq[2])
        if start <= seq_length <= end:
            seq_info = {
                "name": seq[1],
                "seq_raw": seq[2],
                "length": seq_length,
                "contact": get_contact_map(seq),
                "base_info": np.array(1)
            }
            categories.append(seq_info)

    return categories

def save_binned_data(data_path, category, categorized_data, start, end):
    """
    Saves the categorized RNA sequence data to a file.

    Args:
        data_path: Base path for saving the data.
        category: Category name.
        categorized_data: List of categorized sequence information.
        start: Start of the length interval.
        end: End of the length interval.
    """
    binning_dir = os.path.join(data_path, "binning", category)
    os.makedirs(binning_dir, exist_ok=True)
    file_path = os.path.join(binning_dir, f"{category}_data_{start}_{end}.pkl")
    
    with open(file_path, "wb") as f:
        pickle.dump(categorized_data, f)
    
    print(f"Data saved to {file_path} with {len(categorized_data)} sequences.")

def main():
    datasets_path = "/home/fkli/RNAdata/bpRNA_lasted.pkl"
    data_path = "/home/fkli/RNAdata/bpRNA_lasted"

    with open(datasets_path, "rb") as f:
        datasets = pickle.load(f, encoding="bytes")

    print(f"Keys in dataset.pkl: {datasets.keys()}")
    
    for key in datasets.keys():
        print(f"\nNumber of sequences in {key}: {len(datasets[key])}")
        
        category = key.split("_")[0]
        data = datasets[key]
        max_seq_length = max(len(seq[2]) for seq in data)
        step = 80
        max_length_rounded = (max_seq_length // step + int(max_seq_length % step != 0)) * step
        
        print(f"Max sequence length: {max_seq_length}")
        print(f"Rounded max length: {max_length_rounded}")

        for start in range(0, max_length_rounded, step):
            end = start + step
            categorized_data = categorize_sequences(data, start + 1, end)
            if categorized_data:
                save_binned_data(data_path, category, categorized_data, start + 1, end)
        
        print("File writing completed\n")

if __name__ == "__main__":
    main()