# -*- coding: utf-8 -*-
import os
import pickle
from os.path import join
from tqdm import tqdm

def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def get_batch_size(sequence_length):
    if sequence_length == 80:
        return 128
    elif sequence_length == 160:
        return 64
    elif 160 < sequence_length <= 320:
        return 16
    elif 320 < sequence_length <= 640:
        return 4
    elif 640 < sequence_length <= 1280:
        return 2
    else:
        return 1

def process_data(source_dir, target_dir):
    for file_name in tqdm(os.listdir(source_dir)):
        source_path = join(source_dir, file_name)
        sequence_length = int(file_name.split("_")[-1].split(".")[0])
        batch_size = get_batch_size(sequence_length)
        
        data = load_data(source_path)
        sequence_length = len(data)
        
        for start_idx in range(0, sequence_length, batch_size):
            end_idx = min(start_idx + batch_size, sequence_length)
            batch_data = data[start_idx:end_idx]
            
            category = os.path.basename(source_dir)
            target_category_dir = join(target_dir, category)
            os.makedirs(target_category_dir, exist_ok=True)
            
            target_file_name = f"{start_idx}_{end_idx}_{sequence_length}.pkl"
            target_path = join(target_category_dir, target_file_name)
            
            with open(target_path, "wb") as target_file:
                pickle.dump(batch_data, target_file)
            
            print(f"Written {target_path}")

if __name__ == "__main__":
    train_data_path = "/home/fkli/RNAdata/bpRNA_lasted/binning/train"
    test_data_path = "/home/fkli/RNAdata/bpRNA_lasted/binning/test"
    val_data_path = "/home/fkli/RNAdata/bpRNA_lasted/binning/val"
    target_data_path = "/home/fkli/RNAdata/bpRNA_lasted/batching"
    
    data_paths = [train_data_path, val_data_path, test_data_path]
    
    for data_path in data_paths:
        process_data(data_path, target_data_path)