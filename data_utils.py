import pandas as pd
import os
import json
def prepare_datasets(data_dir='data', force_split=False):
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    info_path = os.path.join(data_dir, 'dataset_info.json')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training set file does not exist: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation set file does not exist: {val_path}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)
        print(f"data info: {dataset_info}")
    return train_df, val_df