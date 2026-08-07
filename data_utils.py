import pandas as pd
import os
def prepare_datasets(data_dir='data'):
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    return train_data, val_data
