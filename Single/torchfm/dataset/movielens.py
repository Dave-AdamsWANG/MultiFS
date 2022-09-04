import numpy as np
import pandas as pd
import torch.utils.data



class MovieLens1MDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='./ml-1m/train.txt'):
        data = pd.read_csv(data_dir, header=None).to_numpy()
        self.field = data[:,:-1]
        self.label = data[:,-1]
        if self.field.shape[-1] == 8:
            self.field_dims = [3706,301,81,6040,21,7,2,3402]
        else:
            self.field_dims = [3706, 301, 81, 6040, 21, 7, 2, 3402, 59112, 22457, 7152, 1524, 6100, 2076, 601, 1648, 561, 162,134, 42, 5349, 14, 4794, 4079]
    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        field = self.field[item]
        label = self.label[item]
        return field, label