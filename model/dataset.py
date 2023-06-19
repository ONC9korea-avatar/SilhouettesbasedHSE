import numpy as np
from torch.utils.data import Dataset

import os
from tqdm import tqdm

def repeat_data(x):
    row, col = x.shape
    data = np.zeros([row, col+2])
    data[:, 0] = x[:, -1]
    data[:, 1:-1] = x
    data[:, -1] = x[:, 0]
    return data

def _transform(x):
    return repeat_data(x.transpose())

class HSEDataset(Dataset):
    def __init__(self, data_path, transform=_transform, predict_only=False):
        self.frontal = []
        self.lateral = []
        self.shape = []
        self.names = []

        self.transform = transform
        self.predict_only = predict_only

        for dir in tqdm(os.listdir(data_path), desc='loading data'):
            path = os.path.join(data_path, dir)
            f = np.load(os.path.join(path, 'frontal.npy'))
            l = np.load(os.path.join(path, 'lateral.npy'))

            if self.predict_only:
                s = []
            else:
                s = np.load(os.path.join(path, 'shape.npy'))
            

            self.frontal.append(f)
            self.lateral.append(l)
            self.shape.append(s)
            self.names.append(dir)

    def __getitem__(self, i):
        f = self.frontal[i]
        l = self.lateral[i]
        s = self.shape[i]
        n = self.names[i]

        if self.transform:
            f = self.transform(f)
            l = self.transform(l)
        
        return n, f, l, s

    def __len__(self):
        return len(self.frontal)