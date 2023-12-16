import numpy as np
from torch.utils.data import Dataset

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
    def __init__(self, data_path, index=None, transform=_transform):
        dataset = np.load(data_path)

        if index is None:
            self.frontal = dataset['frontal_sample_points']
            self.lateral = dataset['lateral_sample_points']
            self.beta = dataset['betas']
        else:
            self.frontal = dataset['frontal_sample_points'][index]
            self.lateral = dataset['lateral_sample_points'][index]
            self.beta = dataset['betas'][index]

        self.transform = transform

    def __getitem__(self, i):
        f = self.frontal[i]
        l = self.lateral[i]
        s = self.beta[i]

        if self.transform:
            f = self.transform(f)
            l = self.transform(l)
        
        return f, l, s

    def __len__(self):
        return len(self.frontal)