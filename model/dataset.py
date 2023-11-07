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
    def __init__(self, data_path, pose_path, index=None, transform=_transform):
        dataset = np.load(data_path)
        pose = np.load(pose_path)

        if index is None:
            self.frontal = dataset['frontal']
            self.lateral = dataset['lateral']
            self.beta = dataset['beta']
            self.pose = pose
        else:
            self.frontal = dataset['frontal'][index]
            self.lateral = dataset['lateral'][index]
            self.beta = dataset['beta'][index]
            self.pose = pose[index,:]

        self.transform = transform

    def __getitem__(self, i):
        f = self.frontal[i]
        l = self.lateral[i]
        s = self.beta[i]
        p = self.pose[i]

        if self.transform:
            f = self.transform(f)
            l = self.transform(l)
        
        return f, l, s, p

    def __len__(self):
        return len(self.frontal)   