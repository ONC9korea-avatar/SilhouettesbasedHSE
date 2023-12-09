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

class HSEDataset_aug(Dataset):
    def __init__(self, data_path, index=None, transform=_transform):
        dataset = np.load(data_path)        

        if index is None:
            self.frontal = dataset['frontal_sample_points']
            self.lateral = dataset['lateral_sample_points']
            self.betas = dataset['betas']
            self.poses = dataset['poses']
            #self.vertices = dataset['vertices']            
        else:
            self.frontal = dataset['frontal_sample_points'][index]
            self.lateral = dataset['lateral_sample_points'][index]
            self.betas = dataset['betas'][index]
            self.poses = dataset['poses'][index]
            #self.vertices = dataset['vertices'][index]

        self.transform = transform


######################################################################################

    def __getitem__(self, i):
        f = self.frontal[i]
        l = self.lateral[i]
        s = self.betas[i]
        p = self.poses[i]
        #v = self.vertices[i]

        #입력의 차원을 맞춰주기 위해 아래 코드가 동작해야함(ex. (650,2) -> (2,650))
        if self.transform:
            f = self.transform(f)
            l = self.transform(l)
        
        return f, l, s, p

    def __len__(self):
        return len(self.frontal)   