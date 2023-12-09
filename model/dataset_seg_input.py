# from torch.utils.data import Dataset


# class HSEDataset_seg_input(Dataset):
#     def __init__(self, dataset, index=None):
#         if index is None:
#             self.frontal = dataset['frontal_img']
#             self.lateral = dataset['lateral_img']
#             self.betas = dataset['betas']
#             self.poses = dataset['poses']
#         else:
#             self.frontal = dataset['frontal_img'][index]
#             self.lateral = dataset['lateral_img'][index]
#             self.betas = dataset['betas'][index]
#             self.poses = dataset['poses'][index]

#     def __getitem__(self, i):
#         f = self.frontal[i]
#         l = self.lateral[i]
#         s = self.betas[i]
#         p = self.poses[i]

#         # 이미지 데이터는 이미 처리되어 있으므로 추가 변환은 필요 없음
#         return f, l, s, p

#     def __len__(self):
#         return len(self.frontal)


import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

class HSEDataset_seg_input(Dataset):
    def __init__(self, data_path, index=None):
        self.dataset = np.load(data_path)
        self.index = index
        self.base_path = '/home/user/avatar-root/20231024_new_augmentation_pose/SilhouettesbasedHSE/result/silhouettes'  # 이미지 파일의 경로

    def __getitem__(self, i):
        if self.index is not None:
            i = self.index[i]

        frontal_path = os.path.join(self.base_path, str(i), 'frontal.png')
        lateral_path = os.path.join(self.base_path, str(i), 'lateral.png')

        f_img = Image.open(frontal_path)
        l_img = Image.open(lateral_path)

        new_size = (f_img.width // 4, f_img.height//4)

        f = np.array(f_img.resize(new_size), dtype=np.uint8)[:, :, 0][None, :, :]
        l = np.array(l_img.resize(new_size), dtype=np.uint8)[:, :, 0][None, :, :]
        s = self.dataset['betas'][i]
        p = self.dataset['poses'][i]

        return f, l, s, p

    def __len__(self):
        return len(self.index) if self.index is not None else self.dataset['betas'].shape[0]
