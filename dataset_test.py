from model.dataset import HSEDataset
import numpy as np
import os

dataset_path = '../dataset-generation/dataset_HSE/SMPL_augmentated_pose_variation/sample_points/'

test_index = np.load(os.path.join(dataset_path, 'train_test_index.npz'))['test_idx']
test_dataset = HSEDataset(os.path.join(dataset_path, 'dataset.npz'), index=test_index)

train_index = np.load(os.path.join(dataset_path, 'train_test_index.npz'))['train_idx']
train_dataset = HSEDataset(os.path.join(dataset_path, 'dataset.npz'), index=train_index)

print(len(train_dataset))
print(len(test_dataset))