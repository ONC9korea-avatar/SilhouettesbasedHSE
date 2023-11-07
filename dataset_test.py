from model.dataset import HSEDataset
import numpy as np
import os

dataset_path = '../dataset-generation/dataset_HSE/SMPL_augmentated_pose_variation/sample_points/'

test_index = np.load(os.path.join(dataset_path, 'train_test_index.npz'))['test_idx']
test_dataset = HSEDataset(os.path.join(dataset_path, 'dataset.npz'),os.path.join(dataset_path, 'variant_pose.npy'),  index=test_index)

train_index = np.load(os.path.join(dataset_path, 'train_test_index.npz'))['train_idx']
train_dataset = HSEDataset(os.path.join(dataset_path, 'dataset.npz'),os.path.join(dataset_path, 'variant_pose.npy'), index=train_index)


pose = np.load(os.path.join(dataset_path, 'variant_pose.npy'))
train_pose = pose[train_index,:]
test_pose = pose[test_index,:]


print(train_dataset.pose.shape)
print(train_pose.shape)

print(test_dataset.pose.shape)
print(test_pose.shape)

print(test_dataset.pose[1,:])
print(test_pose[1,:])

# print("Type of pose:", type(pose))
# print("Shape of pose:", pose.shape)
# print("Data type of pose:", pose.dtype)
# print("First element of pose:", pose[0])


# print("Shape of train_pose:", train_pose.shape)
# print("First element of train_pose:", train_pose[0])

# print("Shape of test_pose:", test_pose.shape)
# print("First element of test_pose:", test_pose[0])



# print(train_index)
# print(test_index)
# print(train_dataset)

# print(len(train_dataset))
# print(len(test_dataset))