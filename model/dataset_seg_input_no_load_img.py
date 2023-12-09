import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

class HSEDataset_seg_no_load(Dataset):
    def __init__(self, data_path, index=None):
        dataset = np.load(data_path)
        self.index = index        

        if index is None:
            self.frontal = dataset['frontal_img']
            self.lateral = dataset['lateral_img']
            self.beta = dataset['betas']
            self.pose = dataset['poses']
        else:
            self.frontal = dataset['frontal_img'][index]
            self.lateral = dataset['lateral_img'][index]
            self.beta = dataset['betas'][index]
            self.pose = dataset['poses'][index]        

    def __getitem__(self, i):
        f = self.frontal[i]
        l = self.lateral[i]
        s = self.beta[i]
        p = self.pose[i]
        
        
        return f, l, s, p

    def __len__(self):
        return len(self.frontal)   
    

if __name__ == '__main__':
    start_time_7 = time.time()  # #7 시작 시간 측정
    dataset_path = '/home/user/avatar-root/dataset-generation/dataset_HSE/SMPL_augmentated_pose_variation/sample_points/'
    indices_name = 'train_test_index_seg_input.npz'
    dataset_name = 'dataset_with_seg_img_2d.npz'
    batch_size = 50

    start_time_1 = time.time()  # #1 시작 시간 측정
    indices = np.load(os.path.join(dataset_path, indices_name))     
    train_index, test_index = indices['train_idx'], indices['test_idx']
    end_time_1 = time.time()  # #1 종료 시간 측정
    print('train_index.shape : ' + str(train_index.shape))
    print('test_index.shape : ' + str(test_index.shape))
    print('Processing time for #1: {:.2f} seconds'.format(end_time_1 - start_time_1))

    start_time_2 = time.time()  # #2 시작 시간 측정
    train_dataset = HSEDataset_seg_no_load(os.path.join(dataset_path, dataset_name), train_index)
    end_time_2 = time.time()  # #2 종료 시간 측정
    print('Processing time for #2: {:.2f} seconds'.format(end_time_2 - start_time_2))

    start_time_3 = time.time()  # #3 시작 시간 측정
    test_dataset = HSEDataset_seg_no_load(os.path.join(dataset_path, dataset_name), test_index)
    end_time_3 = time.time()  # #3 종료 시간 측정
    print('Processing time for #3: {:.2f} seconds'.format(end_time_3 - start_time_3))

    start_time_4 = time.time()  # #4 시작 시간 측정
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    end_time_4 = time.time()  # #4 종료 시간 측정
    print('Processing time for #4: {:.2f} seconds'.format(end_time_4 - start_time_4))

    start_time_5 = time.time()  # #5 시작 시간 측정
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    end_time_5 = time.time()  # #5 종료 시간 측정
    print('Processing time for #5: {:.2f} seconds'.format(end_time_5 - start_time_5))
    print('train dataloader len:', len(train_dataloader))
    print('test dataloader len:', len(test_dataloader))

    start_time_6 = time.time()  # #6 시작 시간 측정
    i = 0
    f, l, s, p = train_dataset[i]  # HSEDataset_seg_no_load 클래스의 __getitem__ 호출
    end_time_6 = time.time()  # #6 종료 시간 측정


    f_image = f.squeeze(0)
    l_image = l.squeeze(0)
    
    plt.imshow(f_image, cmap='gray')  # 첫 번째 채널 사용
    plt.title("Frontal Image")

    # 이미지 파일로 저장
    plt.savefig('./frontal_seg_image.png')

    plt.figure()
    plt.imshow(l_image, cmap='gray')  # 첫 번째 채널 사용
    plt.title("Lateral Image")
    plt.savefig('./lateral_seg_image.png')

    print("Frontal Image: ", str(f.shape))
    print("Lateral Image: ", str(l.shape))
    print("Beta: ", str(s.shape))
    print("Pose: ", str(p.shape))

    print('Processing time for #6: {:.2f} seconds'.format(end_time_6 - start_time_6))

    end_time_7 = time.time()  # #7 종료 시간 측정
    print('Total Processing time : {:.2f} seconds'.format(end_time_7 - start_time_7))
