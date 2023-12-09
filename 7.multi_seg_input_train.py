# %%
# 

import os, sys
from time import time, localtime, strftime
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

from model.regressCNN import RegressionPCA
from model.new_regressCNN import new_RegressionPCA
from model.fc_regressCNN import fc_RegressionPCA
from model.resnetCNN import ResnetPCA
from model.resnetCNN_small import ResnetPCA_small
from model.resnetCNN_wide import ResnetPCA_wide
from model.conv2_regressCNN import conv2_RegressionPCA

from model.dataset_seg_input import HSEDataset_seg_input
from model.dataset_seg_input_no_load_img import HSEDataset_seg_no_load
from torch.utils.data import DataLoader
from SMPL.smpl_torch_batch import SMPLModel
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler


#TIMESTAMP = '_'.join(ctime(time() + 9*3600)[4:].split())
current_time = localtime(time() + 9*3600)
TIMESTAMP = strftime('%Y_%m_%d_%Hh%Mm%Ss', current_time)

CONFIG_TEXT = None

def train(model, train_dataloader, 
          train_sampler,
          epochs, 
          optimizer, 
          scheduler,
          checkpoint_path, 
          smpl_model,
          local_rank,
          validation_dataloader=None,
          device=torch.device('cuda')):
    
    #path = os.path.join(checkpoint_path, TIMESTAMP, type(model).__name__)    

    # local_rank이 0인 경우에만 출력
    if local_rank == 0:
        path = os.path.join(checkpoint_path, TIMESTAMP)
        os.makedirs(path, exist_ok=True)
        print(path)    
        
    criterion = nn.MSELoss()
    
    train_loss = []
    validation_loss = []

    # local_rank이 0인 경우에만 tqdm을 사용
    if local_rank == 0:
        pbar = tqdm(range(1, epochs+1), desc='epoch', leave=False)
    else:
        pbar = range(1, epochs+1)

    #pbar = tqdm(range(1, epochs+1), desc='epoch', leave=False)
    for epoch in pbar:
        train_sampler.set_epoch(epoch)
        loss_n = 0

        if local_rank == 0:
            data_loader_pbar = tqdm(train_dataloader, desc='Batch', leave=False)
        else:
            data_loader_pbar = train_dataloader
        
        #for data in train_dataloader:
        for data in data_loader_pbar:
            f, l, s, p = data

            f = f.to(device, dtype=torch.float)
            l = l.to(device, dtype=torch.float)
            s = s.to(device, dtype=torch.float64)     
            p = p.to(device, dtype=torch.float64)
            
            outputs = model(f, l)
            outputs = outputs.to(device, dtype=torch.float64)

            outputs_10, outputs_20670, outputs_joint_72 = outputs[:, :10], outputs[:, 10:10+20670], outputs[:, 10+20670:]

            pose_tensor = p

            trans = np.zeros((s.shape[0], 3))
            trans_tensor = torch.from_numpy(trans).type(torch.float64).to(device)       
            
            v_gt, _ = smpl_model(s, pose_tensor, trans_tensor)
            v_gt = v_gt.reshape(-1, 20670)

            loss_10 = criterion(outputs_10.float(), s.float())
            loss_20670 = criterion(outputs_20670.float(), v_gt.float())

            loss_joint_72 = criterion(outputs_joint_72.float(), p.float())

            total_loss = loss_10 + loss_20670 + loss_joint_72

            loss_n += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        scheduler.step()
        loss_n /= len(train_dataloader)

        # local_rank이 0인 경우에만 진행 상태 업데이트
        if local_rank == 0:
            train_loss.append(loss_n)
            pbar.set_description(f'epoch:{epoch} | TrainLoss:{loss_n:.6f}')
        
            if epoch%100 == 0:
                torch.save(model.state_dict(), os.path.join(path, f'epochs_{epoch}.ckpt'))

            if validation_dataloader is not None:
                loss_v = validate(model, validation_dataloader, device, smpl_model)
                validation_loss.append(loss_v)
    
    return path, train_loss, validation_loss

def validate(model, validation_dataloader, device, smpl_model):
    criterion = nn.MSELoss()
    loss_n = 0
    
    with torch.no_grad():    
        for data in validation_dataloader:
            f, l, s, p = data
            f = f.to(device, dtype=torch.float)
            l = l.to(device, dtype=torch.float)
            s = s.to(device, dtype=torch.float64)
            p = p.to(device, dtype=torch.float64)
            
            outputs = model(f, l)

            outputs_10, outputs_20670, outputs_joint_72 = outputs[:, :10], outputs[:, 10:10+20670], outputs[:, 10+20670:]

            pose_tensor = p

            trans = np.zeros((s.shape[0], 3))
            trans_tensor = torch.from_numpy(trans).type(torch.float64).to(device)       

            v_gt, _ = smpl_model(s, pose_tensor, trans_tensor)
            v_gt = v_gt.reshape(-1, 20670)

            loss_10 = criterion(outputs_10.float(), s.float())
            loss_20670 = criterion(outputs_20670.float(), v_gt.float())
            loss_joint_72 = criterion(outputs_joint_72.float(), p.float())


            total_loss = loss_10 + loss_20670 + loss_joint_72

            loss_n += total_loss.item()
    
    return loss_n / len(validation_dataloader)

def save_result(path, train_loss, validation_loss, processing_time):
    global CONFIG_TEXT

    #with open(os.path.join(path, 'config.yaml'), 'w') as f:
    #    f.write(CONFIG_TEXT)

    np.save(os.path.join(path, 'trainig_loss.npy'), train_loss)
    np.save(os.path.join(path, 'validation_loss.npy'), validation_loss)

    plt.figure()
    plt.title('Training Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.plot(range(1, len(train_loss)+1), train_loss)
    plt.savefig(os.path.join(path, 'trainig_loss.png'))

    # 전체 Validation loss 그래프
    plt.figure()
    plt.title('Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(range(1, len(validation_loss)+1), validation_loss)
    plt.savefig(os.path.join(path, 'validation_loss.png'))

    # 150 에포크부터 1000 에포크까지의 Training loss 그래프
    plt.figure()
    plt.title('Training Loss (150~1000 epochs)')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(range(150, 1001), train_loss[149:1000])  # 리스트 인덱스는 0부터 시작하므로
    plt.savefig(os.path.join(path, 'trainig_loss_150_1000.png'))   

    # processing time save
    with open(os.path.join(path, 'processing_time.txt'), 'w') as f:
        f.write(f'Processing Time: {processing_time}') 

    with open(os.path.join(path, 'path.txt'), 'w') as f:
        f.write(f'{TIMESTAMP}') 
    

    

def main():
    torch.backends.cudnn.benchmark = True

    # Distributed training setup
    #local_rank = None
    #for arg in sys.argv[1:]:
    #    if arg.startswith('--local-rank'):
    #        local_rank = int(arg.split('=')[1])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # multi GPU start
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if local_rank == 0:
        print(distributed)

    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')    
        print(local_rank)

    # ---------- Device Check ---------- #
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda_available else 'cpu')

    print('cuda available:', cuda_available)
    print('using device', device)
    # ---------- Device Check ---------- #

    # ---------- Reading Config ---------- #    
    config_file = './config.yaml'
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")

    with open(config_file) as f:
        CONFIG_TEXT = f.read()
    
    conf = yaml.safe_load(CONFIG_TEXT)

    dataset_path = conf['paths']['dataset_path']
    checkpoint_path = conf['paths']['checkpoint_path']

    train_settings = conf['train_settings']
    batch_size = train_settings['batch_size']
    epochs = train_settings['epochs']

    lr = train_settings['optimizer']['lr']
    betas = train_settings['optimizer']['betas']

    gamma = train_settings['scheduler']['gamma']
    fc1 = train_settings['fc1']
    fc2 = train_settings['fc2']
    model_type = train_settings['model']
    dataset_name = train_settings['datasets']    
    # ---------- Reading Config ---------- #
    
    # ---------- Writing Config ---------- #

    if local_rank == 0:
        with open(os.path.join(checkpoint_path, 'config.yaml'), 'w') as f:
            f.write(CONFIG_TEXT)

    # ---------- Writing Config ---------- #

    print('config load done!')
    # ---------- Prepare Dataset ----------#
    #indices = np.load(os.path.join(dataset_path, 'train_test_index.npz'))    
    indices = np.load(os.path.join(dataset_path, 'train_test_index_seg_input.npz'))  

    train_index, test_index = indices['train_idx'], indices['test_idx']
    #train_dataset = HSEDataset_seg_input(os.path.join(dataset_path, dataset_name), train_index)
    #test_dataset = HSEDataset_seg_input(os.path.join(dataset_path, dataset_name), test_index)        
    
    train_dataset = HSEDataset_seg_no_load(os.path.join(dataset_path, dataset_name), train_index)
    test_dataset = HSEDataset_seg_no_load(os.path.join(dataset_path, dataset_name), test_index)  

    print('train_dataset.frontal.shape : '+str(train_dataset.frontal.shape))
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    print('train dataloader len:', len(train_dataloader))
    print('test dataloader len:', len(test_dataloader))
    # ---------- Prepare Dataset ----------#

    model_mapping = {
        'RegressionPCA': RegressionPCA,
        'ResnetPCA_small': ResnetPCA_small,
        'ResnetPCA': ResnetPCA,
        'ResnetPCA_wide': ResnetPCA_wide,
        'new_RegressionPCA': new_RegressionPCA, 
        'fc_RegressionPCA': fc_RegressionPCA,
        'conv2_RegressionPCA' : conv2_RegressionPCA,
    }
    

    ModelClass = model_mapping[model_type]

    # ---------- Prepare Model and Optimizer ----------#
    

    model = ModelClass((10+(6890*3)+(24*3)), fc1, fc2).cuda()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [local_rank])


    # multi GPU end

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=gamma)
    
    # ---------- Prepare model of SMPL ----------#
    smpl_model_path = './SMPL/model.pkl'

    smpl_model = SMPLModel(device=device, model_path=smpl_model_path)

    start_time = time()

    path, train_loss, validation_loss =\
        train(model, 
              train_dataloader, 
              train_sampler,
              epochs, 
              optimizer, 
              scheduler,
              checkpoint_path, 
              smpl_model,
              local_rank,
              validation_dataloader=test_dataloader,
              device=device              )
    
    end_time = time()
    processing_time = end_time - start_time  # 처리 시간 계산
    formatted_time = str(timedelta(seconds=processing_time))

    if local_rank == 0:
        save_result(path, train_loss, validation_loss, formatted_time)
    

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()