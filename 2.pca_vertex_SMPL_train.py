# %%
import os, sys
from time import time, ctime
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

from model.regressCNN import RegressionPCA
from model.resnetCNN import ResnetPCA
from model.resnetCNN_small import ResnetPCA_small
from model.resnetCNN_wide import ResnetPCA_wide

from model.dataset import HSEDataset
from torch.utils.data import DataLoader
from SMPL.smpl_torch_batch import SMPLModel


TIMESTAMP = '_'.join(ctime(time() + 9*3600)[4:].split())
CONFIG_TEXT = None

def get_A_pose_parameter(size, pose_variant=False):
    """
    Returns 'A-pose' SMPL pose paramters sized size

    Arguments:
        - size
        - pose_variant (optional)

    Return:
        - poses
    """
    poses = np.zeros((size, 72))
    
    left_arm_noise = np.radians(np.random.uniform(-5, 5, size)) if pose_variant else 0
    right_arm_noise = np.radians(np.random.uniform(-5, 5, size)) if pose_variant else 0

    poses[:,16 *3 + 2] = - np.pi / 3 + left_arm_noise # Left arm 
    poses[:,17 *3 + 2] = np.pi / 3 + right_arm_noise # Right arm

    left_leg_noise = np.radians(np.random.uniform(-3, 3, size)) if pose_variant else 0
    right_leg_noise = np.radians(np.random.uniform(-3, 3, size)) if pose_variant else 0

    poses[:,1 *3 + 2] = +np.pi / 36 + left_leg_noise # Left leg
    poses[:,2 *3 + 2] = -np.pi / 36 + right_leg_noise # Right leg

    poses[:,10 *3 + 2] = -np.pi / 6 - left_leg_noise # Left foot
    poses[:,11 *3 + 2] = +np.pi / 6 - right_leg_noise # Right foot

    return poses

def train(model, train_dataloader, epochs, 
          optimizer, 
          scheduler,
          checkpoint_path, 
          smpl_model,
          validation_dataloader=None,
          device=torch.device('cuda')):
    
    path = os.path.join(checkpoint_path, TIMESTAMP, type(model).__name__)
    os.makedirs(path)

    print(path)
        
    criterion = nn.MSELoss()
    
    train_loss = []
    validation_loss = []
    pbar = tqdm(range(1, epochs+1), desc='epoch', leave=False)
    for epoch in pbar:
        loss_n = 0
        
        for data in train_dataloader:
            f, l, s, p = data

            f = f.to(device, dtype=torch.float)
            l = l.to(device, dtype=torch.float)
            s = s.to(device, dtype=torch.float64)     
            p = p.to(device, dtype=torch.float64)

            #print(type(s))
            #print(s.shape)
            
            # feed data and forward pass
            outputs = model(f, l)
            #********************************************************************************************
            #s = s.to(device, dtype=torch.float64)
            outputs = outputs.to(device, dtype=torch.float64)

            outputs_10, outputs_20670 = outputs[:, :10], outputs[:, 10:]

            #pose = get_A_pose_parameter(s.shape[0])
            #pose_tensor = torch.from_numpy(pose).type(torch.float64).to(device)
            pose_tensor = p

            trans = np.zeros((s.shape[0], 3))
            trans_tensor = torch.from_numpy(trans).type(torch.float64).to(device)       

            v_gt, _ = smpl_model(s, pose_tensor, trans_tensor)
            v_gt = v_gt.reshape(-1, 20670)

            loss_10 = criterion(outputs_10.float(), s.float())
            loss_20670 = criterion(outputs_20670.float(), v_gt.float())

            total_loss = loss_10 + loss_20670

            #****************************************************
            #loss = criterion(outputs.float(), v_gt.float())
            #loss_n += loss.item()

            loss_n += total_loss.item()
            #********************************************************************************************
            # backward and optimize
            optimizer.zero_grad()
            #loss.backward()
            total_loss.backward()
            optimizer.step()
        
        scheduler.step()
        loss_n /= len(train_dataloader)

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

            outputs_10, outputs_20670 = outputs[:, :10], outputs[:, 10:]

            #***************************************
            # pose = get_A_pose_parameter(s.shape[0])
            # pose_tensor = torch.from_numpy(pose).type(torch.float64).to(device)
            pose_tensor = p

            trans = np.zeros((s.shape[0], 3))
            trans_tensor = torch.from_numpy(trans).type(torch.float64).to(device)       

            v_gt, _ = smpl_model(s, pose_tensor, trans_tensor)
            v_gt = v_gt.reshape(-1, 20670)

            #****************************************************
            loss_10 = criterion(outputs_10.float(), s.float())
            loss_20670 = criterion(outputs_20670.float(), v_gt.float())

            total_loss = loss_10 + loss_20670

            #****************************************************
            #loss = criterion(outputs.float(), v_gt.float())
            #loss_n += loss.item()

            loss_n += total_loss.item()
            #***************************************
            #loss = criterion(outputs.float(), v_gt.float())
            #***************************************
            
            #loss = criterion(outputs, s.float())
            #loss_n += loss.item()
    
    return loss_n / len(validation_dataloader)

def save_result(path, train_loss, validation_loss):
    global CONFIG_TEXT

    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        f.write(CONFIG_TEXT)

    np.save(os.path.join(path, 'trainig_loss.npy'), train_loss)
    np.save(os.path.join(path, 'validation_loss.npy'), validation_loss)

    plt.figure()
    plt.title('Training Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.plot(range(1, len(train_loss)+1), train_loss)
    plt.savefig(os.path.join(path, 'trainig_loss.png'))

def main():
    global CONFIG_TEXT
    # ---------- Device Check ---------- #
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda_available else 'cpu')

    print('cuda available:', cuda_available)
    print('using device', device)
    # ---------- Device Check ---------- #



    # ---------- Reading Config ---------- #    
    #only train kjk
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = './config.yaml'
    #only train kjk

    #only debugging kjk
    #config_file = "/home/user/avatar-root/SilhouettesbasedHSE/config.yaml"
    #only debugging kjk

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
    model_type = train_settings['model']
    # ---------- Reading Config ---------- #



    # ---------- Prepare Dataset ----------#
    indices = np.load(os.path.join(dataset_path, 'train_test_index.npz'))    

    train_index, test_index = indices['train_idx'], indices['test_idx']
    train_dataset = HSEDataset(os.path.join(dataset_path, 'dataset.npz'),os.path.join(dataset_path, 'variant_pose.npy'), train_index)
    test_dataset = HSEDataset(os.path.join(dataset_path, 'dataset.npz'),os.path.join(dataset_path, 'variant_pose.npy'), test_index)    

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print('train dataloader len:', len(train_dataloader))
    # ---------- Prepare Dataset ----------#

    model_mapping = {
        'RegressionPCA': RegressionPCA,
        'ResnetPCA_small': ResnetPCA_small,
        'ResnetPCA': ResnetPCA,
        'ResnetPCA_wide': ResnetPCA_wide,
    }

    ModelClass = model_mapping[model_type]

    # ---------- Prepare Model and Optimizer ----------#
    model = ModelClass(10+(6890*3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=gamma)
    
    
    # ---------- Prepare model of SMPL ----------#
    smpl_model_path = './SMPL/model.pkl'
    #smpl_model_path = '/home/user/avatar-root/SilhouettesbasedHSE/SMPL/model.pkl'
    

    smpl_model = SMPLModel(device=device, model_path=smpl_model_path)




    path, train_loss, validation_loss =\
        train(model, 
              train_dataloader, 
              epochs, 
              optimizer, 
              scheduler,
              checkpoint_path, 
              smpl_model,
              validation_dataloader=test_dataloader,
              device=device)
    save_result(path, train_loss, validation_loss)
    

if __name__ == '__main__':
    main()