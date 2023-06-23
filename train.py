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
from model.dataset import HSEDataset
from torch.utils.data import DataLoader

TIMESTAMP = '_'.join(ctime(time() + 9*3600)[4:].split())
CONFIG_TEXT = None

def train(model, train_dataloader, epochs, 
          optimizer, 
          scheduler,
          checkpoint_path, 
          validation_dataloader=None,
          device=torch.device('cuda')):
    
    path = os.path.join(checkpoint_path, TIMESTAMP)
    os.makedirs(path)
    
    criterion = nn.MSELoss()
    
    train_loss = []
    validation_loss = []
    pbar = tqdm(range(1, epochs+1), desc='epoch', leave=False)
    for epoch in pbar:
        loss_n = 0
        
        for data in train_dataloader:
            f, l, s = data
            f = f.to(device, dtype=torch.float)
            l = l.to(device, dtype=torch.float)
            s = s.to(device, dtype=torch.float)
            
            # feed data and forward pass
            outputs = model(f, l)
            #********************************************************************************************
            
            loss = criterion(outputs, s.float())
            loss_n += loss.item()
            #********************************************************************************************
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        loss_n /= len(train_dataloader)

        train_loss.append(loss_n)
        pbar.set_description(f'epoch:{epoch} | TrainLoss:{loss_n:.6f}')
        
        if epoch%100 == 0:
            torch.save(model.state_dict(), os.path.join(path, f'epochs_{epoch}.ckpt'))

            if validation_dataloader is not None:
                loss_v = validate(model, validation_dataloader, device)
                validation_loss.append(loss_v)
    
    return path, train_loss, validation_loss

def validate(model, validation_dataloader, device):
    criterion = nn.MSELoss()
    loss_n = 0
    
    with torch.no_grad():    
        for data in validation_dataloader:
            f, l, s = data
            f = f.to(device, dtype=torch.float)
            l = l.to(device, dtype=torch.float)
            s = s.to(device, dtype=torch.float)
            
            outputs = model(f, l)
            
            loss = criterion(outputs, s.float())
            loss_n += loss.item()
    
    return loss_n / len(validation_dataloader)

def save_result(path, train_loss, validation_loss):
    global CONFIG_TEXT

    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        f.write(CONFIG_TEXT)

    np.save(os.path.join(path, 'training_loss.npy'), train_loss)
    np.save(os.path.join(path, 'validation_loss.npy'), validation_loss)

    plt.figure()
    plt.title('Training Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.plot(range(1, len(train_loss)+1), train_loss)
    plt.savefig(os.path.join(path, 'training_loss.png'))

def main():
    global CONFIG_TEXT
    # ---------- Device Check ---------- #
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda_available else 'cpu')

    print('cuda available:', cuda_available)
    print('using device', device)
    # ---------- Device Check ---------- #



    # ---------- Reading Config ---------- #
    config_file = sys.argv[1]
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
    # ---------- Reading Config ---------- #



    # ---------- Prepare Dataset ----------#
    indices = np.load(os.path.join(dataset_path, 'train_test_index.npz'))

    train_index, test_index = indices['train_idx'], indices['test_idx']
    train_dataset = HSEDataset(os.path.join(dataset_path, 'dataset.npz'), train_index)
    test_dataset = HSEDataset(os.path.join(dataset_path, 'dataset.npz'), test_index)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print('train dataloader len:', len(train_dataloader))
    # ---------- Prepare Dataset ----------#



    # ---------- Prepare Model and Optimizer ----------#
    model = RegressionPCA(10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=gamma)
    # ---------- Prepare Model and Optimizer ----------#



    path, train_loss, validation_loss =\
        train(model, train_dataloader, epochs, 
              optimizer, 
              scheduler,
              checkpoint_path, 
              validation_dataloader=test_dataloader,
              device=device)
    save_result(path, train_loss, validation_loss)

if __name__ == '__main__':
    main()