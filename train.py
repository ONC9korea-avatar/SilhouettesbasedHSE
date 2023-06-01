# %%
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

from model.regressCNN import RegressionPCA
from model.dataset import HSEDataset
from torch.utils.data import DataLoader

cuda_available = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_available else 'cpu')

print('cuda available:', cuda_available)
print('using device', device)



batch_size = 128
learning_rate = 1e-5
training_epoch = 1000

train_dataset = HSEDataset('./dataset_HSE/vertex_6890/sample_648/train/')

train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
print('train dataloader len:', len(train_data_loader))

model = RegressionPCA(10).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

gender='male'
loss_train_epoch1 = []
loss_train = []

checkpoint_path = './checkpoints/checkpoint_{}/'.format(time.ctime().replace(' ', '_'))
os.mkdir(checkpoint_path)

pbar = tqdm(range(1, training_epoch+1), desc='epoch', leave=False)

for epoch in pbar:
    loss_n = 0
    
    for i, data in enumerate(train_data_loader):
        _, f, l, s = data
        f = f.to(device, dtype = torch.float)
        l = l.to(device, dtype = torch.float)
        s = s.to(device, dtype = torch.float)
        
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

        if epoch == 1:
            loss_train_epoch1.append(loss.item())
    
    loss_n /= len(train_data_loader)

    pbar.set_description(f'epoch:{epoch} | TrainLoss:{loss_n:.6f}')
    loss_train.append(loss_n)
    
    if epoch%50 == 0:
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'{gender}_{epoch}.ckpt'))

plt.figure()
plt.title('Training Loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.plot(range(1, training_epoch+1), loss_train)
plt.savefig(os.path.join(checkpoint_path, 'trainig_loss.png'))