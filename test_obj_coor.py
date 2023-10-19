import os, sys
from torch.utils.data import DataLoader
from SMPL.smpl_torch_batch import SMPLModel
import numpy as np

global CONFIG_TEXT
# ---------- Device Check ---------- #
cuda_available = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_available else 'cpu')

print('cuda available:', cuda_available)
print('using device', device)
# ---------- Device Check ---------- #



# ---------- Reading Config ---------- #    
#only train kjk
config_file = sys.argv[1]
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

indices = np.load(os.path.join(dataset_path, 'train_test_index.npz'))

train_index, test_index = indices['train_idx'], indices['test_idx']
train_dataset = HSEDataset(os.path.join(dataset_path, 'dataset.npz'), train_index)
test_dataset = HSEDataset(os.path.join(dataset_path, 'dataset.npz'), test_index)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print('train dataloader len:', len(train_dataloader))