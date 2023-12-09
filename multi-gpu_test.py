import torch
import os, sys
import yaml


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # Distributed training setup
    local_rank = None
    for arg in sys.argv[1:]:
        if arg.startswith('--local-rank'):
            local_rank = int(arg.split('=')[1])
    

    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda_available else 'cpu')

    print('cuda available:', cuda_available)
    print('using device', device)
    # ---------- Device Check ---------- #

    # ---------- Reading Config ---------- #    
    #only train kjk
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

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    print(distributed)

    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    


    




