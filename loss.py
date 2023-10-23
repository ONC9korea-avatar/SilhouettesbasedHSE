import os
import numpy as np
import yaml

def process_folder(path):
    config_path = os.path.join(path, 'config.yaml')
    t_loss_path = os.path.join(path, 'trainig_loss.npy')
    v_loss_path = os.path.join(path, 'validation_loss.npy')

    # config 파일이 존재하는지 확인
    if not os.path.exists(config_path):
        print(f"config file not found in {path}")
        return

    print('path : ' + path)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config['train_settings']['model']
    learning_rate = config['train_settings']['optimizer']['lr']

    t_loss = np.load(t_loss_path)
    v_loss = np.load(v_loss_path)

    indices = [99, 199, 299, 399, 499]

    t_selected_values = np.array([t_loss[i] for i in indices])
    v_selected_values = v_loss

    print('model_name : ' + model_name)
    print('learning_rate : ' + str(learning_rate))
    print('t_selected_values :', t_selected_values)
    print('v_selected_values :', v_selected_values)
    print('\n')

# 현재 디렉토리의 모든 하위 디렉토리를 가져옵니다.
base_path = 'checkpoints/'
folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# 각 폴더에 대해 반복합니다.
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    
    # 각 폴더 내의 추가 하위 폴더들을 검색합니다.
    subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    # 하위 폴더들을 반복 처리합니다.
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        process_folder(subfolder_path)
