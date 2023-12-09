import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

import torch
from model.dataset_aug import HSEDataset_aug
from torch.utils.data import DataLoader

from model.regressCNN import RegressionPCA
from model.new_regressCNN import new_RegressionPCA
from model.resnetCNN import ResnetPCA
from model.resnetCNN_small import ResnetPCA_small
from model.resnetCNN_wide import ResnetPCA_wide
from model.fc_regressCNN import fc_RegressionPCA

from SMPL.smpl_torch_batch import SMPLModel
from obj_utils.io import *

# 시작 날짜 설정 (예: 2023년 11월 10일)
cutoff_date = datetime.datetime(2023, 11, 8)

# 끝 날짜 설정 (예: 2023년 11월 15일)
end_date = datetime.datetime(2023, 12, 5)

# 현재 디렉토리의 내용을 나열
current_directory = './checkpoints'  # 이 부분을 원하는 디렉토리 경로로 변경하세요.
directories = []

for item in os.listdir(current_directory):
    item_path = os.path.join(current_directory, item)
    if os.path.isdir(item_path):
        # 폴더의 생성 또는 수정 날짜 가져오기
        creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(item_path))
        # 시작 날짜와 끝 날짜 사이인지 확인
        if cutoff_date <= creation_time <= end_date:
            directories.append(item)

# 결과 정렬
directories.sort()


# 결과 출력
for directory in directories:
    print("infer_model_name : " + str(directory))
    infer_model_name = directory
    model_name = 'new_RegressionPCA'
    b_save_obj = False                              #User Input
    infer_model_path = f'./checkpoints/{infer_model_name}/{model_name}/epochs_1000.ckpt'
    config_path = f'./checkpoints/{infer_model_name}/{model_name}/config.yaml'
    results_path_for_config = f'./test_results/{infer_model_name}'


    # YAML 파일이 존재하는지 확인
    with open(config_path) as f:
        CONFIG_TEXT = f.read()

    if not os.path.exists(results_path_for_config):
        os.makedirs(results_path_for_config)

    with open(os.path.join(results_path_for_config, 'config.yaml'), 'w') as f:
        f.write(CONFIG_TEXT)
