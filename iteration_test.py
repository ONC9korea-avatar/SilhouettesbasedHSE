# 

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

# 설정할 날짜 (예: 2023년 11월 10일)
cutoff_date = datetime.datetime(2023, 11, 10)

# 현재 디렉토리의 내용을 나열
current_directory = './checkpoints'  # 이 부분을 원하는 디렉토리 경로로 변경하세요.
directories = []

for item in os.listdir(current_directory):
    item_path = os.path.join(current_directory, item)
    if os.path.isdir(item_path):
        # 폴더의 생성 날짜 가져오기
        creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(item_path))
        if creation_time > cutoff_date:
            directories.append(item)

# 결과 정렬
directories.sort()

cuda_available = torch.cuda.is_available()
print('cuda available:', cuda_available)

device = torch.device('cuda:0' if cuda_available else 'cpu')
print('using device', device)

# 결과 출력
for directory in directories:
    print("infer_model_name : " + str(directory))
    infer_model_name = directory
    model_name = 'fc_RegressionPCA'
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

    conf = yaml.safe_load(CONFIG_TEXT)    

    model_type = conf['train_settings']['model']
    model_mapping = {
        'RegressionPCA': RegressionPCA,
        'ResnetPCA_small': ResnetPCA_small,
        'ResnetPCA': ResnetPCA,
        'ResnetPCA_wide': ResnetPCA_wide,
        'new_RegressionPCA': new_RegressionPCA,
        'fc_RegressionPCA': fc_RegressionPCA,
    }

    ModelClass = model_mapping[model_type]

    fc1 = conf['train_settings']['fc1']
    fc2 = conf['train_settings']['fc2']

    print('fc1 : ' + str(fc1))
    print('fc2 : ' + str(fc2))

    # ---------- Prepare Model and Optimizer ----------#
    #infer_model = ModelClass(10+(6890*3)+(24*3)).to(device)
    infer_model = ModelClass((10+(6890*3)+(24*3)), fc1, fc2).to(device)

    #infer_model = RegressionPCA(10+(6890*3)+(24*3)).to(device)
    infer_model.load_state_dict(torch.load(infer_model_path))
    _ = infer_model.eval()

    dataset_path = '../dataset-generation/dataset_HSE/SMPL_augmentated_pose_variation/sample_points/'

    test_index = np.load(os.path.join(dataset_path, 'train_test_index.npz'))['test_idx']
    test_dataset = HSEDataset_aug(os.path.join(dataset_path, 'dataset_modified_sample_point_no_vertices.npz'), test_index)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)

    def infer(frontal, lateral):
        with torch.no_grad():
            beta = infer_model(frontal, lateral)
        
        return beta

    smpl_model_path = './SMPL/model.pkl'
    smpl_model = SMPLModel(device=torch.device('cuda'), model_path=smpl_model_path)

    beta_gt_list = []
    beta_out_list = []
    vertex_out_list = []
    out_p_list = []
    gt_p_list = []

    for data in tqdm(test_dataloader):
        f, l, b_gt, p = data   

        f = f.to(device, dtype=torch.float)
        l = l.to(device, dtype=torch.float)    
        p = p.to(device, dtype=torch.float64)

        #print(f.shape)
        #print(l.shape)

        outputs = infer(f, l)
        outputs_10, outputs_20670, outputs_joint_72 = outputs[:, :10], outputs[:, 10:10+20670], outputs[:, 10+20670:]
        
        beta_out_arr = np.array(outputs_10.to('cpu'))
        vertex_out_arr = np.array(outputs_20670.to('cpu'))
        out_p_arr = np.array(outputs_joint_72.to('cpu'))
        gt_p_arr = np.array(p.to('cpu'))
        
        beta_gt_list.extend(b_gt)
        beta_out_list.extend(beta_out_arr)
        vertex_out_list.extend(vertex_out_arr)
        gt_p_list.extend(gt_p_arr)
        out_p_list.extend(out_p_arr)
        

    beta_gt_vstack = np.vstack(beta_gt_list)
    beta_out_vstack = np.vstack(beta_out_list)
    vertex_out_vstack = np.vstack(vertex_out_list)
    gt_p_vstack = np.vstack(gt_p_list)
    out_p_vstack = np.vstack(out_p_list)

    print(beta_gt_vstack.shape, beta_out_vstack.shape, vertex_out_vstack.shape, gt_p_vstack.shape)

    beta_gt_tensor = torch.from_numpy(beta_gt_vstack).type(torch.float64).to(device)
    beta_out_tensor = torch.from_numpy(beta_out_vstack).type(torch.float64).to(device)
    vertex_out_tensor = torch.from_numpy(vertex_out_vstack).type(torch.float64).to(device)
    gt_p_tensor = torch.from_numpy(gt_p_vstack).type(torch.float64).to(device)
    out_p_tensor = torch.from_numpy(out_p_vstack).type(torch.float64).to(device)

    #pose = get_A_pose_parameter(beta_out.shape[0])
    #pose_tensor = torch.from_numpy(pose).type(torch.float64).to(device)

    print('beta_out_vstack.shape[0]:' + str(beta_out_vstack.shape[0]))

    trans = np.zeros((beta_out_vstack.shape[0], 3))
    trans_tensor = torch.from_numpy(trans).type(torch.float64).to(device)

    batch_size = 1024
    num_batch = len(beta_out_vstack) // batch_size

    v_beta_outs = None
    v_vertex_outs = None
    v_gts = None

    for i in tqdm(range(num_batch)):
        if i < num_batch-1:
            b_b_out = beta_out_tensor[i*batch_size:(i+1)*batch_size]
            b_v_out = vertex_out_tensor[i*batch_size:(i+1)*batch_size]
            p = gt_p_tensor[i*batch_size:(i+1)*batch_size]
            out_p = out_p_tensor[i*batch_size:(i+1)*batch_size]
            t = trans_tensor[i*batch_size:(i+1)*batch_size]        
            b_gt = beta_gt_tensor[i*batch_size:(i+1)*batch_size]

        else:
            b_b_out = beta_out_tensor[i*batch_size:]
            b_v_out = vertex_out_tensor[i*batch_size:]
            p = gt_p_tensor[i*batch_size:]
            out_p = out_p_tensor[i*batch_size:]
            t = trans_tensor[i*batch_size:]        
            b_gt = beta_gt_tensor[i*batch_size:]
        
        v_b_out, _ = smpl_model(b_b_out, out_p, t)
        v_gt, _ = smpl_model(b_gt, p, t)

        if v_beta_outs is None:
            v_beta_outs = np.array(v_b_out.cpu())
            v_vertex_outs = np.array(b_v_out.cpu())
            v_gts = np.array(v_gt.cpu())
        else:
            v_beta_outs = np.concatenate([v_beta_outs, v_b_out.cpu()])
            v_vertex_outs = np.concatenate([v_vertex_outs, b_v_out.cpu()])
            v_gts = np.concatenate([v_gts, v_gt.cpu()])

    #print(type(v_beta_outs))
    #print(type(v_vertex_outs))
    print(v_beta_outs.shape)
    print(v_vertex_outs.shape)
    v_vertex_outs = v_vertex_outs.reshape(-1,6890,3)

    print('v_beta_outs.shape : ' + str(v_beta_outs.shape))
    print('v_b_out : ' + str(v_vertex_outs.shape))
    print('v_gts.shape : ' + str(v_gts.shape))


    beta_displacement = v_beta_outs - v_gts
    vertex_displacement = v_vertex_outs - v_gts

    beta_distance = np.linalg.norm(beta_displacement, axis=2)
    vertex_distance = np.linalg.norm(vertex_displacement, axis=2)
    beta_v2v = beta_distance.mean(axis=1) * 1000
    vertex_v2v = vertex_distance.mean(axis=1) * 1000

    #################################################
    ################    BETA    #####################
    #################################################

    print(f'beta_mean, beta_std = {beta_v2v.mean():.3f} mm, {beta_v2v.std():.3f} mm')
    print(f'beta_min, beta_max = {beta_v2v.min():.3f} mm, {beta_v2v.max():.3f} mm')

    beta_mean_std_str = f"beta_mean, beta_std = {beta_v2v.mean():.3f} mm, {beta_v2v.std():.3f} mm"
    beta_min_max_str = f"beta_min, beta_max = {beta_v2v.min():.3f} mm, {beta_v2v.max():.3f} mm"

    beta_hist, beta_bins = np.histogram(beta_v2v, bins=100)

    plt.title('beta_Histogram of V2V')
    plt.xlabel('beta_V2V (mm)')
    plt.ylabel('beta_frequency')

    plt.hist(beta_bins[:-1], beta_bins, weights=beta_hist)
    plt.show()

    beta_large_v2v_indices = np.where(beta_v2v > 4.0)[0]
    print('beta_large_v2v_indices : ' + str(beta_large_v2v_indices))

    #################################################
    ################    VERTEX    ###################
    #################################################
    print(f'vertex_mean, vertex_std = {vertex_v2v.mean():.3f} mm, {vertex_v2v.std():.3f} mm')
    print(f'vertex_min, vertex_max = {vertex_v2v.min():.3f} mm, {vertex_v2v.max():.3f} mm')

    vertex_mean_std_str = f"vertex_mean, vertex_std = {vertex_v2v.mean():.3f} mm, {vertex_v2v.std():.3f} mm"
    vertex_min_max_str = f"vertex_min, vertex_max = {vertex_v2v.min():.3f} mm, {vertex_v2v.max():.3f} mm"

    vertex_hist, vertex_bins = np.histogram(vertex_v2v, bins=100)

    plt.title('vertex_Histogram of V2V')
    plt.xlabel('vertex_V2V (mm)')
    plt.ylabel('vertex_frequency')

    plt.hist(vertex_bins[:-1], vertex_bins, weights=vertex_hist)
    plt.show()

    vertex_large_v2v_indices = np.where(vertex_v2v > 4.0)[0]
    print('vertex_large_v2v_indices : ' + str(vertex_large_v2v_indices))

    #################################################

    from obj_utils.misc import *

    save_beta_indices = beta_large_v2v_indices
    save_vertex_indices = vertex_large_v2v_indices

    results_path = f'./test_results/{infer_model_name}'
    results_obj_path = f'./test_results/{infer_model_name}/obj'
    results_obj_beta_path = f'./test_results/{infer_model_name}/obj/beta'
    results_obj_vertex_path = f'./test_results/{infer_model_name}/obj/vertex'
    results_file_path = os.path.join(results_path, "results.txt")
    results_file_path_only_number = os.path.join(results_path, "results_only_number.txt")

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(results_obj_path, exist_ok=True)
    os.makedirs(results_obj_beta_path, exist_ok=True)
    os.makedirs(results_obj_vertex_path, exist_ok=True)

    with open(results_file_path, 'w') as f:
        f.write(beta_mean_std_str + "\n")
        f.write(beta_min_max_str + "\n")
        f.write(vertex_mean_std_str + "\n")
        f.write(vertex_min_max_str + "\n")

    with open(results_file_path_only_number, 'w') as f:
        f.write(f"{beta_v2v.mean():.3f}\t")
        f.write(f"{beta_v2v.std():.3f}\t")
        f.write(f"{beta_v2v.min():.3f}\t")
        f.write(f"{beta_v2v.max():.3f}\t")
        f.write(f"{vertex_v2v.mean():.3f}\t")
        f.write(f"{vertex_v2v.std():.3f}\t")
        f.write(f"{vertex_v2v.min():.3f}\t")
        f.write(f"{vertex_v2v.max():.3f}\t")

    if b_save_obj:
        for i in tqdm(save_beta_indices):
            v_out, v_gt = v_beta_outs[i], v_gts[i]
            name = f'smpl_{test_index[i]:05d}'

            v_out_colored = add_vertices_color(v_out, [1., 0., 0.,])
            v_gt_colored = add_vertices_color(v_gt, [0., 1., 0.,])

            v_merged = merge_vertices(v_out_colored, v_gt_colored)
            save_obj(os.path.join(results_path, f'obj/beta/{name}_merged.obj'), v_merged)


        for i in tqdm(vertex_large_v2v_indices):
            v_out, v_gt = v_vertex_outs[i], v_gts[i]
            name = f'smpl_{test_index[i]:05d}'

            v_out_colored = add_vertices_color(v_out, [1., 0., 1.,])
            v_gt_colored = add_vertices_color(v_gt, [0., 1., 0.,])

            v_merged = merge_vertices(v_out_colored, v_gt_colored)
            save_obj(os.path.join(results_path, f'obj/vertex/{name}_merged.obj'), v_merged)
