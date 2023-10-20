import os
import random

import argparse
from multiprocessing import Pool

import numpy as np
import torch
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from SMPL.smpl_torch_batch import SMPLModel

from obj_utils.smpl import *
from obj_utils.projection import *

from utils.yaml import load_yaml

class SMPLDataset(Dataset):
    def __init__(self, beta_arr, pose_arr=None, trans_arr=None, device=torch.device('cuda'), float_type = torch.float64):
        super().__init__()
        self.num_betas = beta_arr.shape[0]
        self.device = device
        self.float_type = float_type

        self.beta_arr = beta_arr
        self.pose_arr = pose_arr if pose_arr is not None else np.zeros((self.num_betas, 72))
        self.trans_arr = trans_arr if trans_arr is not None else np.zeros((self.num_betas, 3))

    def __len__(self):
        return self.num_betas

    def __getitem__(self, i):
        beta, pose, trans = self.beta_arr[i], self.pose_arr[i], self.trans_arr[i]

        beta = torch.from_numpy(beta).type(self.float_type).to(self.device)
        pose = torch.from_numpy(pose).type(self.float_type).to(self.device)
        trans = torch.from_numpy(trans).type(self.float_type).to(self.device)

        return beta, pose, trans

def worker(args):
    i, vertices, faces,  conf = args
    
    cam_conf = conf['camera']
    cam_distance = cam_conf['distance']
    run_pinhole = cam_conf['pinhole']['run']

    silhouette_conf = conf['silhouette']
    image_width = silhouette_conf['image_width']
    image_height = silhouette_conf['image_height']
    max_human_height = silhouette_conf['max_human_height']
    save_png = silhouette_conf['save_png']
    save_path = silhouette_conf['save_path']

    sample_points_conf = conf['sample_points']
    run_sample_points = sample_points_conf['run_sample_points']
    num_sample_points = sample_points_conf['num_sample_points']

    vertices -= (vertices.min(axis=0) + vertices.max(axis=0))/2
    human_height = max(vertices[:, 1]) - min(vertices[:, 1])

    vertices_frontal = vertices.copy()
    vertices_lateral = vertices.copy()[:,(2,1,0)]

    vertices_frontal[:, 2] += cam_distance
    vertices_frontal[:, 2] = 2 * cam_distance  - vertices_frontal[:, 2]
    vertices_lateral[:, 2] += cam_distance

    if run_pinhole:
        proj_frontal = pinhole(vertices_frontal)
        proj_lateral = pinhole(vertices_lateral)
    else:
        proj_frontal = vertices_frontal[:,[0,1]]
        proj_lateral = vertices_lateral[:,[0,1]]

    silhouette_frontal = make_silhouette(proj_frontal, faces, image_width, image_height, human_height, max_human_height)
    silhouette_lateral = make_silhouette(proj_lateral, faces, image_width, image_height, human_height, max_human_height)
    
    if save_png:
        path = os.path.join(save_path, 'silhouettes',f'{i}')
        os.makedirs(path, exist_ok= True)
        silhouette_frontal.save(os.path.join(path, 'frontal.png'))
        silhouette_lateral.save(os.path.join(path, 'lateral.png'))

    output = []
    if run_sample_points:
        sample_points_frontal = get_sample_points(silhouette_frontal, num_sample_points)
        sample_points_lateral = get_sample_points(silhouette_lateral, num_sample_points)
        output.append([sample_points_frontal, sample_points_lateral])

    return output

def get_sample_points(im, sample_num):
    im = np.array(im)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    contours, _ = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    largest_cont = max(contours, key=lambda c:cv.contourArea(c))
    
    num, _, cols = largest_cont.shape
    points = largest_cont.reshape([num, cols])

    idx = np.linspace(0, num, sample_num, endpoint=False, dtype=int)
    sample_points = points[idx, :]

    return sample_points

def smpl_reconsturction(betas, poses, smpl_model: SMPLModel, batch_size = 512):
    smpl_dataset = SMPLDataset(beta_arr=betas, pose_arr=poses, device=smpl_model.device)
    smpl_dataloader = DataLoader(smpl_dataset, batch_size=batch_size, shuffle=False)

    smpl_meshs = []
    pbar = tqdm(smpl_dataloader)
    for parameters_batch in pbar:
        batch_vertices, _ = smpl_parameters_batch_to_vertices_and_joints(parameters_batch, smpl_model)
        
        for vertices in batch_vertices:
            smpl_meshs.append(vertices)
    pbar.close()

    return smpl_meshs

def main(conf):
    random.seed(conf['seed'])

    betas = np.load(conf['betas_path'])
    per_beta = conf['pose']['per_beta']
    betas = np.repeat(betas, per_beta, 0)

    pose_type = conf['pose']['type']
    if pose_type == 'random_FL':
        poses = get_random_A_pose(len(betas), True)
    elif pose_type == 'random_L':
        poses = get_random_A_pose(len(betas), False)
    elif pose_type == 'A-pose':
        poses = get_A_pose(len(betas))
    elif pose_type== 'T-pose':
        poses = get_T_pose(len(betas))
    else:
        raise Exception(f'{pose_type}: It is invalid pose.')
    
    # smpl reconstruction

    torch_device = torch.device('cuda')
    smpl_model = SMPLModel(device=torch_device, model_path=conf['smpl_reconstruction']['model_path'])
    
    meshs = smpl_reconsturction(betas, poses, smpl_model)
    faces = smpl_model.faces

    # make silhuoettes & sample points
    args = [(i, v, faces, conf) for i, v in enumerate(meshs)]

    with Pool() as pool:
        output_list = list(tqdm(pool.imap(worker, args), total = len(args)))
    
    # save dataset npz
    save_output = {}
    save_conf = conf['npz']
    if 'beta' in save_conf:
        save_output['betas'] = betas
    if 'pose' in save_conf:
        save_output['poses'] = poses
        
    if 'sample_point' in save_conf:
        sample_points = np.array(list(map(lambda x: x[0], output_list)))
        save_output['frontal_sample_points'] = sample_points[:, 0]
        save_output['lateral_sample_points'] = sample_points[:, 1]

    if 'vertices' in save_conf:
        save_output['vertices'] = np.array(meshs)

    np.savez_compressed('dataset.npz', **save_output)
    dataset = np.load('dataset.npz')

    for k in dataset:
        print(k, dataset[k].shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSE dataset generation')   # read config file
    
    parser.add_argument('config_filename', metavar='F', help='a config file for generation', type=str)

    args = parser.parse_args()
    config_filename = args.config_filename

    conf = load_yaml(config_filename)
    main(conf)

