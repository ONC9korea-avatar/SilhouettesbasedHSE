import os
import numpy as np

from PIL import Image
import numpy as np

import cv2 as cv

import torch
from model.regressCNN import RegressionPCA
from SMPL.smpl_torch_batch import SMPLModel
from obj_utils.misc import *
from obj_utils.io import *

def repeat_data(x):
    row, col = x.shape
    data = np.zeros([row, col+2])
    data[:, 0] = x[:, -1]
    data[:, 1:-1] = x
    data[:, -1] = x[:, 0]
    return data

def transform(x):
    return repeat_data(x.transpose())

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

def infer(frontal, lateral):
    frontal_tensor = torch.from_numpy(frontal).float().to(device)
    lateral_tensor = torch.from_numpy(lateral).float().to(device)

    # 차원 추가: [num_samples, 2] -> [1, 2, num_samples]
    # 여기서 1은 배치 크기, 2는 채널(여기서는 x, y 좌표)
    frontal_tensor = frontal_tensor.unsqueeze(0)
    lateral_tensor = lateral_tensor.unsqueeze(0)
    with torch.no_grad():
        out = infer_model(frontal_tensor, lateral_tensor)
    
    return out

# Number of sample points
num_sample_points = 648

# Load images
frontal_image = Image.open("frontal.png")
lateral_image = Image.open("lateral.png")

# Get sample points for each image
frontal_sample_points = get_sample_points(frontal_image, num_sample_points)
lateral_sample_points = get_sample_points(lateral_image, num_sample_points)


f = transform(frontal_sample_points)
l = transform(lateral_sample_points)

print(f.shape)
print(l.shape)


#########################################################
####################### Inference #######################
#########################################################

cuda_available = torch.cuda.is_available()
print('cuda available:', cuda_available)

device = torch.device('cuda:0' if cuda_available else 'cpu')
print('using device', device)

infer_model_name = 'Oct_27_02:02:00_2023'

infer_model_path = f'./checkpoints/{infer_model_name}/RegressionPCA/epochs_1000.ckpt'

results_path_for_config = f'./demo/{infer_model_name}'

infer_model = RegressionPCA(10+(6890*3)+(24*3)).to(device)
infer_model.load_state_dict(torch.load(infer_model_path))
_ = infer_model.eval()




smpl_model_path = './SMPL/model.pkl'
smpl_model = SMPLModel(device=torch.device('cuda'), model_path=smpl_model_path)

outputs = infer(f, l)

out_beta, out_vertex, out_joint = outputs[:, :10], outputs[:, 10:10+20670], outputs[:, 10+20670:]

trans = np.zeros((1, 3))
trans_tensor = torch.from_numpy(trans).type(torch.float64).to(device)



out_beta = torch.from_numpy(np.vstack(out_beta.to('cpu'))).type(torch.float64).to(device)
out_joint = torch.from_numpy(np.vstack(out_joint.to('cpu'))).type(torch.float64).to(device)
#trans_tensor = trans_tensor.float()

out_beta_np = out_beta.cpu().numpy()
print(out_beta_np)

np.save('out_beta_Doo.npy', out_beta_np)

smpl_out, _ = smpl_model(out_beta, out_joint, trans_tensor)

out_vertex_cpu = out_vertex.cpu()
out_vertex_reshaped = out_vertex_cpu.view(-1, 3)

smpl_out = smpl_out.cpu()
smpl_out_reshaped = smpl_out.squeeze(0)

print(out_vertex_reshaped.shape)
print(smpl_out_reshaped.shape)

v_out_vertex = add_vertices_color(out_vertex_reshaped, [1., 0., 0.,])
v_smpl_vertex = add_vertices_color(smpl_out_reshaped, [0., 1., 0.,])

results_path = f'./demo/{infer_model_name}'

save_obj(os.path.join(results_path, f'obj/beta/vertex.obj'), v_out_vertex)
save_obj(os.path.join(results_path, f'obj/beta/smpl.obj'), v_smpl_vertex)

v_merged = merge_vertices(v_out_vertex, v_smpl_vertex)

save_obj(os.path.join(results_path, f'obj/beta/merged.obj'), v_merged)





# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from model.regressCNN import RegressionPCA
# from SMPL.smpl_torch_batch import SMPLModel
# import torch

# ######################################################
# # 1. 카메라 정면 이미지, 측면 이미지 load (png 파일)
# ######################################################
# original_img_frontal = cv2.imread('frontal_image_path.png', cv2.IMREAD_GRAYSCALE)
# original_img_lateral = cv2.imread('lateral_image_path.png', cv2.IMREAD_GRAYSCALE)

# ######################################################
# # 2. original_img에 대해 binary segmentation
# ######################################################
# _, seg_img_frontal = cv2.threshold(original_img_frontal, 127, 255, cv2.THRESH_BINARY)
# _, seg_img_lateral = cv2.threshold(original_img_lateral, 127, 255, cv2.THRESH_BINARY)

# ######################################################
# # 3. seg_img에 대해 contour를 추출하여 좌표를 머리 꼭대기의 좌표부터 반시계방향으로 정렬
# ######################################################
# contours_frontal, _ = cv2.findContours(seg_img_frontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours_lateral, _ = cv2.findContours(seg_img_lateral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ######################################################
# # Assuming the largest contour is our target contour
# ######################################################
# contour_frontal = max(contours_frontal, key=cv2.contourArea)
# contour_lateral = max(contours_lateral, key=cv2.contourArea)

# ######################################################
# # Sort the points in anti-clockwise order
# ######################################################
# contour_frontal = sorted(contour_frontal, key=lambda x: (-x[0][1], x[0][0]))
# contour_lateral = sorted(contour_lateral, key=lambda x: (-x[0][1], x[0][0]))

# net_input_frontal = np.array(contour_frontal).reshape(-1, 2)
# net_input_lateral = np.array(contour_lateral).reshape(-1, 2)

# ######################################################
# # 4. from model.regressCNN import RegressionPCA 에서 받아온 net에 위 net_input을 입력으로 주고 inference
# ######################################################
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = RegressionPCA(10+(6890*3)+(24*3)).to(device)

# # Loading the pretrained model (assuming you have the path)

# model.load_state_dict(torch.load('path_to_model.ckpt'))
# model.eval()

# with torch.no_grad():
#     outputs = model(torch.tensor(net_input_frontal).float().to(device), torch.tensor(net_input_lateral).float().to(device))

# ######################################################
# # 5. 위에서 나온 결과에서 outputs[:,:10], 즉 beta를 SMPL 모델로 통과시켜서 나온 vertex를 visualization
# ######################################################
# beta = outputs[:, :10]
# smpl_model = SMPLModel(device=device, model_path='path_to_SMPL_model.pkl')
# vertices, _ = smpl_model(beta, None, None)
# # Here, visualize your vertices as required

# # ##############################
# # 6. 위에서 나온 결과에서 outputs[:, 10:10+20670], 즉 vertex를 visualization
# # ##############################
# vertices_from_output = outputs[:, 10:10+20670].reshape(-1, 6890, 3)
# # Visualize these vertices as well

# # ##############################
# # 7. 위의 결과와 GT에 대한 vertex 오차 비교
# # ##############################
# GT_vertices = None  # Load your ground truth vertices here
# error = np.linalg.norm(vertices_from_output.cpu().numpy() - GT_vertices, axis=2)
# mean_error = error.mean()
# print("Mean Error:", mean_error)
