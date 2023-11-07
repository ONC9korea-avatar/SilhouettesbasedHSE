import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model.regressCNN import RegressionPCA
from SMPL.smpl_torch_batch import SMPLModel
import torch

######################################################
# 1. 카메라 정면 이미지, 측면 이미지 load (png 파일)
######################################################
original_img_frontal = cv2.imread('frontal_image_path.png', cv2.IMREAD_GRAYSCALE)
original_img_lateral = cv2.imread('lateral_image_path.png', cv2.IMREAD_GRAYSCALE)

######################################################
# 2. original_img에 대해 binary segmentation
######################################################
_, seg_img_frontal = cv2.threshold(original_img_frontal, 127, 255, cv2.THRESH_BINARY)
_, seg_img_lateral = cv2.threshold(original_img_lateral, 127, 255, cv2.THRESH_BINARY)

######################################################
# 3. seg_img에 대해 contour를 추출하여 좌표를 머리 꼭대기의 좌표부터 반시계방향으로 정렬
######################################################
contours_frontal, _ = cv2.findContours(seg_img_frontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_lateral, _ = cv2.findContours(seg_img_lateral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

######################################################
# Assuming the largest contour is our target contour
######################################################
contour_frontal = max(contours_frontal, key=cv2.contourArea)
contour_lateral = max(contours_lateral, key=cv2.contourArea)

######################################################
# Sort the points in anti-clockwise order
######################################################
contour_frontal = sorted(contour_frontal, key=lambda x: (-x[0][1], x[0][0]))
contour_lateral = sorted(contour_lateral, key=lambda x: (-x[0][1], x[0][0]))

net_input_frontal = np.array(contour_frontal).reshape(-1, 2)
net_input_lateral = np.array(contour_lateral).reshape(-1, 2)

######################################################
# 4. from model.regressCNN import RegressionPCA 에서 받아온 net에 위 net_input을 입력으로 주고 inference
######################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = RegressionPCA(10+(6890*3)+(24*3)).to(device)

# Loading the pretrained model (assuming you have the path)

model.load_state_dict(torch.load('path_to_model.ckpt'))
model.eval()

with torch.no_grad():
    outputs = model(torch.tensor(net_input_frontal).float().to(device), torch.tensor(net_input_lateral).float().to(device))

######################################################
# 5. 위에서 나온 결과에서 outputs[:,:10], 즉 beta를 SMPL 모델로 통과시켜서 나온 vertex를 visualization
######################################################
beta = outputs[:, :10]
smpl_model = SMPLModel(device=device, model_path='path_to_SMPL_model.pkl')
vertices, _ = smpl_model(beta, None, None)
# Here, visualize your vertices as required

# ##############################
# 6. 위에서 나온 결과에서 outputs[:, 10:10+20670], 즉 vertex를 visualization
# ##############################
vertices_from_output = outputs[:, 10:10+20670].reshape(-1, 6890, 3)
# Visualize these vertices as well

# ##############################
# 7. 위의 결과와 GT에 대한 vertex 오차 비교
# ##############################
GT_vertices = None  # Load your ground truth vertices here
error = np.linalg.norm(vertices_from_output.cpu().numpy() - GT_vertices, axis=2)
mean_error = error.mean()
print("Mean Error:", mean_error)
