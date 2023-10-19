import torch
import torch.nn as nn

from regressCNN import RegressionPCA
from resnetCNN_small import ResnetPCA_small

model1 = RegressionPCA(len_out=9)
total_params_model1 = sum(p.numel() for p in model1.parameters())
print("Total Parameters in RegressionPCA:", total_params_model1)

model2 = ResnetPCA_small(len_out=10)
total_params_model2 = sum(p.numel() for p in model2.parameters())
print("Total Parameters in ResnetPCA_small:", total_params_model2)

print('times : ' , total_params_model2/ total_params_model1)