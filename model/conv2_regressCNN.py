#####################################
# File Name: Conv2_regressCNN.py
# Author : Junkwang Kim
# Email : kjk1208@dgist.ac.kr
# Creation Date : 2023.12.01
# Last Modified Date : 2023.12.01
# Change Log :
# - 2023.12.01 : 1D Conv to 2D Conv
#####################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Conv2_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2_1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class MLP_liu(nn.Module):
    def __init__(self, in_channels, out_channels, size_kernel=3, size_padding=1, size_pooling=3):
        super(MLP_liu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(size_kernel, size_kernel), padding=(size_padding, size_padding), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.pooling = nn.MaxPool2d(kernel_size=(size_pooling, size_pooling), stride=(size_pooling, size_pooling))

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pooling(out)

        return out


class conv2_RegressionPCA(nn.Module):
    def __init__(self, len_out,fc1=2048, fc2=1024):
        super(conv2_RegressionPCA, self).__init__()

        hidden1 = fc1
        hidden2 = fc2
        #num_keypoints = 8 # (num of points: num_keypoints) (648, 8) Namely, (divide 81)
        c1, c2, c3, c4, c5 = (64, 128, 192, 192, 144)

        # front
        self.layer11 = Conv2_1(1, c1)
        self.layer12 = MLP_liu(c1, c2)
        self.layer13 = MLP_liu(c2, c3)
        self.layer14 = MLP_liu(c3, c4)
        # side
        self.layer21 = Conv2_1(1, c1)
        self.layer22 = MLP_liu(c1, c2)
        self.layer23 = MLP_liu(c2, c3)
        self.layer24 = MLP_liu(c3, c4)
        # merge
        self.layer31 = MLP_liu(c1, c2)  
        self.layer32 = MLP_liu(c2, c3)
        self.layer33 = MLP_liu(c3, c4)
        self.layer34 = MLP_liu(c4, c5)

        flattened_size = 144

        self.fc = nn.Sequential(
            #nn.Linear(num_keypoints*c5, hidden1), 
            nn.Linear(flattened_size, hidden1), 
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, len_out)                             
        )

    def forward(self, input1, input2): # feed data

        f1 = self.layer11(input1)
        s1 = self.layer21(input2)
        fs1 = self.layer31(f1+s1)

        f2 = self.layer12(f1)
        s2 = self.layer22(s1)
        fs2 = self.layer32(f2+s2+fs1)

        f3 = self.layer13(f2)
        s3 = self.layer23(s2)
        fs3 = self.layer33(f3+s3+fs2)

        f4 = self.layer14(f3)
        s4 = self.layer24(s3)
        fs4 = self.layer34(f4+s4+fs3)
        #print("fs4 size:", fs4.size())  # fs4의 크기 출력

        fs5 = fs4.view(fs4.shape[0], -1)
        out = self.fc(fs5)

        return out   

if __name__ == '__main__':
    len_out = 10 + 6890*3 + 24*3
    model = conv2_RegressionPCA(len_out, 2048, 1024)    

    batch_size = 32
    num_samples = batch_size
    frontal_img_data = torch.randn(num_samples, 1, 150, 125)
    lateral_img_data = torch.randn(num_samples, 1, 150, 125)

    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"{name} has\t{num_params} parameters")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Total parameters: {total_params*4/(1024**3)} GB")
    

    output = model(frontal_img_data, lateral_img_data)

    print("Output size:", output.size())  # 예상 출력 크기: [batch_size, 10 + 6890*3 + 24*3]

    # print(model)