import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResnetPCA_mini(nn.Module):
    def __init__(self, len_out):
        super(ResnetPCA_mini, self).__init__()

        hidden1 = 256 
        hidden2 = 128
        num_keypoints = 650
        c1, c2, c3, c4, c5 = (32, 64, 96, 96, 72)

        self.layer11 = ResBlock(2, c1)
        self.layer12 = ResBlock(c1, c2)
        self.layer21 = ResBlock(2, c1)
        self.layer22 = ResBlock(c1, c2)
        self.layer31 = ResBlock(2*c1, c2)

        self.fc = nn.Sequential(
            nn.Linear(num_keypoints*c2, hidden1), 
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, len_out)                             
        )

    def forward(self, input1, input2):
        f1 = self.layer11(input1)
        s1 = self.layer21(input2)
        fs1 = self.layer31(torch.cat((f1, s1), 1))

        f2 = self.layer12(f1)
        s2 = self.layer22(s1)
        fs2 = self.layer31(torch.cat((f2, s2), 1))

        fs3 = torch.cat((fs1, fs2), 1)

        fs4 = fs3.view(fs3.shape[0], -1)
        out = self.fc(fs4)

        return out


if __name__ == "__main__":
    front_e_ith = torch.randn(64, 2, 650)
    side_e_ith = torch.randn(64, 2, 650)

    model = ResnetPCA_mini(22)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    front_e_ith = front_e_ith.to(device)
    side_e_ith = side_e_ith.to(device)

    outputs = model(front_e_ith, side_e_ith)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters :", total_params)

    print(outputs)

