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
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)  # Change stride and padding
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),  # Change stride and padding
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(x)  # Apply downsample to the residual
        out += residual
        out = self.relu(out)
        return out


class ResnetPCA_small(nn.Module):
    def __init__(self, len_out):
        super(ResnetPCA_small, self).__init__()

        hidden1 = 256 *2*2
        hidden2 = 128*2*2
        num_keypoints = 650 # Changed to match input size
        c1, c2, c3, c4, c5 = (32, 64, 96, 96, 72)  # Channels have been halved

        # front
        self.layer11 = ResBlock(2, c1)
        self.layer12 = ResBlock(c1, c2)
        self.layer13 = ResBlock(c2, c3)
        # side
        self.layer21 = ResBlock(2, c1)
        self.layer22 = ResBlock(c1, c2)
        self.layer23 = ResBlock(c2, c3)
        # merge
        self.layer31 = ResBlock(2*c1, c2) # Input channels doubled
        self.layer32 = ResBlock(2*c2, c3) # Input channels doubled
        self.layer33 = ResBlock(2*c3, c4) # Input channels doubled

        self.fc = nn.Sequential(
            nn.Linear(num_keypoints*c4, hidden1), 
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, len_out)                             
        )

    def forward(self, input1, input2): # feed data
        f1 = self.layer11(input1)
                #print("Size of f1:", f1.size())
        s1 = self.layer21(input2)
        #print("Size of s1:", s1.size())
        fs1 = self.layer31(torch.cat((f1, s1), 1))  # Concatenate the features
        #print("Size of fs1:", fs1.size())

        f2 = self.layer12(f1)
        #print("Size of f2:", f2.size())
        s2 = self.layer22(s1)
        #print("Size of s2:", s2.size())
        fs2 = self.layer32(torch.cat((f2, s2), 1))  # Concatenate the features
        #print("Size of fs2:", fs2.size())

        f3 = self.layer13(f2)
        #print("Size of f3:", f3.size
        s3 = self.layer23(s2)
        #print("Size of s3:", s3.size())
        fs3 = self.layer33(torch.cat((f3, s3), 1))  # Concatenate the features
        #print("Size of fs3:", fs3.size())

        fs4 = fs3.view(fs3.shape[0], -1)
        #print("Size of fs4:", fs4.size())
        out = self.fc(fs4)
        #print("Size of out:", out.size())

        return out


if __name__ == "__main__":
    front_e_ith = torch.randn(64, 2, 650)  # Batch size = 64, Channels = 2, Length = 8
    side_e_ith = torch.randn(64, 2, 650)   # Batch size = 64, Channels = 2, Length = 8

    model = ResnetPCA_small(10)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    front_e_ith = front_e_ith.to(device)
    side_e_ith = side_e_ith.to(device)


    outputs = model(front_e_ith, side_e_ith)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters :", total_params)

    print(outputs)

