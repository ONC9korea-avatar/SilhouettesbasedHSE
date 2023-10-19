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
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)  # Additional Conv layer
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)  # Additional Conv layer
        out = self.bn3(out)
        residual = self.downsample(x) 
        out += residual
        out = self.relu(out)
        return out


class ResnetPCA_wide(nn.Module):
    def __init__(self, len_out):
        super(ResnetPCA_wide, self).__init__()

        hidden1 = 2*512 
        hidden2 = 512
        num_keypoints = 650 
        c1, c2, c3, c4, c5, c6 = (64, 128, 192, 192, 144, 144)  # Added one more layer dimension

        self.layer11 = ResBlock(2, c1)
        self.layer12 = ResBlock(c1, c2)
        self.layer13 = ResBlock(c2, c3)
        self.layer14 = ResBlock(c3, c4)
        self.layer15 = ResBlock(c4, c5)  # Additional layer

        self.layer21 = ResBlock(2, c1)
        self.layer22 = ResBlock(c1, c2)
        self.layer23 = ResBlock(c2, c3)
        self.layer24 = ResBlock(c3, c4)
        self.layer25 = ResBlock(c4, c5)  # Additional layer

        self.layer31 = ResBlock(2*c1, c2)
        self.layer32 = ResBlock(2*c2, c3)
        self.layer33 = ResBlock(2*c3, c4)
        self.layer34 = ResBlock(2*c4, c5)
        self.layer35 = ResBlock(2*c5, c6)  # Additional layer

        self.fc = nn.Sequential(
            nn.Linear(num_keypoints*c6, hidden1),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout Layer
            nn.Linear(hidden1,            hidden2),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout Layer
            nn.Linear(hidden2, len_out)                             
        )

    def forward(self, input1, input2): # feed data
        f1 = self.layer11(input1)
        s1 = self.layer21(input2)
        fs1 = self.layer31(torch.cat((f1, s1), 1))  

        f2 = self.layer12(f1)
        s2 = self.layer22(s1)
        fs2 = self.layer32(torch.cat((f2, s2), 1)) 

        f3 = self.layer13(f2)
        s3 = self.layer23(s2)
        fs3 = self.layer33(torch.cat((f3, s3), 1))  

        f4 = self.layer14(f3)
        s4 = self.layer24(s3)
        fs4 = self.layer34(torch.cat((f4, s4), 1)) 

        f5 = self.layer15(f4)  # Additional layer
        s5 = self.layer25(s4)  # Additional layer
        fs5 = self.layer35(torch.cat((f5, s5), 1))  # Additional layer

        fs6 = fs5.view(fs5.shape[0], -1)  # Flatten before FC layers
        out = self.fc(fs6)

        return out


if __name__ == "__main__":
    front_e_ith = torch.randn(64, 2, 650)  # Batch size = 64, Channels = 2, Length = 8
    side_e_ith = torch.randn(64, 2, 650)   # Batch size = 64, Channels = 2, Length = 8

    model = ResnetPCA_wide(22) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    front_e_ith = front_e_ith.to(device)
    side_e_ith = side_e_ith.to(device)

    outputs = model(front_e_ith, side_e_ith)
    print(outputs)

