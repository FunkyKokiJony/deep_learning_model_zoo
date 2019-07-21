"""
This module is for DummyCNN pytorch implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyCNN(nn.Module):
    """
    This class is based on AlexNet and tuned the final fully connected layer with 10 outputs
    The input tensor is required as [batch size, 3, 227, 227]
    """
    def __init__(self):
        super(DummyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(96, 256, 5, padding=2)
        self.pool4 = nn.MaxPool2d(3, 2)
        self.conv5 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv6 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv7 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv8 = nn.Conv2d(384, 256, 3, padding=1)
        self.pool9 = nn.MaxPool2d(3, 2)
        self.fc10 = nn.Linear(6 * 6 * 256, 4096)
        self.fc11 = nn.Linear(4096, 4096)
        self.fc12 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool9(x)
        x = torch.flatten(x, start_dim=1) # flatten into batch # of vectors
        x = self.fc12(self.fc11(self.fc10(x)))

        return x
