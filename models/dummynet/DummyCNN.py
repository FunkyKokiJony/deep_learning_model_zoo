import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyCNN(nn.Module):
    def __init__(self):
        super(DummyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11,stride=4)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(96, 256, 5, padding = 2)
        self.pool4 = nn.MaxPool2d(3, 2)
        self.conv5 = nn.Conv2d(256, 384, 3, padding = 1)
        self.conv6 = nn.Conv2d(384, 384, 3, padding = 1)
        self.conv7 = nn.Conv2d(384, 384, 3, padding = 1)
        self.conv8 = nn.Conv2d(384, 256, 3, padding = 1)
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
        x = x.view(-1, self.num_flat_features(x)) # flatten into batch # of vectors
        x = self.fc12(self.fc11(self.fc10(x)))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
