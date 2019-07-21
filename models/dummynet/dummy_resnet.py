"""
This module is for ResNet50 dummy implementation
"""
import torch
import torch.nn as nn

class Residual2d(nn.Module):
    """
    1. This class is used for creating Residual block
    2. downsample is a boolean to determine if we need to downsample
        the input to match the size of residual block
        Since the size change in residual block is caused by the first 1x1 Conv
        , we can use the same parameter of that layers
    3. we use the 3 layer residual is because it can reduce the number of parameters
    """
    def __init__(self, in_channels, mid_channels, out_channels
                 , kernel_size, stride=1, downsample=False):
        super(Residual2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=stride)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_bn = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size
                               , padding=(kernel_size - 1) // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2_bn = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3_bn = nn.BatchNorm2d(out_channels)
        if downsample:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.conv3_bn(out)

        if self.downsample is not None:
            out += self.downsample(identity)
        else:
            out += identity
        out = self.relu3(out)

        return out



class DummyResNet(nn.Module):
    """
    This class is ResNet50 implementation, the number of outputs is tuned to be 10
    The input tensor size is required as [batch, 3, 224, 224]
    """
    def __init__(self):
        super(DummyResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv_block2 = Residual2d(64, 64, 256, 3, 1, True)
        self.identity21 = Residual2d(256, 64, 256, 3)
        self.identity22 = Residual2d(256, 64, 256, 3)

        self.conv_block3 = Residual2d(256, 128, 512, 3, 2, True)
        self.identity31 = Residual2d(512, 128, 512, 3)
        self.identity32 = Residual2d(512, 128, 512, 3)
        self.identity33 = Residual2d(512, 128, 512, 3)

        self.conv_block4 = Residual2d(512, 256, 1024, 3, 2, True)
        self.identity41 = Residual2d(1024, 256, 1024, 3)
        self.identity42 = Residual2d(1024, 256, 1024, 3)
        self.identity43 = Residual2d(1024, 256, 1024, 3)
        self.identity44 = Residual2d(1024, 256, 1024, 3)
        self.identity45 = Residual2d(1024, 256, 1024, 3)

        self.conv_block5 = Residual2d(1024, 512, 2048, 3, 2, True)
        self.identity51 = Residual2d(2048, 256, 2048, 3)
        self.identity52 = Residual2d(2048, 256, 2048, 3)

        #The stride for pool6 will be set to the same as kernel_size (default)
        self.pool6 = nn.AvgPool2d(2)
        self.fc6 = nn.Linear(3 * 3 * 2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv_block2(x)
        x = self.identity21(x)
        x = self.identity22(x)

        x = self.conv_block3(x)
        x = self.identity31(x)
        x = self.identity32(x)
        x = self.identity33(x)

        x = self.conv_block4(x)
        x = self.identity41(x)
        x = self.identity42(x)
        x = self.identity43(x)
        x = self.identity44(x)
        x = self.identity45(x)

        x = self.conv_block5(x)
        x = self.identity51(x)
        x = self.identity52(x)

        x = self.pool6(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc6(x)

        return x
