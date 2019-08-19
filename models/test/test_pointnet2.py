"""
This is the module for testing PointNet++ integration with dummy input data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pointnet2.models import Pointnet2ClsMSG as PointnetCls
from pointnet2.models import Pointnet2SemMSG as PointnetSem
from configuration import settings

def test_pointnetcls_forward():
    """
    Pass as simply dummy tensor and test Pointnet2ClsMSG model forward
    """
    net = PointnetCls(input_channels=0, num_classes=5, use_xyz=True)
    #model PointnetCls only support GPU, our test should be performed on GPU
    net.cuda()
    #model tensor shape is (B, N, hidden xyz_channels + input_channels)
    #B represents the batch size
    #N represents the number of points in the input point cloud
    #Since sampling points on input layer in this model is 512
    #, the input number of point cloud should better be larger than 512
    inputs = torch.randn(2, 4096, 3).cuda()
    target = torch.tensor([1, 2]).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    #The output shape is (B, number of classes)
    #The Cross Entropy Loss method in pytorch will transform it into 0 based class indices
    #The target should only be 0 based class indices for each batch
    output = net.forward(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

def test_pointnetcls_with_checkpoint_best():
    """
    Check if the pretrained best checkpoint can be loaded
    """
    net = PointnetCls(input_channels=0, num_classes=40, use_xyz=True)
    checkpoint_best_path = settings.RESOURCES_ADDR +\
                           "/checkpoints/pointnet2_cls_best_modelnet40_pretrained.pth.tar"
    checkpoint_best = torch.load(checkpoint_best_path)
    net.load_state_dict(checkpoint_best["model_state"])
    net.cuda()
    inputs = torch.randn(2, 4096, 3).cuda()
    target = torch.tensor([1, 2]).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    output = net.forward(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

def test_pointnetcls_with_checkpoint_200():
    """
    Check if the pretrained 200 epochs checkpoint can be loaded
    """
    net = PointnetCls(input_channels=0, num_classes=40, use_xyz=True)
    checkpoint_200_path = settings.RESOURCES_ADDR +\
                          "/checkpoints/pointnet2_cls_200_epoch_modelnet40_pretrained.pth.tar"
    checkpoint_200 = torch.load(checkpoint_200_path)
    net.load_state_dict(checkpoint_200["model_state"])
    net.cuda()
    inputs = torch.randn(2, 4096, 3).cuda()
    target = torch.tensor([1, 2]).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    output = net.forward(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
