"""
This is the module for test DummyResNet
"""
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
from models.dummynet import dummy_resnet
from utils.memory_tracking.mem_tracker import MemTracker


def test_forward():
    """
    Pass a simple dummy tensor and test if there is any error when forward
    """
    gpu_tracker = MemTracker(inspect.currentframe())

    gpu_tracker.track()
    net = dummy_resnet.DummyResNet()
    inputs = torch.randn(2, 3, 224, 224)
    target = torch.tensor([1, 2])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    output = net.forward(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    gpu_tracker.track()
