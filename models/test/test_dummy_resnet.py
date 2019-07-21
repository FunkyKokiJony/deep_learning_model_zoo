"""
This is the module for test DummyResNet
"""
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from models.dummynet import dummy_resnet

class TestDummyResNet(unittest.TestCase):
    """
    This the unit test class for test DummyResNet basic forward
    """
    def __init__(self):
        super().__init__()
        self.net = dummy_resnet.DummyResNet()

    def test_forward(self):
        """
        Pass a simple dummy tensor and test if there is any error when forward
        """
        inputs = torch.randn(2, 3, 224, 224)
        target = torch.tensor([1, 2])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        optimizer.zero_grad()
        output = self.net.forward(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if __name__ == "__main__":
        unittest.main()
