"""
This is the module for test DummyCNN
"""
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from models.dummynet import dummy_cnn


class TestDummyCNN(unittest.TestCase):
    """
    This the unit test class for test DummyCNN basic forward
    """
    def test_forward(self):
        """
        Pass a simple dummy tensor and test if there is any error when forward
        """
        net = dummy_cnn.DummyCNN()
        inputs = torch.randn(2, 3, 227, 227)
        target = torch.tensor([1, 2])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        optimizer.zero_grad()
        output = net.forward(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    unittest.main()
