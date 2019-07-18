import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from models.dummynet import DummyResNet

class TestDummyResNet(unittest.TestCase):

    def testForward(self):
        net = DummyResNet.DummyResNet()
        input = torch.randn(2, 3, 224, 224)
        target = torch.tensor([1, 2])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        optimizer.zero_grad()
        output = net.forward(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if __name__ == "__main__":
        unittest.main()