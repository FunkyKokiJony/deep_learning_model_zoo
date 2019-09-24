"""

"""
import torch
import inspect

from torch import optim

from utils.memory_tracking.mem_tracker import MemTracker
from models.jointnet.monet import MONet


def test_monet():
    gpu_tracker = MemTracker(inspect.currentframe())

    gpu_tracker.track()
    monet = MONet().cuda()
    imgs = torch.randn(2, 3, 256, 256).cuda()
    scopes = torch.ones(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]).cuda()
    outputs = monet(imgs, scopes)
    optimizer = optim.SGD(monet.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = monet.calculate_loss(outputs, imgs, 0.5, 0.5)
    loss.backward()
    optimizer.step()

    gpu_tracker.track()
