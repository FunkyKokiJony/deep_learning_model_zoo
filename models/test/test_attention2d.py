"""

"""
import torch
import torch.nn as nn
import torch.optim as optim
from models.jointnet.attention2d import MultiHeadAttention2D

def test_attention2d_forward():
    q = torch.randn(2, 2, 3, 224, 224)
    k = torch.randn(2, 2, 5, 226, 226)
    v = torch.randn(2, 2, 7, 228, 228)
    target = torch.randn(2, 2, 9, 222, 222)
    attention = MultiHeadAttention2D(q_in_channels=3, q_out_channels=5, q_kernel=3, q_stride=1
                                     , k_in_channels=5, k_out_channels=7, k_kernel=5, k_stride=1
                                     , v_in_channels=7, v_out_channels=9, v_kernel=7, v_stride=1
                                     , transform_kernel=1, transform_stride=1)
    output = attention(q, k, v)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(attention.parameters(), lr=0.01)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
