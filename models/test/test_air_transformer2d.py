"""

"""
import torch
import inspect
from utils.memory_tracking.mem_tracker import MemTracker
from models.jointnet.air_transformer2d import Encoder2D, AirTransformer2D


def test_encoder2d():
    encoder = Encoder2D(256, 256, 3, [2, 2, 2])
    inputs = torch.randn(2, 1, 3, 256, 256)
    embeddings = encoder(inputs)
    for i in range(len(embeddings)):
        print(embeddings[i].shape)

def test_airtransformer2d():
    gpu_tracker = MemTracker(inspect.currentframe())

    gpu_tracker.track()
    transformer = AirTransformer2D(4).cuda()
    imgs = torch.randn(2, 3, 256, 256).cuda()
    scopes = torch.ones(2, 1, 256, 256).cuda()
    target = torch.randn(2, 4, 256, 256).cuda()

    masks = transformer(imgs, scopes)

    gpu_tracker.track()

    del transformer
    del imgs
    del scopes
    del masks
    del target
    torch.cuda.empty_cache()

    gpu_tracker.track()
