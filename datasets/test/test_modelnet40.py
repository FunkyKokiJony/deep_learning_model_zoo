"""

"""
import logging

import pytest
from configuration import constants

from datasets.modelnet40 import ModelNet40
from torchvision.transforms import transforms
from pointnet2.data import data_utils as d_utils
from pointnet2.models import Pointnet2ClsMSG as PointnetCls
import numpy as np
import torch

@pytest.fixture
def modelnet40classes():
    return ["airplane", "bathtub", "bed", "bench", "bookshelf"
        , "bottle", "bowl", "car", "chair", "cone", "cup", "curtain", "desk"
        , "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard"
        , "lamp", "laptop", "mantel", "monitor", "night_stand", "person"
        , "piano", "plant", "radio", "range_hood", "sink", "sofa", "stairs", "stool"
        , "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]

def test_modelnet40(modelnet40classes):
    transform = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale(),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
            d_utils.PointcloudRandomInputDropout(),
        ]
    )
    modelnet40 = ModelNet40(2048, transform=transform)

    modelnet40_loader = torch.utils.data.DataLoader(modelnet40, batch_size=4, shuffle=True, num_workers=4)
    net = PointnetCls(input_channels=0, num_classes=40, use_xyz=True)
    checkpoint_200_path = constants.RESOURCES_ADDR + \
                          "/checkpoints/pointnet2_cls_200_epoch_modelnet40_pretrained.pth.tar"
    checkpoint_200 = torch.load(checkpoint_200_path)
    net.load_state_dict(checkpoint_200["model_state"])
    net.cuda()
    for idx, batch in enumerate(modelnet40_loader):
        if idx >= 3: break
        inputs, labels = batch[0].cuda(), batch[1].cuda()
        logging.info(inputs)
        net.eval()
        outputs = net.forward(inputs)
        _, predicted = torch.max(outputs, 1)
        results = [(modelnet40classes[predicted[i]]  + " : "+ modelnet40classes[labels[i]]) for i in range(len(predicted))]
        logging.info(str(results))
        del inputs
        del predicted
        del outputs

    del net
    torch.cuda.empty_cache()
