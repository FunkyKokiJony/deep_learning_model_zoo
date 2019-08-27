"""

"""
import logging
import pytest
import torch
from torchvision.transforms import transforms

from configuration import constants
from datasets.pix3d import LoadImg, LoadMask, LoadPointModel, Pix3D, universe_collate
from pointnet2.models import Pointnet2ClsMSG as PointnetCls
from pointnet2.data import data_utils as d_utils
import numpy as np

@pytest.fixture
def modelnet40classes():
    return ["airplane", "bathtub", "bed", "bench", "bookshelf"
        , "bottle", "bowl", "car", "chair", "cone", "cup", "curtain", "desk"
        , "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard"
        , "lamp", "laptop", "mantel", "monitor", "night_stand", "person"
        , "piano", "plant", "radio", "range_hood", "sink", "sofa", "stairs", "stool"
        , "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]

def test_pix3d_with_transforms_and_pointnet2(modelnet40classes):
    composed = transforms.Compose([LoadImg(),
                                   LoadMask(),
                                   LoadPointModel(4096)])
    trans = transforms.Compose([
            d_utils.PointcloudScale(),
        ])
    pix3d = Pix3D(transform=composed)
    pix3d_loader = torch.utils.data.DataLoader(pix3d, batch_size=4, shuffle=True, num_workers=4, collate_fn=universe_collate)
    net = PointnetCls(input_channels=0, num_classes=40, use_xyz=True)
    checkpoint_200_path = constants.RESOURCES_ADDR + \
                          "/checkpoints/pointnet2_cls_200_epoch_modelnet40_pretrained.pth.tar"
    checkpoint_200 = torch.load(checkpoint_200_path)
    net.load_state_dict(checkpoint_200["model_state"])
    net.cuda()
    for idx, batch in enumerate(pix3d_loader):
        if idx >= 10: break
        batch_tensors = torch.zeros((0, 4096, 3), dtype=torch.float32)
        category = []
        for row_dict in batch:
            points = torch.from_numpy(row_dict["model"][:, 0:3]).unsqueeze(0)
            batch_tensors = torch.cat([batch_tensors, points])
            category.append(row_dict["category"])
        #batch_tensors = trans(batch_tensors)
        batch_tensors = batch_tensors.cuda()
        net.eval()
        outputs = net.forward(batch_tensors)
        _, predicted = torch.max(outputs, 1)
        labels = [(modelnet40classes[predicted[idx]] + " : " + category[idx]) for idx in range(len(predicted))]
        logging.info(str(labels))
        del batch_tensors
        del predicted
        del outputs

    del net
    torch.cuda.empty_cache()

