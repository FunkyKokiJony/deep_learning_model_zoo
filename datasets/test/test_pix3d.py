"""

"""
import logging
import pytest
import torch
from torchvision.transforms import transforms

from configuration import constants
from datasets.pix3d import LoadImgToResizeTensor, LoadMaskToResizeTensor, LoadPointModel, Pix3D, SplitDictToTensors
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
    composed = transforms.Compose([LoadImgToResizeTensor((227, 227)),
                                   LoadMaskToResizeTensor((227, 227)),
                                   LoadPointModel(2048),
                                   SplitDictToTensors(3)])
    pix3d = Pix3D(transform=composed)
    pix3d_loader = torch.utils.data.DataLoader(pix3d, batch_size=4, shuffle=True, num_workers=4)
    net = PointnetCls(input_channels=0, num_classes=40, use_xyz=True)
    checkpoint_200_path = constants.RESOURCES_ADDR + \
                          "/checkpoints/pointnet2_cls_200_epoch_modelnet40_pretrained.pth.tar"
    checkpoint_200 = torch.load(checkpoint_200_path)
    net.load_state_dict(checkpoint_200["model_state"])
    net.cuda()
    for i, batch in enumerate(pix3d_loader):
        if i >= 10: break
        _, _, points_tensors, indices = batch
        pix3d.transform = None
        category = [pix3d.__getitem__(index.item())["category"] for index in indices]
        pix3d.transform = composed
        points_tensors = points_tensors.cuda()
        net.eval()
        outputs = net.forward(points_tensors)
        _, predicted = torch.max(outputs, 1)
        labels = [(modelnet40classes[predicted[idx]] + " : " + category[idx]) for idx in range(len(predicted))]
        logging.info(str(labels))
        del points_tensors
        del predicted
        del outputs

    del net
    torch.cuda.empty_cache()

