"""

"""
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize

from configuration.config import Config
from configuration.constants import DatasetConfig
import pandas as pd
from PIL import Image
import numpy as np
import re
from pyntcloud import PyntCloud as pc
import torch

class LoadImgToResizeTensor:
    def __init__(self, size):
        self.size = size

    def __call__(self, row_dict):
        img_dir = row_dict[DatasetConfig.PIX3D_DIR] + "/" + row_dict["img"]
        with Image.open(img_dir) as img:
            #some image in data set has 4th channel, convert to RGB
            rgb_img = img.convert("RGB")
            row_dict["img"] = ToTensor()(Resize(self.size)(rgb_img))

        return row_dict

class LoadMaskToResizeTensor:
    def __init__(self, size):
        self.size = size

    def __call__(self, row_dict):
        mask_dir = row_dict[DatasetConfig.PIX3D_DIR] + "/" + row_dict["mask"]
        with Image.open(mask_dir) as mask:
            row_dict["mask"] = ToTensor()(Resize(self.size)(mask))

        return row_dict

class LoadPointModel:
    def __init__(self, num_points=10000):
        self.num_points = num_points

    def __call__(self, row_dict):
        ply_point_dir = row_dict[DatasetConfig.PIX3D_DIR] + "/" + row_dict["model"]
        ply_point_dir = re.sub(r".obj$", "_sampled.ply", ply_point_dir)
        ply_point = pc.from_file(ply_point_dir)
        #This transform will choose a small number of points in model to save time
        #The number of points selected is define by num_points
        choices = np.random.choice(len(ply_point.points), min(len(ply_point.points), self.num_points))
        point_list = np.empty((0, 6), dtype='f')
        for i in choices:
            #In the Pix3D dataset, the ply point cloud model has 6 columns
            #The column names are [x, y, z, nx, ny, nz]
            #The first three is the coordinate and
            # the last three is the direction on the mesh of each point
            point_list = np.append(point_list, np.array([ply_point.points.iloc[i].to_list()], dtype='f'), axis=0)
        row_dict["model"] = point_list
        return row_dict

class SplitDictToTensors:
    def __init__(self, input_channels=3):
        self.input_channels = input_channels

    def __call__(self, row_dict):
        points = torch.from_numpy(row_dict["model"][:, 0:self.input_channels])
        img = row_dict["img"]
        mask = row_dict["mask"]
        idx = row_dict["idx"]
        return img, mask, points, idx

def universe_collate(batch):
    return batch

class Pix3D(Dataset):
    """
    Pix3D 3D model reconstruction from 2D dataset.
    You need to download the Pix3D dataset and set up path in configuration before use
    """
    def __init__(self, root=None, transform=None):
        self.cfg = Config()
        if root is not None:
            self.pix3d_dir = root
        else:
            self.pix3d_dir = self.cfg.get(DatasetConfig.SECTION
                                      , DatasetConfig.PIX3D_DIR)
        self.data = pd.read_json(self.pix3d_dir + "/pix3d.json")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row_dict = self.data.iloc[idx].to_dict()
        row_dict[DatasetConfig.PIX3D_DIR] = self.pix3d_dir
        row_dict["idx"] = idx
        if self.transform is not None:
            return self.transform(row_dict)

        return row_dict