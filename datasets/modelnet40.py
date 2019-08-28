"""

"""
import h5py
import torch
from torch.utils.data import Dataset

from configuration.config import Config
from configuration.constants import DatasetConfig
import os
import numpy as np

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]

def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label

class ModelNet40(Dataset):
    def __init__(self, num_points, root=None, transform=None, train=True):
        self.cfg = Config()
        if root is not None:
            self.modelnet40_dir = root
        else:
            self.modelnet40_dir = self.cfg.get(DatasetConfig.SECTION
                                          , DatasetConfig.MODELNET40_DIR)
        self.train = train
        self.num_points = num_points
        self.transform = transform
        if self.train:
            self.files = _get_data_files(os.path.join(self.modelnet40_dir, "train_files.txt"))
        else:
            self.files = _get_data_files(os.path.join(self.modelnet40_dir, "test_files.txt"))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(self.modelnet40_dir, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transform is not None:
            current_points = self.transform(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]
